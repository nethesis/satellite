"""Transactional application control state, independent of optional telemetry."""

import hashlib
import secrets
import time
import uuid

from psycopg.types.json import Jsonb

from agent.monitoring.repository import HistoryRepository
from .contracts import ApplicationError, canonical, digest

MAX_RECORDS = 100000
MAX_CONTENT = 256 * 1024 * 1024
TERMINAL = ("completed", "failed", "cancelled", "interrupted")
DDL = """
CREATE SCHEMA IF NOT EXISTS agent_application;
CREATE TABLE IF NOT EXISTS agent_application.migrations(version integer PRIMARY KEY);
INSERT INTO agent_application.migrations VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS agent_application.settings(
 id integer PRIMARY KEY CHECK(id=1), enabled boolean NOT NULL DEFAULT false);
INSERT INTO agent_application.settings(id) VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS agent_application.resources(
 kind varchar(16), resource_id varchar(48), revision bigint NOT NULL,
 draft jsonb NOT NULL, published_version integer NOT NULL DEFAULT 0,
 PRIMARY KEY(kind,resource_id));
CREATE TABLE IF NOT EXISTS agent_application.versions(
 kind varchar(16), resource_id varchar(48), version integer,
 definition jsonb NOT NULL, revoked boolean NOT NULL DEFAULT false,
 PRIMARY KEY(kind,resource_id,version));
CREATE TABLE IF NOT EXISTS agent_application.secrets(
 secret_id varchar(48) PRIMARY KEY, ciphertext bytea NOT NULL,
 revoked boolean NOT NULL DEFAULT false);
CREATE TABLE IF NOT EXISTS agent_application.grants(
 agent_id varchar(48) PRIMARY KEY, revision bigint NOT NULL, operations jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS agent_application.clients(
 client_id varchar(48) PRIMARY KEY, verifier varchar(64) UNIQUE NOT NULL,
 definition jsonb NOT NULL, expires double precision NOT NULL,
 revoked boolean NOT NULL DEFAULT false);
CREATE TABLE IF NOT EXISTS agent_application.runs(
 run_id varchar(64) PRIMARY KEY, client_id varchar(128) NOT NULL,
 idempotency_hash varchar(64) NOT NULL, request_digest varchar(64) NOT NULL,
 preset_version integer NOT NULL, revision bigint NOT NULL,
 status varchar(16) NOT NULL DEFAULT 'accepted', epoch varchar(64),
 started double precision NOT NULL, ended double precision,
 error_code varchar(64), cancel_requested boolean NOT NULL DEFAULT false,
 reconciliation_required boolean NOT NULL DEFAULT false,
 input bytea, snapshot bytea, result bytea, result_expires double precision,
 result_hours integer NOT NULL, metadata jsonb NOT NULL,
 expires double precision NOT NULL,
 UNIQUE(client_id,idempotency_hash));
CREATE INDEX IF NOT EXISTS application_run_status ON agent_application.runs(status,started);
CREATE TABLE IF NOT EXISTS agent_application.effects(
 operation_id varchar(64) PRIMARY KEY, run_id varchar(64) NOT NULL,
 effect_key varchar(128) NOT NULL, input_digest varchar(64) NOT NULL,
 connector_id varchar(48) NOT NULL, version integer NOT NULL,
 operation varchar(48) NOT NULL, state varchar(16) NOT NULL,
 response bytea, result_expires double precision, error_code varchar(64),
 created double precision NOT NULL, updated double precision NOT NULL,
 UNIQUE(run_id,effect_key));
CREATE TABLE IF NOT EXISTS agent_application.audit(
 id bigserial PRIMARY KEY, actor varchar(128) NOT NULL,
 action varchar(48) NOT NULL, resource varchar(128) NOT NULL,
 timestamp double precision NOT NULL);
"""


class ApplicationRepository(HistoryRepository):
    def audit(self, db, actor, action, resource):
        db.execute("INSERT INTO agent_application.audit(actor,action,resource,timestamp) VALUES(%s,%s,%s,%s)",
                   (actor, action, resource, time.time()))
        db.execute("DELETE FROM agent_application.audit WHERE id <= "
                   "(SELECT max(id) FROM agent_application.audit)-%s", (MAX_RECORDS,))

    def record_audit(self, actor, action, resource):
        with self.connect() as db:
            self.audit(db, actor, action, resource)

    def initialize(self, epoch):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783504)")
            for sql in DDL.split(";"):
                if sql.strip():
                    db.execute(sql)
            if db.execute("SELECT max(version) AS v FROM agent_application.migrations").fetchone()["v"] != 1:
                raise ApplicationError("application_schema_newer", 503)
            db.execute("UPDATE agent_application.runs SET status='interrupted',error_code='runtime_restart',"
                       "ended=%s,input=NULL,snapshot=NULL WHERE status IN ('accepted','running') AND epoch IS DISTINCT FROM %s", (time.time(), epoch))
            db.execute("UPDATE agent_application.effects SET state='unknown',error_code='runtime_restart',updated=%s "
                       "WHERE state='dispatched' AND run_id IN (SELECT run_id FROM agent_application.runs WHERE status='interrupted')", (time.time(),))
            db.execute("UPDATE agent_application.runs SET reconciliation_required=true WHERE run_id IN "
                       "(SELECT run_id FROM agent_application.effects WHERE state='unknown')")

    def enabled(self):
        with self.connect() as db:
            return db.execute("SELECT enabled FROM agent_application.settings WHERE id=1").fetchone()["enabled"]

    def recover_orphans(self, epoch, active):
        """Settle durable admissions whose in-process owner could not settle them."""
        with self.connect() as db:
            rows = db.execute("UPDATE agent_application.runs SET status='interrupted',error_code='owner_lost',"
                "ended=%s,input=NULL,snapshot=NULL WHERE epoch=%s AND status IN ('accepted','running') "
                "AND NOT (run_id=ANY(%s::varchar[])) RETURNING run_id", (time.time(), epoch, list(active))).fetchall()
            ids = [row["run_id"] for row in rows]
            if ids:
                db.execute("UPDATE agent_application.effects SET state='unknown',error_code='owner_lost',updated=%s "
                           "WHERE state='dispatched' AND run_id=ANY(%s::varchar[])", (time.time(), ids))
                db.execute("UPDATE agent_application.runs SET reconciliation_required=true WHERE run_id=ANY(%s::varchar[]) "
                           "AND run_id IN (SELECT run_id FROM agent_application.effects WHERE state='unknown')", (ids,))

    def set_enabled(self, enabled, actor):
        with self.connect() as db:
            db.execute("UPDATE agent_application.settings SET enabled=%s WHERE id=1", (enabled,))
            self.audit(db, actor, "access_enabled" if enabled else "access_disabled", "settings")
        return {"enabled": enabled}

    def inventory(self):
        with self.connect() as db:
            resources = db.execute("SELECT * FROM agent_application.resources ORDER BY kind,resource_id").fetchall()
            clients = db.execute("SELECT client_id,definition,expires,revoked FROM agent_application.clients ORDER BY client_id").fetchall()
            grants = db.execute("SELECT * FROM agent_application.grants ORDER BY agent_id").fetchall()
            keys = db.execute("SELECT secret_id,revoked FROM agent_application.secrets ORDER BY secret_id").fetchall()
            versions = db.execute("SELECT kind,resource_id,version,revoked,definition FROM agent_application.versions ORDER BY kind,resource_id,version").fetchall()
            enabled = db.execute("SELECT enabled FROM agent_application.settings WHERE id=1").fetchone()["enabled"]
            effects = db.execute("SELECT operation_id,run_id,connector_id,version,operation,state,error_code,created "
                                 "FROM agent_application.effects WHERE state='unknown' ORDER BY created DESC LIMIT 100").fetchall()
        return {"enabled": enabled, "resources": resources, "clients": clients, "grants": grants,
                "secrets": keys, "versions": versions, "unresolved_effects": effects}

    def save(self, kind, resource_id, definition, expected, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783505)")
            row = db.execute("SELECT * FROM agent_application.resources WHERE kind=%s AND resource_id=%s FOR UPDATE", (kind, resource_id)).fetchone()
            if (row["revision"] if row else 0) != expected:
                raise ApplicationError("revision_conflict", 409)
            if row is None and db.execute("SELECT count(*) AS n FROM agent_application.resources WHERE kind=%s", (kind,)).fetchone()["n"] >= (20 if kind == "connector" else 1):
                raise ApplicationError("resource_capacity", 429)
            self.configuration_capacity(db, definition)
            revision = expected + 1
            db.execute("INSERT INTO agent_application.resources(kind,resource_id,revision,draft) VALUES(%s,%s,%s,%s) "
                       "ON CONFLICT(kind,resource_id) DO UPDATE SET revision=excluded.revision,draft=excluded.draft",
                       (kind, resource_id, revision, Jsonb(definition)))
            self.audit(db, actor, "draft_saved", f"{kind}:{resource_id}:{revision}")
        return {"revision": revision}

    def publish(self, kind, resource_id, expected, actor, validator):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783505)")
            row = db.execute("SELECT * FROM agent_application.resources WHERE kind=%s AND resource_id=%s FOR UPDATE", (kind, resource_id)).fetchone()
            if row is None:
                raise ApplicationError("not_found", 404)
            if row["revision"] != expected:
                raise ApplicationError("revision_conflict", 409)
            definition = validator(row["draft"])
            self.configuration_capacity(db, definition)
            secret = db.execute("SELECT revoked FROM agent_application.secrets WHERE secret_id=%s", (definition["secret_ref"],)).fetchone()
            if not secret or secret["revoked"]:
                raise ApplicationError("secret_unavailable", 409)
            if kind == "preset":
                for ref in definition["operations"]:
                    self.resolve_reference(db, ref)
            version = row["published_version"] + 1
            if version > 1000000 or db.execute("SELECT count(*) AS n FROM agent_application.versions").fetchone()["n"] >= 1000:
                raise ApplicationError("version_capacity", 429)
            db.execute("INSERT INTO agent_application.versions(kind,resource_id,version,definition) VALUES(%s,%s,%s,%s)", (kind, resource_id, version, Jsonb(definition)))
            db.execute("UPDATE agent_application.resources SET published_version=%s,revision=revision+1 WHERE kind=%s AND resource_id=%s", (version, kind, resource_id))
            self.audit(db, actor, "published", f"{kind}:{resource_id}:{version}")
        return {"version": version, "revision": expected + 1}

    def configuration_capacity(self, db, definition):
        used = db.execute("SELECT COALESCE((SELECT sum(octet_length(draft::text)) FROM agent_application.resources),0) + "
                          "COALESCE((SELECT sum(octet_length(definition::text)) FROM agent_application.versions),0) AS n").fetchone()["n"]
        if used + len(canonical(definition).encode()) > 1024 * 1024:
            raise ApplicationError("configuration_capacity", 429)

    def version(self, kind, resource_id, version):
        with self.connect() as db:
            return self._version(db, kind, resource_id, version)

    def _version(self, db, kind, resource_id, version):
        row = db.execute("SELECT definition,revoked FROM agent_application.versions WHERE kind=%s AND resource_id=%s AND version=%s", (kind, resource_id, version)).fetchone()
        if not row or row["revoked"]:
            raise ApplicationError("version_unavailable", 409)
        return row["definition"]

    def resolve_reference(self, db, ref):
        definition = self._version(db, "connector", ref["connector_id"], ref["version"])
        operation = next((o for o in definition["operations"] if o["id"] == ref["operation_id"]), None)
        if operation is None:
            raise ApplicationError("unknown_operation", 409)
        return {"reference": ref, "connector": definition, "operation": operation}

    def resolve(self, references):
        with self.connect() as db:
            return [self.resolve_reference(db, ref) for ref in references]

    def revoke_version(self, kind, resource_id, version, actor):
        with self.connect() as db:
            if db.execute("UPDATE agent_application.versions SET revoked=true WHERE kind=%s AND resource_id=%s AND version=%s RETURNING version", (kind, resource_id, version)).fetchone() is None:
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "version_revoked", f"{kind}:{resource_id}:{version}")
        return {"revoked": True}

    def add_secret(self, secret_id, value, key, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783505)")
            if db.execute("SELECT count(*) AS n FROM agent_application.secrets").fetchone()["n"] >= 100:
                raise ApplicationError("secret_capacity", 429)
            if db.execute("SELECT 1 FROM agent_application.secrets WHERE secret_id=%s", (secret_id,)).fetchone():
                raise ApplicationError("immutable_secret", 409)
            db.execute("INSERT INTO agent_application.secrets(secret_id,ciphertext) VALUES(%s,%s)", (secret_id, key.encrypt(f"secret:{secret_id}", value)))
            self.audit(db, actor, "secret_added", secret_id)
        return {"secret_id": secret_id}

    def secret(self, secret_id, key):
        with self.connect() as db:
            row = db.execute("SELECT ciphertext,revoked FROM agent_application.secrets WHERE secret_id=%s", (secret_id,)).fetchone()
        if not row or row["revoked"]:
            raise ApplicationError("secret_unavailable", 503)
        return key.decrypt(f"secret:{secret_id}", row["ciphertext"])

    def revoke_secret(self, secret_id, actor):
        with self.connect() as db:
            if not db.execute("UPDATE agent_application.secrets SET revoked=true WHERE secret_id=%s RETURNING secret_id", (secret_id,)).fetchone():
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "secret_revoked", secret_id)
        return {"revoked": True}

    def grants(self, agent_id):
        with self.connect() as db:
            row = db.execute("SELECT revision,operations FROM agent_application.grants WHERE agent_id=%s", (agent_id,)).fetchone()
        return row or {"revision": 0, "operations": []}

    def save_grants(self, agent_id, operations, expected, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783505)")
            row = db.execute("SELECT revision FROM agent_application.grants WHERE agent_id=%s FOR UPDATE", (agent_id,)).fetchone()
            if (row["revision"] if row else 0) != expected:
                raise ApplicationError("revision_conflict", 409)
            for ref in operations:
                self.resolve_reference(db, ref)
            db.execute("INSERT INTO agent_application.grants VALUES(%s,%s,%s) ON CONFLICT(agent_id) DO UPDATE SET revision=excluded.revision,operations=excluded.operations", (agent_id, expected + 1, Jsonb(operations)))
            self.audit(db, actor, "grants_saved", agent_id)
        return {"revision": expected + 1}

    def create_client(self, client_id, definition, expires, actor):
        token = "nv_agent_" + secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783505)")
            if db.execute("SELECT count(*) AS n FROM agent_application.clients").fetchone()["n"] >= 100:
                raise ApplicationError("client_capacity", 429)
            if db.execute("SELECT 1 FROM agent_application.clients WHERE client_id=%s", (client_id,)).fetchone():
                raise ApplicationError("client_exists", 409)
            db.execute("INSERT INTO agent_application.clients(client_id,verifier,definition,expires) VALUES(%s,%s,%s,%s)", (client_id, hashlib.sha256(token.encode()).hexdigest(), Jsonb(definition), expires))
            self.audit(db, actor, "client_created", client_id)
        return {"client_id": client_id, "token": token}

    def authenticate(self, token):
        with self.connect() as db:
            row = db.execute("SELECT client_id,definition,expires,revoked FROM agent_application.clients WHERE verifier=%s", (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not row or row["revoked"] or row["expires"] <= time.time():
            raise ApplicationError("unauthorized", 401)
        return row

    def client(self, client_id):
        with self.connect() as db:
            row = db.execute("SELECT client_id,definition,expires,revoked FROM agent_application.clients WHERE client_id=%s", (client_id,)).fetchone()
        if not row or row["revoked"] or row["expires"] <= time.time():
            raise ApplicationError("client_revoked", 403)
        return row

    def revoke_client(self, client_id, actor):
        with self.connect() as db:
            if not db.execute("UPDATE agent_application.clients SET revoked=true WHERE client_id=%s RETURNING client_id", (client_id,)).fetchone():
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "client_revoked", client_id)
        return {"revoked": True}

    def admit(self, client_id, idempotency, request, snapshot, epoch, key, revision, metadata):
        now = time.time(); run_id = uuid.uuid4().hex
        identity = hashlib.sha256(idempotency.encode()).hexdigest()
        request_hash = digest(request)
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783506)")
            existing = db.execute("SELECT * FROM agent_application.runs WHERE client_id=%s AND idempotency_hash=%s", (client_id, identity)).fetchone()
            if existing:
                if existing["request_digest"] != request_hash:
                    raise ApplicationError("idempotency_conflict", 409)
                return existing, False
            count = db.execute("SELECT count(*) AS n, count(*) FILTER(WHERE status IN ('accepted','running')) AS active "
                               "FROM agent_application.runs").fetchone()
            size = db.execute("SELECT COALESCE(sum(COALESCE(octet_length(input),0)+COALESCE(octet_length(snapshot),0)),0) + "
                              "COALESCE(sum(octet_length(result)),0) + COALESCE((SELECT sum(octet_length(response)) "
                              "FROM agent_application.effects),0) AS n FROM agent_application.runs").fetchone()["n"]
            encrypted_input = key.encrypt(f"input:{run_id}", request["input"])
            encrypted_snapshot = key.encrypt(f"snapshot:{run_id}", snapshot)
            # Include the actual snapshot and reserve bounded results/effect receipts.
            required = len(encrypted_input) + len(encrypted_snapshot) + (count["active"] + 1) * 131072
            if count["n"] >= MAX_RECORDS or count["active"] >= 24 or size + required >= MAX_CONTENT:
                raise ApplicationError("execution_capacity", 429)
            row = db.execute("INSERT INTO agent_application.runs(run_id,client_id,idempotency_hash,request_digest,preset_version,revision,epoch,started,input,snapshot,result_hours,metadata,expires) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
                (run_id, client_id, identity, request_hash, request["version"], revision, epoch, now,
                 encrypted_input, encrypted_snapshot, snapshot["preset"]["result_retention_hours"], Jsonb(metadata), now + 30 * 86400)).fetchone()
        return row, True

    def claim(self, run_id, epoch):
        with self.connect() as db:
            return db.execute("UPDATE agent_application.runs SET status='running' WHERE run_id=%s AND epoch=%s AND status='accepted' RETURNING *", (run_id, epoch)).fetchone()

    def run(self, run_id, client_id=None):
        with self.connect() as db:
            row = db.execute("SELECT * FROM agent_application.runs WHERE run_id=%s AND expires>%s", (run_id, time.time())).fetchone()
        if not row or (client_id is not None and row["client_id"] != client_id):
            raise ApplicationError("run_not_found", 404)
        return row

    def cancel(self, run_id, client_id):
        with self.connect() as db:
            row = db.execute("SELECT status FROM agent_application.runs WHERE run_id=%s AND client_id=%s AND expires>%s FOR UPDATE", (run_id, client_id, time.time())).fetchone()
            if not row:
                raise ApplicationError("run_not_found", 404)
            if row["status"] not in TERMINAL:
                db.execute("UPDATE agent_application.runs SET cancel_requested=true WHERE run_id=%s", (run_id,))
        return {"status": row["status"], "cancel_requested": row["status"] not in TERMINAL}

    def finish(self, run_id, status, error, result, key):
        now = time.time()
        with self.connect() as db:
            row = db.execute("SELECT status,result_hours,cancel_requested FROM agent_application.runs WHERE run_id=%s FOR UPDATE", (run_id,)).fetchone()
            if not row or row["status"] in TERMINAL:
                return
            uncertain = bool(db.execute("SELECT 1 FROM agent_application.effects WHERE run_id=%s AND state IN ('unknown','dispatched')", (run_id,)).fetchone())
            if uncertain:
                status, error = "failed", "reconciliation_required"
            elif row["cancel_requested"]:
                status, error = "cancelled", "cancelled"
            expires = now + row["result_hours"] * 3600
            db.execute("UPDATE agent_application.runs SET status=%s,error_code=%s,ended=%s,reconciliation_required=%s,input=NULL,snapshot=NULL,result=%s,result_expires=%s WHERE run_id=%s",
                       (status, error, now, uncertain, key.encrypt(f"result:{run_id}", result) if result is not None else None, expires, run_id))

    def prepare_effect(self, run_id, effect_key, args, reference):
        now = time.time(); operation_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783507)")
            row = db.execute("SELECT * FROM agent_application.effects WHERE run_id=%s AND effect_key=%s FOR UPDATE", (run_id, effect_key)).fetchone()
            if row:
                if row["input_digest"] != digest(args) or (row["connector_id"], row["version"], row["operation"]) != (reference["connector_id"], reference["version"], reference["operation_id"]):
                    raise ApplicationError("effect_conflict", 409)
                return row, False
            if db.execute("SELECT count(*) AS n FROM agent_application.effects").fetchone()["n"] >= MAX_RECORDS:
                raise ApplicationError("effect_capacity", 429)
            row = db.execute("INSERT INTO agent_application.effects(operation_id,run_id,effect_key,input_digest,connector_id,version,operation,state,created,updated) VALUES(%s,%s,%s,%s,%s,%s,%s,'prepared',%s,%s) RETURNING *",
                             (operation_id, run_id, effect_key, digest(args), reference["connector_id"], reference["version"], reference["operation_id"], now, now)).fetchone()
        return row, True

    def dispatch_effect(self, operation_id):
        with self.connect() as db:
            if not db.execute("UPDATE agent_application.effects SET state='dispatched',updated=%s WHERE operation_id=%s AND state='prepared' RETURNING operation_id", (time.time(), operation_id)).fetchone():
                raise ApplicationError("effect_conflict", 409)

    def settle_effect(self, operation_id, state, response, key, error=None):
        if state not in ("committed", "rejected", "unknown"):
            raise ApplicationError()
        with self.connect() as db:
            db.execute("UPDATE agent_application.effects SET state=%s,response=%s,result_expires=%s,error_code=%s,updated=%s WHERE operation_id=%s AND state IN ('dispatched','unknown')",
                       (state, key.encrypt(f"effect:{operation_id}", response) if response is not None else None, time.time() + 86400, error, time.time(), operation_id))
            db.execute("UPDATE agent_application.runs SET reconciliation_required=EXISTS(SELECT 1 FROM agent_application.effects e WHERE e.run_id=agent_application.runs.run_id AND e.state IN ('unknown','dispatched')) WHERE run_id=(SELECT run_id FROM agent_application.effects WHERE operation_id=%s)", (operation_id,))

    def effect(self, operation_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM agent_application.effects WHERE operation_id=%s", (operation_id,)).fetchone()
        if row is None:
            raise ApplicationError("not_found", 404)
        return row

    def effects(self, run_id):
        with self.connect() as db:
            return db.execute("SELECT operation_id,connector_id,version,operation,state,error_code FROM agent_application.effects WHERE run_id=%s ORDER BY created", (run_id,)).fetchall()

    def purge(self):
        now = time.time()
        with self.connect() as db:
            db.execute("UPDATE agent_application.runs SET result=NULL WHERE result_expires<=%s", (now,))
            db.execute("UPDATE agent_application.effects SET response=NULL WHERE result_expires<=%s", (now,))
            db.execute("DELETE FROM agent_application.runs WHERE expires<=%s AND status IN ('completed','failed','cancelled','interrupted') AND NOT EXISTS(SELECT 1 FROM agent_application.effects e WHERE e.run_id=agent_application.runs.run_id AND e.state IN ('unknown','dispatched'))", (now,))
            db.execute("DELETE FROM agent_application.effects WHERE updated<%s AND state IN ('committed','rejected','prepared') AND NOT EXISTS(SELECT 1 FROM agent_application.runs r WHERE r.run_id=agent_application.effects.run_id)", (now - 30 * 86400,))
            db.execute("DELETE FROM agent_application.audit WHERE timestamp<%s", (now - 365 * 86400,))

    def disable_clone(self):
        with self.connect() as db:
            db.execute("UPDATE agent_application.settings SET enabled=false WHERE id=1")
            db.execute("UPDATE agent_application.clients SET revoked=true")
