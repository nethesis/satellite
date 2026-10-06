"""Transactional definitions and encrypted data, separate from lossy monitoring."""

import time
import uuid

from psycopg.types.json import Jsonb

from agent.application.contracts import ApplicationError, canonical, digest
from agent.application.repository import ApplicationRepository
from .contracts import definition, execution_hash

DDL = """
CREATE SCHEMA IF NOT EXISTS agent_workflows;
CREATE TABLE IF NOT EXISTS agent_workflows.migrations(version integer PRIMARY KEY);
INSERT INTO agent_workflows.migrations VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS agent_workflows.definitions(
 kind varchar(16), agent_id varchar(48), revision bigint NOT NULL,
 draft jsonb NOT NULL, published_version integer NOT NULL DEFAULT 0,
 active_version integer NOT NULL DEFAULT 0, enabled boolean NOT NULL DEFAULT false,
 PRIMARY KEY(kind,agent_id));
CREATE TABLE IF NOT EXISTS agent_workflows.versions(
 kind varchar(16), agent_id varchar(48), version integer, definition jsonb NOT NULL,
 execution_hash varchar(64) NOT NULL, revoked boolean NOT NULL DEFAULT false,
 created double precision NOT NULL, PRIMARY KEY(kind,agent_id,version));
CREATE TABLE IF NOT EXISTS agent_workflows.data(
 resource_id varchar(48) PRIMARY KEY, revision bigint NOT NULL,
 settings jsonb NOT NULL, published_version integer NOT NULL DEFAULT 0,
 status varchar(32) NOT NULL, error_code varchar(64), last_refresh double precision);
CREATE TABLE IF NOT EXISTS agent_workflows.data_versions(
 resource_id varchar(48), version integer, metadata jsonb NOT NULL,
 ciphertext bytea NOT NULL, original bytea, created double precision NOT NULL,
 revoked boolean NOT NULL DEFAULT false, PRIMARY KEY(resource_id,version));
CREATE TABLE IF NOT EXISTS agent_workflows.executions(
 run_id varchar(128) PRIMARY KEY, agent_id varchar(48) NOT NULL, version integer NOT NULL,
 parent_run_id varchar(128), execution_kind varchar(16) NOT NULL, principal varchar(128),
 epoch varchar(64) NOT NULL, status varchar(32) NOT NULL, started double precision NOT NULL,
 ended double precision, error_code varchar(64), cancel_requested boolean NOT NULL DEFAULT false,
 definition jsonb NOT NULL, result bytea, result_expires double precision,
 idempotency_hash varchar(64), input_digest varchar(64),
 UNIQUE(principal,idempotency_hash));
CREATE TABLE IF NOT EXISTS agent_workflows.steps(
 run_id varchar(128) REFERENCES agent_workflows.executions ON DELETE CASCADE,
 sequence integer, node_id varchar(48), block_type varchar(48), status varchar(32),
 outcome varchar(48), started double precision NOT NULL, duration_ms integer,
 error_code varchar(64), PRIMARY KEY(run_id,sequence));
ALTER TABLE agent_workflows.steps ADD COLUMN IF NOT EXISTS subflow_path varchar(256) NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS workflow_execution_status ON agent_workflows.executions(status,started);
CREATE TABLE IF NOT EXISTS agent_workflows.ingestion_jobs(
 job_id varchar(32) PRIMARY KEY, resource_id varchar(48), expected_revision bigint,
 actor varchar(128), state varchar(32), input bytea, result jsonb, error_code varchar(64),
 created double precision NOT NULL);
"""


class WorkflowRepository(ApplicationRepository):
    def initialize_workflows(self, epoch, active_jobs=(), active_runs=None):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            for sql in DDL.split(";"):
                if sql.strip():
                    db.execute(sql)
            if db.execute("SELECT max(version) AS v FROM agent_workflows.migrations").fetchone()["v"] != 1:
                raise ApplicationError("workflow_schema_newer", 503)
            db.execute("UPDATE agent_workflows.executions SET status='interrupted',ended=%s,error_code='runtime_restart' "
                       "WHERE status IN ('accepted','running') AND (epoch<>%s OR (%s AND NOT(run_id=ANY(%s::varchar[]))))",
                       (time.time(), epoch, active_runs is not None, active_runs or []))
            db.execute("UPDATE agent_workflows.steps SET status='interrupted',error_code='runtime_restart' WHERE status='running' "
                       "AND run_id IN (SELECT run_id FROM agent_workflows.executions WHERE status='interrupted')")
            db.execute("UPDATE agent_application.effects SET state='unknown',error_code='runtime_restart',updated=%s "
                       "WHERE state='dispatched' AND run_id IN (SELECT run_id FROM agent_workflows.executions WHERE status='interrupted')", (time.time(),))
            db.execute("UPDATE agent_workflows.ingestion_jobs SET state='interrupted',input=NULL,error_code='runtime_restart' WHERE state IN ('queued','processing') AND NOT(job_id=ANY(%s::varchar[]))", (list(active_jobs),))

    def workflow_inventory(self):
        with self.connect() as db:
            definitions = db.execute("SELECT * FROM agent_workflows.definitions ORDER BY kind,agent_id").fetchall()
            versions = db.execute("SELECT kind,agent_id,version,execution_hash,revoked,created FROM agent_workflows.versions ORDER BY agent_id,version DESC").fetchall()
            resources = db.execute("SELECT * FROM agent_workflows.data ORDER BY resource_id").fetchall()
        return {"definitions": definitions, "versions": versions, "data_sources": resources}

    def workflow_save(self, kind, agent_id, draft, expected, actor):
        if kind not in ("agent", "subflow") or draft["agent_id"] != agent_id:
            raise ApplicationError("invalid_resource")
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT revision FROM agent_workflows.definitions WHERE kind=%s AND agent_id=%s FOR UPDATE", (kind, agent_id)).fetchone()
            if (row["revision"] if row else 0) != expected:
                raise ApplicationError("revision_conflict", 409)
            count = db.execute("SELECT count(*) AS n,COALESCE(sum(octet_length(draft::text)),0) AS bytes FROM agent_workflows.definitions").fetchone()
            if (not row and count["n"] >= 100) or count["bytes"] + len(canonical(draft).encode()) > 16 * 1024**2:
                raise ApplicationError("definition_capacity", 429)
            db.execute("INSERT INTO agent_workflows.definitions(kind,agent_id,revision,draft) VALUES(%s,%s,%s,%s) "
                       "ON CONFLICT(kind,agent_id) DO UPDATE SET revision=excluded.revision,draft=excluded.draft",
                       (kind, agent_id, expected + 1, Jsonb(draft)))
            self.audit(db, actor, "workflow_draft_saved", agent_id)
        return {"revision": expected + 1}

    def _workflow_version(self, db, kind, agent_id, version):
        row = db.execute("SELECT definition,execution_hash,revoked FROM agent_workflows.versions WHERE kind=%s AND agent_id=%s AND version=%s", (kind, agent_id, version)).fetchone()
        if not row or row["revoked"]:
            raise ApplicationError("definition_unavailable", 409)
        return row["definition"]

    def workflow_version(self, kind, agent_id, version):
        with self.connect() as db:
            return self._workflow_version(db, kind, agent_id, version)

    def workflow_active(self, agent_id):
        with self.connect() as db:
            row = db.execute("SELECT active_version,enabled FROM agent_workflows.definitions WHERE kind='agent' AND agent_id=%s", (agent_id,)).fetchone()
            if not row or not row["enabled"] or not row["active_version"]:
                raise ApplicationError("agent_disabled", 409)
            return {"version": row["active_version"], "definition": self._workflow_version(db, "agent", agent_id, row["active_version"])}

    def _subflow_version(self, db, agent_id, version):
        row = db.execute("SELECT enabled FROM agent_workflows.definitions WHERE kind='subflow' AND agent_id=%s", (agent_id,)).fetchone()
        if not row or not row['enabled']:
            raise ApplicationError('subflow_disabled', 409)
        return self._workflow_version(db, 'subflow', agent_id, version)

    def subflow_version(self, agent_id, version):
        with self.connect() as db:
            return self._subflow_version(db, agent_id, version)

    def validate_references(self, db, graph, stack=(), depth=0):
        from agent.tools.registry import MANIFESTS
        from agent.application.service import connector_tool_id
        builtin = {tool.id for tool in MANIFESTS}
        if graph.get('text_provider'):
            secret = db.execute('SELECT revoked FROM agent_application.secrets WHERE secret_id=%s', (graph['text_provider']['secret_ref'],)).fetchone()
            if not secret or secret['revoked']:
                raise ApplicationError('secret_unavailable', 409)
        for grant in graph["tool_grants"]:
            if grant.startswith("connector."):
                parts = grant.split(".")
                try:
                    if len(parts) != 4 or not parts[3].startswith("v"):
                        raise ValueError()
                    self.resolve_reference(db, {"connector_id": parts[1], "operation_id": parts[2], "version": int(parts[3][1:])})
                except (ValueError, IndexError):
                    raise ApplicationError("unknown_tool", 409) from None
            elif grant not in builtin:
                raise ApplicationError("unknown_tool", 409)
        if depth > 4 or graph["agent_id"] in stack:
            raise ApplicationError("recursive_subflow", 422)
        inputs = {}; outputs = {}
        for node in graph["nodes"]:
            cfg = node["config"]
            if node["type"] == "connector.invoke":
                if connector_tool_id(cfg['operation']) not in graph['tool_grants']:
                    raise ApplicationError('operation_ungranted', 409)
                operation = self.resolve_reference(db, cfg["operation"])['operation']
                inputs[node['id']] = operation['input_schema']; outputs[node['id']] = operation['output_schema']
            if node['type'] in ('conversation.collect', 'conversation.decision'):
                if 'api' in graph['entrypoints'] and not graph.get('text_provider'):
                    raise ApplicationError('text_provider_required', 409)
                for tool in cfg.get('tools', []):
                    if tool in builtin:
                        if not next(manifest for manifest in MANIFESTS if manifest.id == tool).read_only:
                            raise ApplicationError('effect_tool_requires_action_node', 409)
                    else:
                        parts = tool.split('.')
                        item = self.resolve_reference(db, {'connector_id':parts[1], 'operation_id':parts[2], 'version':int(parts[3][1:])})
                        if not item['operation']['read_only']:
                            raise ApplicationError('effect_tool_requires_action_node', 409)
            if node["type"] in ("data.lookup", "identity.resolve", "identity.verify"):
                ref = cfg["resource"]
                row = db.execute("SELECT revoked FROM agent_workflows.data_versions WHERE resource_id=%s AND version=%s", (ref["resource_id"], ref["version"])).fetchone()
                if not row or row["revoked"]:
                    raise ApplicationError("data_unavailable", 409)
            if node["type"] == "subflow":
                ref = cfg["resource"]
                child = self._subflow_version(db, ref["resource_id"], ref["version"])
                inputs[node['id']] = child['input_schema']; outputs[node['id']] = child['output_schema']
                if not set(graph["entrypoints"]) <= set(child["entrypoints"]):
                    raise ApplicationError("subflow_capability_mismatch", 409)
                if not set(child["tool_grants"]) <= set(graph["tool_grants"]):
                    raise ApplicationError("subflow_scope_mismatch", 409)
                self.validate_references(db, child, stack + (graph["agent_id"],), depth + 1)
        definition(graph, input_schemas=inputs, output_schemas=outputs)

    def workflow_validate(self, graph):
        with self.connect() as db:
            self.validate_references(db, graph)

    def workflow_publish(self, kind, agent_id, expected, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT * FROM agent_workflows.definitions WHERE kind=%s AND agent_id=%s FOR UPDATE", (kind, agent_id)).fetchone()
            if not row:
                raise ApplicationError("not_found", 404)
            if row["revision"] != expected:
                raise ApplicationError("revision_conflict", 409)
            graph = definition(row["draft"], subflow=kind == "subflow")
            self.validate_references(db, graph)
            if db.execute("SELECT count(*) AS n FROM agent_workflows.versions").fetchone()["n"] >= 2000:
                raise ApplicationError("version_capacity", 429)
            version = row["published_version"] + 1
            db.execute("INSERT INTO agent_workflows.versions VALUES(%s,%s,%s,%s,%s,false,%s)",
                       (kind, agent_id, version, Jsonb(graph), execution_hash(graph), time.time()))
            db.execute("UPDATE agent_workflows.definitions SET revision=revision+1,published_version=%s,active_version=%s WHERE kind=%s AND agent_id=%s", (version, version, kind, agent_id))
            self.audit(db, actor, "workflow_published", f"{agent_id}:{version}")
        return {"version": version, "revision": expected + 1, "execution_hash": execution_hash(graph)}

    def workflow_activate(self, kind, agent_id, version, enabled, expected, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT revision FROM agent_workflows.definitions WHERE kind=%s AND agent_id=%s FOR UPDATE", (kind, agent_id)).fetchone()
            if not row or row["revision"] != expected:
                raise ApplicationError("revision_conflict", 409)
            graph = self._workflow_version(db, kind, agent_id, version)
            if enabled:
                self.validate_references(db, graph)
            db.execute("UPDATE agent_workflows.definitions SET active_version=%s,enabled=%s,revision=revision+1 WHERE kind=%s AND agent_id=%s", (version, enabled, kind, agent_id))
            self.audit(db, actor, "workflow_activation", f"{agent_id}:{version}")
        return {"version": version, "enabled": enabled, "revision": expected + 1}

    def data_save(self, resource_id, settings, expected, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT revision FROM agent_workflows.data WHERE resource_id=%s FOR UPDATE", (resource_id,)).fetchone()
            if (row["revision"] if row else 0) != expected:
                raise ApplicationError("revision_conflict", 409)
            if not row and db.execute("SELECT count(*) AS n FROM agent_workflows.data").fetchone()["n"] >= 100:
                raise ApplicationError("data_capacity", 429)
            db.execute("INSERT INTO agent_workflows.data(resource_id,revision,settings,status) VALUES(%s,%s,%s,'draft') "
                       "ON CONFLICT(resource_id) DO UPDATE SET revision=excluded.revision,settings=excluded.settings,status='draft',error_code=NULL",
                       (resource_id, expected + 1, Jsonb(settings)))
            self.audit(db, actor, "data_draft_saved", resource_id)
        return {"revision": expected + 1}

    def data_settings(self, resource_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM agent_workflows.data WHERE resource_id=%s", (resource_id,)).fetchone()
        if not row:
            raise ApplicationError("not_found", 404)
        return row

    def data_refresh_error(self, resource_id, code):
        with self.connect() as db:
            db.execute("UPDATE agent_workflows.data SET status='refresh_failed',error_code=%s WHERE resource_id=%s", (code, resource_id))

    def data_publish(self, resource_id, expected, rows, original, metadata, key, actor, job_id=None):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT * FROM agent_workflows.data WHERE resource_id=%s FOR UPDATE", (resource_id,)).fetchone()
            if not row or row["revision"] != expected:
                raise ApplicationError("revision_conflict", 409)
            size = db.execute("SELECT COALESCE(sum(octet_length(ciphertext)+COALESCE(octet_length(original),0)),0) AS n FROM agent_workflows.data_versions").fetchone()["n"]
            version = row["published_version"] + 1
            content = key.encrypt(f"table:{resource_id}:{version}", rows)
            original_content = key.encrypt(f"original:{resource_id}:{version}", original) if original else None
            if size + len(content) + (len(original_content) if original_content else 0) > 256 * 1024**2:
                raise ApplicationError("data_storage_capacity", 429)
            now = time.time()
            db.execute("INSERT INTO agent_workflows.data_versions VALUES(%s,%s,%s,%s,%s,%s,false)", (resource_id, version, Jsonb(metadata), content, original_content, now))
            db.execute("UPDATE agent_workflows.data SET published_version=%s,revision=revision+1,status='ready',error_code=NULL,last_refresh=%s WHERE resource_id=%s", (version, now, resource_id))
            self.audit(db, actor, "data_published", f"{resource_id}:{version}")
            if job_id:
                db.execute("UPDATE agent_workflows.ingestion_jobs SET state='ready',input=NULL,result=%s WHERE job_id=%s",
                    (Jsonb({"version": version, "revision": expected + 1, "row_count": len(rows), "created": now}), job_id))
        return {"version": version, "revision": expected + 1, "row_count": len(rows), "created": now}

    def data_rows(self, resource_id, version, key):
        with self.connect() as db:
            row = db.execute("SELECT * FROM agent_workflows.data_versions WHERE resource_id=%s AND version=%s", (resource_id, version)).fetchone()
        if not row or row["revoked"]:
            raise ApplicationError("data_unavailable", 409)
        return {"rows": key.decrypt(f"table:{resource_id}:{version}", row["ciphertext"]), "created": row["created"], "metadata": row["metadata"]}

    def data_live(self, resource_id, version):
        with self.connect() as db:
            row = db.execute("SELECT revoked FROM agent_workflows.data_versions WHERE resource_id=%s AND version=%s", (resource_id, version)).fetchone()
        if not row or row["revoked"]:
            raise ApplicationError("data_unavailable", 409)

    def ingestion_admit(self, job_id, resource_id, expected, content, key, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            if db.execute("SELECT count(*) AS n FROM agent_workflows.ingestion_jobs WHERE state IN ('queued','processing')").fetchone()["n"] >= 4:
                raise ApplicationError("ingestion_capacity", 429)
            db.execute("INSERT INTO agent_workflows.ingestion_jobs VALUES(%s,%s,%s,%s,'queued',%s,NULL,NULL,%s)",
                (job_id, resource_id, expected, actor, key.encrypt(f"ingestion:{job_id}", content), time.time()))
        return {"job_id": job_id, "state": "queued"}

    def ingestion_job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT job_id,resource_id,state,result,error_code,created FROM agent_workflows.ingestion_jobs WHERE job_id=%s", (job_id,)).fetchone()
        if not row:
            raise ApplicationError("not_found", 404)
        return row

    def ingestion_fail(self, job_id, code):
        with self.connect() as db:
            db.execute("UPDATE agent_workflows.ingestion_jobs SET state='failed',input=NULL,error_code=%s WHERE job_id=%s", (code, job_id))

    def data_revoke(self, resource_id, version, actor):
        with self.connect() as db:
            if not db.execute("UPDATE agent_workflows.data_versions SET revoked=true,ciphertext=''::bytea,original=NULL WHERE resource_id=%s AND version=%s RETURNING version", (resource_id, version)).fetchone():
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "data_revoked", f"{resource_id}:{version}")
        return {"revoked": True}

    def execution_admit(self, run_id, graph, version, kind, principal, epoch, parent=None, idempotency=None, inputs=None):
        identity = digest(idempotency) if idempotency else None
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783511)")
            if identity:
                previous = db.execute("SELECT * FROM agent_workflows.executions WHERE principal=%s AND idempotency_hash=%s", (principal, identity)).fetchone()
                if previous:
                    if previous["input_digest"] != digest(inputs) or previous["agent_id"] != graph["agent_id"] or previous["version"] != version:
                        raise ApplicationError("idempotency_conflict", 409)
                    return previous, False
            count = db.execute("SELECT count(*) AS n,count(*) FILTER(WHERE status IN ('accepted','running')) AS active FROM agent_workflows.executions").fetchone()
            if count["n"] >= 100000 or count["active"] >= 64:
                raise ApplicationError("workflow_execution_capacity", 429)
            row = db.execute("INSERT INTO agent_workflows.executions(run_id,agent_id,version,parent_run_id,execution_kind,principal,epoch,status,started,definition,idempotency_hash,input_digest) "
                       "VALUES(%s,%s,%s,%s,%s,%s,%s,'running',%s,%s,%s,%s) RETURNING *", (run_id or uuid.uuid4().hex, graph["agent_id"], version, parent, kind, principal, epoch, time.time(), Jsonb(graph), identity, digest(inputs))).fetchone()
        return row, True

    def execution(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM agent_workflows.executions WHERE run_id=%s", (run_id,)).fetchone()
            if row:
                row["steps"] = db.execute("SELECT * FROM agent_workflows.steps WHERE run_id=%s ORDER BY sequence", (run_id,)).fetchall()
                row["effects"] = db.execute("SELECT operation_id,connector_id,version,operation,state,error_code FROM agent_application.effects WHERE run_id=%s ORDER BY created", (run_id,)).fetchall()
                row["reconciliation_required"] = any(effect["state"] in ("unknown", "dispatched") for effect in row["effects"])
        if not row:
            raise ApplicationError("not_found", 404)
        return row

    def execution_step(self, run_id, sequence, node, status, outcome=None, duration=0, error=None, subflow_path=''):
        with self.connect() as db:
            db.execute("INSERT INTO agent_workflows.steps VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                       "ON CONFLICT(run_id,sequence) DO UPDATE SET status=excluded.status,outcome=excluded.outcome,duration_ms=excluded.duration_ms,error_code=excluded.error_code",
                       (run_id, sequence, node["id"], node["type"], status, outcome, time.time(), duration, error, subflow_path))

    def execution_finish(self, run_id, status, key, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE agent_workflows.executions SET status=%s,ended=%s,error_code=%s,result=%s,result_expires=%s WHERE run_id=%s AND status IN ('accepted','running')",
                       (status, time.time(), error, key.encrypt(f"workflow-result:{run_id}", result) if result is not None else None, time.time() + 86400, run_id))
            db.execute("UPDATE agent_workflows.steps SET status='interrupted',error_code=%s WHERE run_id=%s AND status='running'", (error or status, run_id))

    def execution_cancel(self, run_id):
        with self.connect() as db:
            db.execute("UPDATE agent_workflows.executions SET cancel_requested=true WHERE run_id=%s AND status IN ('accepted','running')", (run_id,))
        return {"cancel_requested": True}

    def workflow_purge(self):
        with self.connect() as db:
            db.execute("UPDATE agent_workflows.executions SET result=NULL WHERE result_expires<%s", (time.time(),))
            db.execute("DELETE FROM agent_workflows.ingestion_jobs WHERE created<%s AND state NOT IN ('queued','processing')", (time.time() - 86400,))
            db.execute("DELETE FROM agent_workflows.executions WHERE ended<%s AND status NOT IN ('running','accepted') "
                       "AND NOT EXISTS(SELECT 1 FROM agent_application.effects e WHERE e.run_id=agent_workflows.executions.run_id AND e.state IN ('unknown','dispatched'))", (time.time() - 30 * 86400,))
