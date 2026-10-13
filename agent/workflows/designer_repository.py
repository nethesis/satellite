"""Transactional reusable tools and redacted activity for the Vue designer."""

import time

from psycopg.types.json import Jsonb

from agent.application.contracts import ApplicationError, canonical, digest
from .tool_contracts import tool, tool_key

DDL = """
INSERT INTO agent_workflows.migrations VALUES(3) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS agent_workflows.publication_notes(kind varchar(16), resource_id varchar(48), version integer, note text NOT NULL, PRIMARY KEY(kind,resource_id,version));
CREATE TABLE IF NOT EXISTS agent_workflows.tools(
 tool_id varchar(48) PRIMARY KEY, revision bigint NOT NULL, draft jsonb NOT NULL,
 published_version integer NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS agent_workflows.tool_versions(
 tool_id varchar(48), version integer, definition jsonb NOT NULL,
 revoked boolean NOT NULL DEFAULT false, created double precision NOT NULL,
 PRIMARY KEY(tool_id,version));
CREATE TABLE IF NOT EXISTS agent_workflows.tool_calls(
 run_id varchar(128) REFERENCES agent_workflows.executions ON DELETE CASCADE,
 invocation_id varchar(128), node_id varchar(48), subflow_path varchar(256),
 tool_id varchar(48), version integer, mode varchar(16), state varchar(32),
 started double precision NOT NULL, duration_ms integer, error_code varchar(64),
 input_digest varchar(64) NOT NULL, content bytea, content_expires double precision,
 PRIMARY KEY(run_id,invocation_id));
CREATE INDEX IF NOT EXISTS designer_call_run ON agent_workflows.tool_calls(run_id,started);
CREATE TABLE IF NOT EXISTS agent_workflows.simulations(
 run_id varchar(128) PRIMARY KEY, agent_id varchar(48), status varchar(32),
 started double precision NOT NULL, actor varchar(128), definition jsonb NOT NULL,
 trace jsonb NOT NULL, content bytea, content_expires double precision,
 expires double precision NOT NULL);
"""


class DesignerRepositoryMixin:
    def tool_inventory(self):
        with self.connect() as db:
            return {"tools": db.execute("SELECT * FROM agent_workflows.tools ORDER BY tool_id").fetchall(),
                    "tool_versions": db.execute("SELECT * FROM agent_workflows.tool_versions ORDER BY tool_id,version DESC").fetchall()}

    def tool_save(self, tool_id, draft, expected, actor):
        draft = tool(draft)
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT revision FROM agent_workflows.tools WHERE tool_id=%s FOR UPDATE", (tool_id,)).fetchone()
            if (row["revision"] if row else 0) != expected:
                raise ApplicationError("revision_conflict", 409)
            if not row and db.execute("SELECT count(*) AS n FROM agent_workflows.tools").fetchone()["n"] >= 100:
                raise ApplicationError("tool_capacity", 429)
            db.execute("INSERT INTO agent_workflows.tools(tool_id,revision,draft) VALUES(%s,%s,%s) "
                       "ON CONFLICT(tool_id) DO UPDATE SET revision=excluded.revision,draft=excluded.draft",
                       (tool_id, expected + 1, Jsonb(draft)))
            self.audit(db, actor, "tool_draft_saved", tool_id)
        return {"revision": expected + 1}

    def _tool_version(self, db, tool_id, version):
        row = db.execute("SELECT definition FROM agent_workflows.tool_versions WHERE tool_id=%s AND version=%s AND NOT revoked", (tool_id, version)).fetchone()
        if not row:
            raise ApplicationError("tool_unavailable", 409)
        return row["definition"]

    def tool_version(self, tool_id, version):
        with self.connect() as db:
            return self._tool_version(db, tool_id, version)

    def validate_tool_reference(self, db, value):
        cfg = value["config"]
        if value["kind"] == "api":
            op = self.resolve_reference(db, cfg["operation"])["operation"]
            if value["input_schema"] != op["input_schema"] or value["output_schema"] != op["output_schema"]:
                raise ApplicationError("operation_schema_mismatch")
            if (value["execution_policy"] == "read_only") != op["read_only"]:
                raise ApplicationError("invalid_tool_policy")
        if value["kind"] in ("query", "documents"):
            ref = cfg["resource"]
            row = db.execute("SELECT settings,published_version FROM agent_workflows.data WHERE resource_id=%s", (ref["resource_id"],)).fetchone()
            version = row["published_version"] if row and ref["version"] == "latest" else ref["version"]
            snapshot = db.execute("SELECT metadata FROM agent_workflows.data_versions WHERE resource_id=%s AND version=%s AND NOT revoked", (ref["resource_id"], version)).fetchone() if row else None
            if not snapshot:
                raise ApplicationError("data_unavailable", 409)
            if (value["kind"] == "documents") != (snapshot["metadata"].get("kind") == "documents"):
                raise ApplicationError("source_kind_mismatch")
            if value["kind"] == "query":
                from .data import CANONICAL_FIELDS
                columns = snapshot["metadata"].get("columns", [])
                names = {c["name"] for c in columns} if columns else CANONICAL_FIELDS
                sensitive = {c["name"] for c in columns if c.get("sensitive")} | {"verification_code", "phone"}
                used = set(cfg["fields"]) | {p["field"] for p in cfg["filters"]}
                used |= {cfg[k]["field"] for k in ("sort", "aggregate") if k in cfg}
                if not used <= names or set(cfg["fields"]) & sensitive or cfg.get("aggregate", {}).get("field") in sensitive:
                    raise ApplicationError("unknown_or_sensitive_field")
        if value["kind"] == "subflow":
            ref = cfg["resource"]
            child = self._subflow_version(db, ref["resource_id"], ref["version"])
            if child.get("tool_bindings"):
                # A reusable tool may not recursively add hidden tool authority.
                raise ApplicationError("nested_tool_subflow_forbidden")
            if value["input_schema"] != child["input_schema"] or value["output_schema"] != child["output_schema"]:
                raise ApplicationError("subflow_schema_mismatch")
            if value["execution_policy"] == "read_only":
                from .contracts import CATALOG_BY_TYPE
                for node in child["nodes"]:
                    if not CATALOG_BY_TYPE[node["type"]]["read_only"] or node["type"] == "subflow":
                        raise ApplicationError("invalid_tool_policy")
            self.validate_references(db, child)

    def tool_publish(self, tool_id, expected, actor):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(783510)")
            row = db.execute("SELECT * FROM agent_workflows.tools WHERE tool_id=%s FOR UPDATE", (tool_id,)).fetchone()
            if not row or row["revision"] != expected:
                raise ApplicationError("revision_conflict", 409)
            value = tool(row["draft"])
            self.validate_tool_reference(db, value)
            if db.execute("SELECT count(*) AS n FROM agent_workflows.tool_versions").fetchone()["n"] >= 2000:
                raise ApplicationError("version_capacity", 429)
            version = row["published_version"] + 1
            db.execute("INSERT INTO agent_workflows.tool_versions VALUES(%s,%s,%s,false,%s)", (tool_id, version, Jsonb(value), time.time()))
            db.execute("UPDATE agent_workflows.tools SET published_version=%s,revision=revision+1 WHERE tool_id=%s", (version, tool_id))
            self.audit(db, actor, "tool_published", f"{tool_id}:{version}")
        return {"version": version, "revision": expected + 1}

    def tool_revoke(self, tool_id, version, actor):
        with self.connect() as db:
            if not db.execute("UPDATE agent_workflows.tool_versions SET revoked=true WHERE tool_id=%s AND version=%s RETURNING version", (tool_id, version)).fetchone():
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "tool_revoked", f"{tool_id}:{version}")
        return {"revoked": True}

    def designer_dependencies(self):
        with self.connect() as db:
            graphs = db.execute("SELECT kind,agent_id,0 AS version,draft AS definition FROM agent_workflows.definitions "
                                "UNION ALL SELECT kind,agent_id,version,definition FROM agent_workflows.versions WHERE NOT revoked").fetchall()
            versions = db.execute("SELECT tool_id,version,definition FROM agent_workflows.tool_versions WHERE NOT revoked").fetchall()
        uses = []
        for graph in graphs:
            for b in graph["definition"].get("tool_bindings", []):
                uses.append({"kind": "tool", "resource_id": b["tool_id"], "resource_version": b["version"],
                             "agent_id": graph["agent_id"], "definition_kind": graph["kind"], "version": graph["version"]})
            for node in graph["definition"].get("nodes", []):
                resource = node.get("config", {}).get("resource")
                if resource:
                    uses.append({"kind": "subflow" if node["type"] == "subflow" else "source", "resource_id": resource["resource_id"],
                                 "resource_version": resource["version"], "agent_id": graph["agent_id"], "version": graph["version"], "node_id": node["id"]})
        for row in versions:
            cfg = row["definition"]["config"]
            ref = cfg.get("resource") or cfg.get("operation")
            if ref:
                uses.append({"kind": "connection" if "connector_id" in ref else "subflow" if row["definition"]["kind"] == "subflow" else "source", "resource_id": ref.get("resource_id", ref.get("connector_id")),
                             "resource_version": ref["version"], "tool_id": row["tool_id"], "version": row["version"]})
        return uses

    def invocation_start(self, context, invocation_id, ref, args, mode):
        with self.connect() as db:
            row = db.execute("SELECT input_digest,state FROM agent_workflows.tool_calls WHERE run_id=%s AND invocation_id=%s FOR UPDATE", (context["run_id"], invocation_id)).fetchone()
            fingerprint = digest({"tool": ref, "arguments": args})
            if row:
                if row["input_digest"] != fingerprint:
                    raise ApplicationError("invocation_conflict", 409)
                return False
            db.execute("INSERT INTO agent_workflows.tool_calls(run_id,invocation_id,node_id,subflow_path,tool_id,version,mode,state,started,input_digest) "
                       "VALUES(%s,%s,%s,%s,%s,%s,%s,'running',%s,%s)",
                       (context["run_id"], invocation_id, context.get("workflow_node", {}).get("id"), "/".join(context.get("subflow_path", [])), ref["tool_id"], ref["version"], mode, time.time(), fingerprint))
            return True

    def invocation_finish(self, run_id, invocation_id, duration, error, content, key):
        with self.connect() as db:
            encrypted = key.encrypt(f"tool-call:{run_id}:{invocation_id}", content) if content is not None else None
            db.execute("UPDATE agent_workflows.tool_calls SET state=%s,duration_ms=%s,error_code=%s,content=%s,content_expires=%s WHERE run_id=%s AND invocation_id=%s",
                       ("failed" if error else "completed", duration, error, encrypted, time.time() + 7 * 86400 if encrypted else None, run_id, invocation_id))

    def invocation_records(self, run_id):
        with self.connect() as db:
            return db.execute("SELECT run_id,invocation_id,node_id,subflow_path,tool_id,version,mode,state,started,duration_ms,error_code, "
                              "content IS NOT NULL AND content_expires>%s AS content_available FROM agent_workflows.tool_calls WHERE run_id=%s ORDER BY started,invocation_id", (time.time(), run_id)).fetchall()

    def invocation_content(self, run_id, invocation_id, key, actor):
        with self.connect() as db:
            row = db.execute("SELECT content FROM agent_workflows.tool_calls WHERE run_id=%s AND invocation_id=%s AND content_expires>%s", (run_id, invocation_id, time.time())).fetchone()
            if not row or row["content"] is None:
                raise ApplicationError("content_unavailable", 404)
            self.audit(db, actor, "tool_content_read", f"{run_id}:{invocation_id}"[:128])
            return key.decrypt(f"tool-call:{run_id}:{invocation_id}", row["content"])

    def simulation_save(self, run_id, graph, result, messages, key, actor):
        now = time.time()
        with self.connect() as db:
            if db.execute("SELECT count(*) AS n FROM agent_workflows.simulations").fetchone()["n"] >= 10000:
                raise ApplicationError("simulation_capacity", 429)
            # Trace data is fixture content: encrypt the full trace and keep only safe fields here.
            trace = [{k: s[k] for k in ("node_id", "block_type", "outcome", "duration_ms", "error_code", "subflow_path") if k in s} for s in result["trace"]]
            db.execute("INSERT INTO agent_workflows.simulations VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                       (run_id, graph["agent_id"], result["status"], now, actor, Jsonb(graph), Jsonb(trace),
                        key.encrypt(f"simulation:{run_id}", {"result": result, "messages": messages}), now + 7 * 86400, now + 30 * 86400))
            self.audit(db, actor, "simulation_created", run_id)
        return {"run_id": run_id, "test_mode": "mock"} | result

    def simulation_detail(self, run_id, key, actor):
        with self.connect() as db:
            row = db.execute("SELECT * FROM agent_workflows.simulations WHERE run_id=%s AND expires>%s", (run_id, time.time())).fetchone()
            if not row:
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "simulation_read", run_id)
        content = key.decrypt(f"simulation:{run_id}", row["content"]) if row["content"] and row["content_expires"] > time.time() else None
        return {"run_id": run_id, "agent_id": row["agent_id"], "execution_kind": "mock", "status": row["status"], "definition": row["definition"],
                "trace": row["trace"], "content": content, "test_mode": "mock"}

    def designer_activity(self, since, until, agent_id, kind, status, before, limit, history_available=True):
        now = time.time()
        # Each durable workflow appears once; monitoring contributes native voice calls only.
        query = """WITH all_runs AS (
          SELECT run_id,agent_id,version,execution_kind,status,started,ended,error_code FROM agent_workflows.executions
          UNION ALL SELECT run_id,agent_id,0,'mock',status,started,started,NULL FROM agent_workflows.simulations WHERE expires>%(now)s
          UNION ALL SELECT a.run_id,'support-request',a.preset_version,'api',a.status,a.started,a.ended,a.error_code FROM agent_application.runs a
            WHERE a.expires>%(now)s AND NOT EXISTS(SELECT 1 FROM agent_workflows.executions e WHERE e.run_id=a.run_id)
        ) SELECT * FROM all_runs WHERE started>=%(since)s AND started<=%(until)s
          AND (%(agent)s='' OR agent_id=%(agent)s) AND (%(kind)s='' OR execution_kind=%(kind)s)
          AND (%(status)s='' OR status=%(status)s)
          AND (%(before)s='' OR (started,run_id)<(SELECT started,run_id FROM all_runs WHERE run_id=%(before)s))
          ORDER BY started DESC,run_id DESC LIMIT %(limit)s"""
        if history_available:
            query = query.replace("          UNION ALL SELECT a.run_id", '          UNION ALL SELECT m.run_id,m.agent_id,0,m.execution_kind,m.status,m.started,m.ended,m.reason FROM agent_monitoring.runs m\n            WHERE m.execution_kind=\'voice\' AND m.expires>%(now)s AND NOT EXISTS(SELECT 1 FROM agent_workflows.executions e WHERE e.run_id=m.run_id)\n' + "          UNION ALL SELECT a.run_id")
        summary = query.split("          AND (%(before)s")[0].replace("SELECT * FROM all_runs WHERE", "SELECT status,execution_kind,count(*) AS count FROM all_runs WHERE") + " GROUP BY status,execution_kind"
        with self.connect() as db:
            args = {"now": now, "since": since, "until": until, "agent": agent_id,
                                      "kind": kind, "status": status, "before": before, "limit": limit + 1}
            rows = db.execute(query, args).fetchall()
            counts = db.execute(summary, args).fetchall()
        return {"counts": counts, "runs": rows[:limit], "next_cursor": rows[limit - 1]["run_id"] if len(rows) > limit else None, "since": since, "until": until}

    def data_original(self, resource_id, version, key):
        with self.connect() as db:
            row = db.execute("SELECT original FROM agent_workflows.data_versions WHERE resource_id=%s AND version=%s AND NOT revoked", (resource_id, version)).fetchone()
        if not row: raise ApplicationError("data_unavailable", 409)
        return key.decrypt(f"original:{resource_id}:{version}", row["original"]) if row["original"] else None

    def designer_replace_secret(self, secret_id, value, key, actor):
        with self.connect() as db:
            ciphertext = key.encrypt(f"secret:{secret_id}", value)
            if not db.execute("UPDATE agent_application.secrets SET ciphertext=%s,revoked=false WHERE secret_id=%s RETURNING secret_id", (ciphertext, secret_id)).fetchone():
                raise ApplicationError("not_found", 404)
            self.audit(db, actor, "secret_replaced", secret_id)
        return {"secret_id": secret_id, "configured": True}

    def designer_purge(self):
        now = time.time()
        with self.connect() as db:
            db.execute("UPDATE agent_workflows.tool_calls SET content=NULL WHERE content_expires<%s", (now,))
            db.execute("UPDATE agent_workflows.simulations SET content=NULL WHERE content_expires<%s", (now,))
            db.execute("DELETE FROM agent_workflows.simulations WHERE expires<%s", (now,))
