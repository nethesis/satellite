# Workflow API contract — schema 1

Administrator routes pass through the existing authenticated Wizard REST gateway
at `/freepbx/rest/agents/application/workflows`. Mutation requests use the existing
CSRF token. The corresponding private Satellite prefix is
`/api/agent/v1/application/workflows`, with the module bearer token and a safe
`X-Agents-Actor`. All replies disable caching. Browser code never selects a backend
host or supplies credentials inside graph JSON.

## Administrator routes

| Method | Suffix | Body/result |
|---|---|---|
| GET | `/inventory` | Built-in/custom agents, definitions, versions, data sources, provider binding names, block manifests and safe published operation schemas |
| GET | `/catalog` | Block manifests, copyable templates and Freshdesk/Kapa connector presets |
| PUT | `/definitions/{agent\|subflow}/{id}` | `{definition, expected_revision}` → `{revision}`; 0 creates a draft |
| GET | `/definitions/{kind}/{id}/versions/{version}` | `{definition}` for an immutable publication |
| POST | `/validate` | `{definition}` → `{valid, tools}`; structural and live reference checks |
| POST | `/definitions/{kind}/{id}/publish` | `{expected_revision}` → `{version, revision, execution_hash}` plus PBX sync status for an agent |
| POST | `/definitions/{kind}/{id}/activate` | `{version, enabled, expected_revision}` → activation plus PBX sync status for a voice agent |
| POST | `/test` | `{definition, fixtures, input, caller, tables?, destinations?, subflows?}` → labelled mock result and bounded trace; never performs external effects |
| GET | `/runs/{id}` | Pinned graph, status, safe step metadata, subflow paths and effect states; result content excluded |
| POST | `/runs/{id}/cancel` | Requests cancellation; current voice run returns through its PBX fallback |
| PUT | `/data/{id}` | `{settings, expected_revision}` → `{revision}` |
| POST | `/data/preview` | `{settings, content?}` → safe rows/count/errors. `content` is base64 for files; private Sheets uses its viewer credential, `google_csv` reads the selected published Google CSV |
| POST | `/data/{id}/publish` | `{expected_revision, content}` → queued ingestion job |
| POST | `/data/{id}/refresh` | `{expected_revision}` → queued Sheets refresh job |
| GET | `/jobs/{id}` | Queued/processing/ready/failed/interrupted, safe result or error code; uploaded content excluded |
| DELETE | `/data/{id}/versions/{version}` | Revokes and deletes encrypted source content; references become unavailable |

IDs use `[a-z][a-z0-9_-]{0,47}`; `internal`, `external` and `support-request` are
reserved agent IDs. Draft writes, publication and activation use optimistic
revision checks. Definition/graph size is 128 KiB; uploads are at most 10 MiB,
with a 14 MiB JSON request ceiling for base64 transfer.

Saving a draft does not change the active workflow or reload the PBX.
Publication selects the new active version and keeps the current enabled state.
Agent publication reconciles its PBX binding immediately. When a binding changes,
the gateway marks the configuration for reload and starts `retrieveHelper.sh`.
Activation also applies the binding and starts the reload. Failed binding sync
returns `pbx_sync.pending`; the existing five-minute timer retries reconciliation.

## PBX data endpoint

Satellite reads company contacts and answered-call history through the module's
`/satellite/agent-workflow-data.php` endpoint (also under `/freepbx/satellite/`).
It accepts local, bearer-authenticated POST requests for `pbx.contacts` and
`pbx.history` only. Database access uses the read-only `satellite_workflow` account.
The original `/freepbx/agent-workflow-data.php` URL remains an Apache alias for
existing Satellite runtime images; no redirect or second PHP entrypoint is used.

## Canonical definitions and bindings

Required top-level fields are `schema_version`, `agent_id`, `name`, `description`,
`entrypoints`, `nodes`, `edges`, `input_schema`, `output_schema`, `limits`, `layout`,
`provider_binding_ref`, `fallback`, `permissions` and `tool_grants`. Optional
`voice_settings` overrides model, voice/language; `text_provider` explicitly names
an API model and encrypted credential ID. Resource/operation references live in
the typed node configurations rather than a second top-level resource list.

On an OpenAI binding, `voice_settings.model: gpt-live-1` selects GPT-Live;
`gpt-realtime` selects Realtime. An omitted voice model inherits the external
profile, as before. Both APIs may share one provider binding. The text-provider
settings for API entrypoints do not select the Live delegated backend.

Live conversation steps advance on the existing validated completion tool when
the required data is ready. Deterministic action nodes still enforce grants,
exact-argument confirmation and durable effects. A model cannot approve a write.
Speech-only steps use an internal, step-specific tool to prepare their message;
the runtime submits it to Live and waits for command acceptance. This is semantic
prompt readiness, not proof that audio has finished or been heard. Caller DTMF
confirmation is armed after readiness; Realtime retains its audio-drain gate.
Both modes reject digits received before their gate and retain confirmation
expiry and argument binding.

Operator DTMF remains separate. After a private Live summary, a declined or
failed consultation replaces only the provider leg before reconnecting the
caller. It preserves the run, graph version, permissions, step budget and
deadline, and excludes private conversation from the new session. The caller
remains on hold during replacement; setup failure follows the normal fallback.

Each node has `id`, `type`, `version`, `name`, `config` and `inputs`. Each edge is
`{source, outcome, target}`. Input bindings are a literal `{value: ...}` or a
restricted selection `{node: "source_id", path: "row.amount"}`. An optional branch
binding adds `optional: true` and `default`. Merge alternatives selects exactly
one non-null object supplied by such branch bindings; other inputs default to
null. There are no expressions or scripts.
Publication checks connector/subflow schemas as well as the block manifests.

Connector references are `{connector_id, version, operation_id}` and their
canonical grant ID is `connector.<id>.<operation>.v<version>`. Resource/subflow
references are `{resource_id, version}`. All published versions are immutable. Data references may also use
`version: "latest"`; the runtime resolves all such references at run start,
including references in immutable subflows, and pins the resolved versions for
that run. Successful unchanged refreshes update freshness without creating new
versions. Failed attempts have a separate retry timestamp and do not extend data
freshness. Keep integer versions when a graph must always use one exact table.
The store retains three recent versions plus every explicitly referenced version;
persisted run references protect versions used by active executions. Per-resource capacity is 64 MiB. Selecting a
published operation in the editor does not enable its grant automatically.

Initial limits: 100 nodes, 200 edges, 200 executions of steps across nested
subflows/agent handoffs, subflow and agent-handoff depth 4, 64 KiB active context,
and at most five model responses per conversation block. Overall definition
execution timeout is 10–3600 seconds; a voice call also inherits its provider
profile deadline. Per-operation timeouts remain bounded by the shared transport.
Each response may request at most ten functions; writes are excluded from
conversation tools and require deterministic effect/confirmation nodes.
Subflow execution uses the remaining parent deadline. Its nested engine also
applies the child graph deadline; the default ten-second operation timeout does
not wrap the entire reusable graph. Parent permissions, path and deadline are
restored when the child completes, fails or is cancelled.

A mock fixture keyed by `subflow_path/node_id` (just node ID at the root) has `{outcome, output}`; conversation fixtures
use the conversation's declared output fields. Mock tables contain synthetic
normalized rows, creation time and country-code metadata. Fixtures are validated
against known block outputs/outcomes. Results include `test_mode: "mock"` and
cannot authorize a real business write. Supply reusable graphs in `subflows`
keyed by resource ID. Built-in hours require fixtures; mock verification does not
consume the real verification budget. A child fallback uses the parent subflow’s
error port (`subflow_fallback`); it does not produce a success result.

## Machine API compatibility

The existing `/agents-api/v1/runs`, `/runs/{id}`, `/runs/{id}/result` and
`/runs/{id}/cancel` routes accept custom published API workflows as well as the
existing `support-request` preset. A custom admission body is:

```json
{"agent_id":"api-echo","version":1,"input":{"message":"hello"}}
```

Use the existing bearer client and `Idempotency-Key`. The client definition's
`presets` list includes permitted custom API agent IDs. Its operation references,
customer IDs and `operations:write` scope also constrain execution. Voice-only
nodes cannot be published into an API graph. API text-provider configuration is
explicit and independent of a SIP provider binding.

If a graph can write, admission may include `approved_actions`, a bounded list
of `{operation: <reference>, arguments: <exact object>}`. The workflow's confirmation
node and dispatcher check the same digest, expiry and scoped operation. Changing
an argument invalidates the approval. Voice confirmation instead requires the
caller's DTMF after readback. Permission to publish a graph does not authorize a
machine client to execute it.

Reusing the same principal/idempotency key and identical request returns the same
run. A changed request or version conflicts. Only the owning client can read,
retrieve or cancel its run. Terminal results are encrypted and expire after 24
hours. Disabling a definition/client or revoking a referenced resource/connection
prevents further use. In-flight runs retain their pinned graph version.

## Failure and ownership

Safe errors include `revision_conflict`, `graph_cycle`, `unknown_block_version`,
`unavailable_branch_output`, `unknown_output_field`, `input_type_mismatch`,
`operation_ungranted`, `subflow_disabled`, `data_unavailable`,
`write_confirmation_required`, `urgency_rule_denied`, `caller_scope_mismatch`,
`conversation_turn_budget` and `workflow_store_unavailable`. Structural errors
can include `node_id` for selection in the inspector. No raw SQL/provider errors,
credential values, caller payment rows or private ticket text enter error metadata.

Control records are authoritative independently of monitoring delivery. A run's
safe steps include node ID/type, sequence, subflow path, status/outcome, duration
and error. Nested paths prevent a reused node ID from coloring the wrong parent
node. Provider/private consultation content is not stored as ordinary metadata.

Restart/restore interrupts unfinished owners and marks dispatched effects
unknown. Unknown writes and uncertain call releases are never automatically
replayed. The existing reconciliation surface remains available for effects.

## Payment identity and verification

Caller ID can identify a candidate resident, but is not verification. A matching
CLI sets `verified: false` with CLI assurance; `data.lookup` requires successful
code verification for the same data resource. The payment template asks every
caller for name and code. Names alone cannot authorize disclosure. New codes
must have at least six characters. Keep identifier columns as text to preserve
leading zeroes. Native numeric amounts are independent of the text decimal locale.

Verification attempts are persisted per call, matched resident and resource.
The limits are five attempts per call/resident in ten minutes and 100 per resource
in one minute. Withheld caller IDs have separate call budgets. Rotating caller ID,
restarting the service and running mock tests cannot reset a resident’s budget.

## Standalone Agents designer (definition schema 2)

The administrator document is `/freepbx/satellite/agents/`. The Wizard menu
opens it in a new tab. It uses Vue 3, Tailwind CSS 4, Nethesis Vue components
and Vue Flow. Its REST gateway remains `/freepbx/rest/agents`; browser requests
reuse verified Wizard credentials. `GET /access` supplies the session CSRF
value. Mutations require `X-Agents-CSRF`. The webhook document remains
`/freepbx/satellite/index.php`.

Schema 1 publications retain their existing behavior. Editing a legacy draft
in the new UI adds schema 2 fields without removing legacy nodes or grants.
Schema 2 adds `goal`, `instructions`, `capture_tool_content` (default false)
and `tool_bindings` (a list, default empty). Each binding has:

```json
{"tool_id":"tickets","version":1,"conversation":true,"workflow":true,
 "scope":{"filters":{"priority":4}},"instructions":"Use for urgent support requests",
 "origins":["internal","external"]}
```

Published tool references are immutable. The logical ID is `tool.tickets.v1`;
provider wire names are derived from the full reference without slug collisions.
A conversation's `tool_mode` is `inherit` or `selected`. Inherited phases use
all conversation-enabled bindings; selected phases use only their `tools` list.
Phase selection cannot add authority absent from the agent binding. Disabling
conversation use does not disable explicit workflow use. `tool.invoke` uses
`config.tool: {tool_id, version}` and literal/node-output inputs. Removing a
binding preserves explicit references so validation can identify broken steps.
Execution hashes include executable tool settings and exclude canvas layout.

A tool definition contains `name`, `description`, `kind`, `input_schema`,
`output_schema`, `config`, `execution_policy` and `timeout_seconds` (1–30).
Kinds are `api`, `query`, `documents`, `telephony` and `subflow`. API tools pin
an immutable connector operation. Native tools use exactly the exported native
schemas and retain PBX permissions and caller ownership checks. Subflow tools
pin an immutable child graph and cannot recursively add hidden registry tool
bindings. Child grants and permissions cannot exceed the parent's; step budgets,
subflow paths and deadlines are shared. A complete subflow has its graph
execution deadline rather than a single-operation ten-second deadline.

### Direct write policy

A schema 2 registry tool with `execution_policy: "direct"` executes enabled
writes immediately. It does not ask the caller for confirmation. Arguments,
fixed scope filters, verified identity, connector grants, immutable versions,
application enablement and durable effect state are still checked. A direct
schema 2 subflow applies these bounds to child business operations. Schema 1
child graphs and legacy action/confirmation nodes retain their existing policy.
The urgency operation also retains its trusted ticket/priority rule. For a
direct API tool, bind its fixed `priority: 4` scope when enabling that operation.

Machine clients still require allowed agents, operations, customer IDs and
`operations:write`. Exact `approved_actions` arguments are still required for
machine writes. An uncertain or previously recorded invocation is not replayed.

### Designer private routes

These paths extend `/api/agent/v1/application/workflows`. They require the
private bearer token and a safe `X-Agents-Actor`. The Wizard gateway exposes
only the corresponding concrete routes and preserves JSON objects and lists.

| Method | Path | Input/result |
| --- | --- | --- |
| GET | `/tools` | Draft tools and immutable versions |
| PUT | `/tools/{id}` | `{definition, expected_revision}` |
| POST | `/tools/{id}/publish` | `{expected_revision}` |
| GET / DELETE | `/tools/{id}/versions/{version}` | Read / revoke version |
| POST | `/tools/test` | `{definition, arguments, fixture?, rows?, columns?, identity?, scope?}`; synthetic results only |
| GET | `/dependencies` | Draft/publication/source/operation references |
| POST | `/data/inspect` | `{settings, content?}`; headers, conservative types and row count; no cell values |
| GET | `/data/{id}/versions/{version}/preview` | Masked rows and pinned source metadata |
| POST | `/documents/{id}/publish` | `{expected_revision, files, remove?}`; asynchronous immutable ingestion job |
| POST | `/simulations` | `{definition, fixtures, input, caller, tables?, destinations?, subflows?, messages?}` |
| GET | `/simulations/{id}` | Saved mock trace and unexpired encrypted fixture content |
| POST | `/activity` | `{since?, until?, agent_id?, kind?, status?, before?, limit?}`; unified rows, counts and cursor |
| GET | `/runs/{id}/tool-calls` | Invocation metadata; no arguments/results |
| GET | `/runs/{id}/tool-calls/{invocation}/content` | Audited opt-in encrypted content, or content unavailable |
| POST | `/connections/test` | `{operation, arguments}`; a real read operation only |
| PUT | `/credentials/{id}` | `{value}`; encrypted credential replacement |

Draft and publication calls use optimistic revisions. Publication accepts an
optional bounded `note`. Notes are stored separately from immutable graphs.
Publication/activation continue to reconcile native voice destinations, snapshot
synchronization and PBX reload; draft saves do not alter routing.

### Generic tables and local documents

Source settings add `kind: "table"` and typed `columns`, each with `name`,
`source`, `type`, `required` and `sensitive`. Types are string, integer, number,
boolean, date and month. Text identifiers preserve leading zeroes. Numeric XLSX
cells cannot silently become text identifiers. Existing payment settings retain
their resident verification and private-field rules. CSV, XLSX, private Google
Sheets and published Google CSV use the existing upload/network bounds. XLSX
formulas and macros are not evaluated.

Query tool config has a pinned or per-run `latest` resource, returned `fields`,
`filters`, `limit` (1–100), optional `sort`, `aggregate` and freshness limit.
Filters use typed literals, tool inputs or verified identity. Fixed binding
scope is applied before projection/aggregation. Aggregates are count, sum,
minimum, maximum and average. SQL and expressions are not accepted. Sensitive
columns are excluded from previews and returned rows/aggregates. Published
column metadata controls pinned versions even while a source draft is edited.

Document sources use `kind/format: "documents"`. Each file has `document_id`,
`name`, `format: "pdf" | "txt"` and base64 `content`. A collection has at most
50 documents, a combined 10 MiB upload and 10 MiB extracted text. PDFs are bounded
to 500 pages; encrypted PDFs are rejected. Scanned files without text stay visible
as `needs_verification` and supply no passages. Parsing runs in the killable,
15-second ingestion process. Originals and chunks are encrypted. Search uses
local BM25 ranking, bounded Unicode queries and deterministic document/page
citations. No embedding provider or OCR is called. Replacement/removal creates
a new immutable collection version; refresh failure keeps the last good version.

### Offline tests and Activity

The Test tab is a saved offline simulation. Conversation responses and tool
results are scripted fixtures. A conversation fixture may also include
`tool_calls: [{tool: {tool_id, version}, arguments, result, identity?}]`; phase
permissions and published schemas are checked. No provider request, phone call,
connector request or effect dispatcher is invoked. Only calls on the visited
branch appear in the result. Synthetic tables, destinations and subflows are
bounded. A connection probe and machine `/test-runs` remain real operations and
are separate from the Test tab.

Activity combines durable workflow runs, saved mocks, native call history and
legacy API runs without duplicating a workflow's monitoring record. History
unavailability is explicit. Mock records expire after 30 days; encrypted mock
messages/results and optional successful tool argument/result capture expire
after 7 days. Ordinary invocation metadata has no arguments, results, secrets or
private summaries. Restart marks unfinished invocations interrupted. Database
backup covers the added `agent_workflows` tables; clone still discards copied
control-state volumes.

The FreePBX-only native gateway adds `GET /native/inventory`,
`PUT /native/profiles/{internal|external}` with `{profile, expected_revision,
directory?}`, and `POST /native/bindings` / `PUT /native/bindings/{id}` with
`{binding}`. It keeps trunk ownership/provisioning, encrypted native secrets,
calendar validation, fallback cycle checks, directory visibility and transactional
configuration revision changes in FreePBX.
