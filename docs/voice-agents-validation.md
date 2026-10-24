# Alpha validation

The cleanup starts from NethVoice `d1e23cae8f286af401397afa290cf5fe94d09d30`
and Satellite `327b714ed977970afe1ac97fd9a5fc9af839c94a`, including the
separately preserved callback and compact-channel-ID fixes. Recovery bundles,
tracked patches, untracked files and previous development notes are retained
outside published documentation.

The rewritten branches incorporate current main: NethVoice
`1da25a1554b5a0c0ab350c7cd01079f9f3fb07d7`, Satellite
`1599a81f9ea96f4aea47d2121252267c10976f5f`. Current executable runtime code
is introduced by Satellite's call-execution commit; the paired NethVoice runtime
pin identifies the complete source archive. Python remains 3.12.13.

| Gate | Evidence |
| --- | --- |
| Fresh baseline, unsupported development DB rejection, interrupted retry | Disposable MariaDB; old objects unchanged on rejection; seed and repeated setup verified |
| Paired PHP/Python, provider emulators, ARI, routing and recovery | 266 runtime tests passed, including real disposable DB integration |
| Server authorization, CSRF, immutable publications, grants and restore | PHP/MariaDB harness passed, including separate runtime accounts |
| Native CDR/CEL identifier bounds | Strict native field limits reject UUID-sized channel IDs; compact Local identities and independent session/run correlation persist |
| UI type/lint/format/unit/build/browser | Passed; seven unit tests and 27 browser cases |
| Transcription | 137 Python tests passed, one skipped; nine PHP segmentation regressions passed |
| NS8 unit regressions | 89 passed, including 14 subtests |
| EN/IT administrator manual | Docusaurus built both locales; existing browser-data-age warnings remain |

Staging validation uses a disposable instance. A protected backup includes databases,
secret/environment files, native PBX state, service overrides and exact images.
The pre-reset databases were restored on an isolated MariaDB server before reset;
old image identities are pinned for rollback. This rehearses the database portion,
not a full NS8 machine recovery. Native CDR and unrelated PBX tables matched their
pre-reset fingerprints after installation and supported configuration recreation.

A fresh Receptionist was disabled/unpublished before its test configuration was
saved and published. Runtime readiness requires authentication, reports v8 with
four tools and no monitoring degradation. Both services run independently of
transcription enablement. Installation repetition preserves the current publication.

## Controlled staging observations

- GPT-Live primary WebSocket: caller-side audio was observed; a synthetic goodbye
  invoked the enabled hangup tool and ended as `ended_by_agent`, leaving no channels.
  First caller-side signal energy was about 3.58 seconds in that trial. This is not
  a human listening or intelligibility measurement.
- OpenAI Realtime SIP: an approved extension transfer recorded both native handoff
  and `transferred`, then left zero channels. The test target was a controlled,
  unregistered native extension with safe native hold fallback, not an employee
  answering a telephone.
- A denied destination was not reached. An independent registry check using the
  real restricted DB account returned one approved directory target and one
  callback candidate, rejected the other extension and performed no ARI action.
- Killing only the runner during one owned call exercised supervisor/recovery.
  The run became `interrupted / runtime_restarted`, with zero remaining channels.
- A deadline trial exposed the native watchdog firing before the runtime because
  its seconds were rounded down. The failing regression and fix keep the runtime
  deadline authoritative and add bounded native-watchdog grace. The corrected
  live trial recorded `fallback / duration_exceeded` and left zero channels.
- Actual native CDR/daily CDR/CEL probes retained compact provider/consult Local
  identities, both CEL halves and independent session correlation under strict
  SQL limits, without provider connections or container replacement.

GPT-Live spoken-name transfer and both GPT-Live/Realtime callback trials did
not establish a confirmed handoff. The controlled provider windows closed after
ten attempts; synthetic CDR and test-extension fixtures were removed. Defaults
were restored, services remained healthy and zero channels remained. Do not count their ordinary completion as transfer success. Human
listening, subjective barge-in, long-duration calls, actual employee-answer
confirmation, Grok live coverage and coexistence under sustained transcription
load remain separate acceptance gates. Never infer these from provider
acceptance, media events or signal-energy measurements.

## Image and source provenance

The full Satellite image was built with Python 3.12.13, UID/GID 65532, v8-only
contracts and the exact current runtime source; source-only diagnostics are absent.
The standalone runner image was also built. Staging uses source-verified feature
overlays on the saved installed image identities, including the watchdog fix.
Exact image hashes and deployment receipts remain in protected local verification
material, outside published documentation. FreePBX/module image results and the downloadable Satellite archive pin/checksum
are recorded with the paired NethVoice source. Verify that final pair before
publishing a release. Local Docker builds omit optional BuildKit cache mounts
where the host has no BuildKit plugin; production build recipes are unchanged.
