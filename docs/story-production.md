# Story production tools

Modern tools in `story_tools.py` wrap the deployed Story API contracts. `contracts_get(operation)` retrieves each request schema including referenced definitions. Fixed paths, checked IDs and unmodified bodies preserve the backend's guards and false billing flags. `/story/` still requires `mcp:story:*`.

Workflow: `production_preview` → `production_runs_create` → `production_execute` (phase `samples`, explicit sample scope/billing) → show actual image and voice files → `production_samples_review` (user decision for each exact result) → `production_execute` (phase `production`, separately confirmed remaining scope) → explicit `production_activate`.

An approval is never inferred by the agent. Uncertain outcomes use `production_recover`, never automatic generation retries. The first image can be reused; short voice samples are additional recordings, not replacements for full speech. Existing expert APIs are not globally restricted by this additional guided workflow.

`film_export_preview`, `film_export_create`, `film_export_get` render active media locally to MP4. No new images or voices, no social publication. The returned download path belongs to the Story API host. Studio offers the same workflow under **Proben & Freigabe**.

Samples use Story's public bound-media proxy at `/api/v1/projects/{id}/production-runs/{run_id}/media/{storage_id}`. It holds the Storage key server-side, but does not authenticate viewers. Missing MCP permission is a real denial; obtain appropriate permission through the authorization owner, never bypass it with another identity.

## Story v3.1 prototype tools (read-only)

`story_v3_prototype`, `story_v3_snapshot`, `story_v3_plan`, `story_v3_contract`,
`story_v3_explain` and `story_v3_diff` expose the Story v3.1 prototype for
interactive exploration. They are **not** the production flow: nothing here
generates, activates, approves or spends, and the surface never writes to a
project. `story_v3_prototype` needs no arguments and describes the model, the
sources, every operation, the step format and all error codes, so a client
without prior context can start there.

Two sources: the default `{"kind": "fixture", "name": "p7_scene36_export"}` is
an isolated extract of P7 scene 36 that needs no project access, and
`{"kind": "project", "project_id": N, "scene_id": N}` is one read of a real
project. There is no server-side session: `steps` (refine / freeze) are
replayed on every call, so the same request always yields the same
fingerprints. Clients pass the `snapshot_fingerprint` from
`story_v3_snapshot`; a scene that changed since then answers 409
`snapshot_drift` instead of silently using newer data, and a contract bound to
an older recording reports `stale_binding`.

The meaning layer is authored, never derived: for scenes without a built-in
plan the client supplies `authored_plan` with `speech_id`, `perception` and
exactly two claims, each addressed by phrases that must occur exactly once in
the recording. Missing source data, for example a speaker without a visual
identity, is reported as a gap and never filled in.

Story v3.1 is a prototype. Do not present it as released, and keep using
`workflow_get` / `project_readiness` for actual production.

