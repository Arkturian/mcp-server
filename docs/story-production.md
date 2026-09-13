# Story production tools

Modern tools in `story_tools.py` wrap the deployed Story API contracts. `contracts_get(operation)` retrieves each request schema including referenced definitions. Fixed paths, checked IDs and unmodified bodies preserve the backend's guards and false billing flags. `/story/` still requires `mcp:story:*`.

Workflow: `production_preview` → `production_runs_create` → `production_execute` (phase `samples`, explicit sample scope/billing) → show actual image and voice files → `production_samples_review` (user decision for each exact result) → `production_execute` (phase `production`, separately confirmed remaining scope) → explicit `production_activate`.

An approval is never inferred by the agent. Uncertain outcomes use `production_recover`, never automatic generation retries. The first image can be reused; short voice samples are additional recordings, not replacements for full speech. Existing expert APIs are not globally restricted by this additional guided workflow.

`film_export_preview`, `film_export_create`, `film_export_get` render active media locally to MP4. No new images or voices, no social publication. The returned download path belongs to the Story API host. Studio offers the same workflow under **Proben & Freigabe**.

Samples use Story's public bound-media proxy at `/api/v1/projects/{id}/production-runs/{run_id}/media/{storage_id}`. It holds the Storage key server-side, but does not authenticate viewers. Missing MCP permission is a real denial; obtain appropriate permission through the authorization owner, never bypass it with another identity.
