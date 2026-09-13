"""Modern Story tools share REST contracts; no provider SDK or invented defaults.

Registration is isolated from the federation's other MCP domains so it can be
exercised against a fake transport without loading secrets or external clients.
"""

import inspect
import re
from urllib.parse import quote

# name, method, path. Bodies are the canonical schemas returned by contracts_get.
OPERATIONS = [
    ("film_export_preview", "GET", "/projects/{project_id}/film-export-preview"),
    ("film_export_create", "POST", "/projects/{project_id}/film-exports"),
    ("film_export_get", "GET", "/projects/{project_id}/film-exports/{export_id}"),
    (
        "film_export_retry",
        "POST",
        "/projects/{project_id}/film-exports/{export_id}/retry",
    ),
    ("characters_get", "GET", "/characters/{character_id}"),
    ("characters_update", "PUT", "/characters/{character_id}"),
    ("locations_list", "GET", "/projects/{project_id}/locations"),
    ("locations_create", "POST", "/projects/{project_id}/locations"),
    ("locations_get", "GET", "/locations/{location_id}"),
    ("locations_update", "PUT", "/locations/{location_id}"),
    ("style_guide_get", "GET", "/projects/{project_id}/style-guide"),
    ("style_guide_preview", "POST", "/projects/{project_id}/style-guide-preview"),
    ("style_guide_save", "PUT", "/projects/{project_id}/style-guide"),
    ("shot_prompt_preview", "POST", "/shots/{shot_id}/generate-image-prompt"),
    ("audio_prompt_get", "GET", "/scenes/{scene_id}/audio-prompt"),
    ("audio_prompt_preview", "POST", "/scenes/{scene_id}/audio-prompt-preview"),
    ("audio_direction_save", "PUT", "/scenes/{scene_id}/audio-direction"),
    ("shot_timing_get", "GET", "/scenes/{scene_id}/shot-timing"),
    ("shot_timing_preview", "POST", "/scenes/{scene_id}/shot-timing-preview"),
    ("shot_timing_save", "PUT", "/scenes/{scene_id}/shot-timing"),
    ("scene_development_get", "GET", "/scenes/{scene_id}/development"),
    ("scene_development_generate", "POST", "/scenes/{scene_id}/development"),
    (
        "scene_development_contracts",
        "GET",
        "/scenes/{scene_id}/development/{draft_id}/contracts",
    ),
    (
        "scene_development_apply",
        "POST",
        "/scenes/{scene_id}/development/{draft_id}/apply",
    ),
    ("scene_development_film", "PUT", "/scenes/{scene_id}/development-film"),
    ("production_preview", "POST", "/projects/{project_id}/production-preview"),
    ("production_runs_create", "POST", "/projects/{project_id}/production-runs"),
    ("production_runs_list", "GET", "/projects/{project_id}/production-runs"),
    ("production_runs_get", "GET", "/projects/{project_id}/production-runs/{run_id}"),
    (
        "production_samples_review",
        "PUT",
        "/projects/{project_id}/production-runs/{run_id}/sample-review",
    ),
    (
        "production_execute",
        "POST",
        "/projects/{project_id}/production-runs/{run_id}/execute",
    ),
    (
        "production_recover",
        "POST",
        "/projects/{project_id}/production-runs/{run_id}/recover",
    ),
    (
        "production_activate",
        "POST",
        "/projects/{project_id}/production-runs/{run_id}/activate",
    ),
    ("audio_attempts_list", "GET", "/scenes/{scene_id}/audio-attempts"),
    (
        "audio_attempts_recover",
        "POST",
        "/scenes/{scene_id}/audio-attempts/{attempt_id}/recover",
    ),
    (
        "audio_attempts_mix",
        "POST",
        "/scenes/{scene_id}/audio-attempts/{attempt_id}/mix",
    ),
]
NO_BODY = {
    "film_export_retry",
    "production_recover",
    "audio_attempts_recover",
    "audio_attempts_mix",
}


def register_story_tools(mcp, call_api):
    for name, method, path in OPERATIONS:
        fields = re.findall(r"{(\w+)}", path)
        with_body = method in {"POST", "PUT"} and name not in NO_BODY
        # The signature supplies discoverable path parameters. The body schema is
        # fetched from the deployed API instead of copying drifting Pydantic types.
        params = [
            inspect.Parameter(
                key,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=(
                    str
                    if key in {"run_id", "attempt_id", "draft_id", "export_id"}
                    else int
                ),
            )
            for key in fields
        ]
        if with_body:
            params.append(
                inspect.Parameter(
                    "body", inspect.Parameter.KEYWORD_ONLY, annotation=dict
                )
            )

        def make_handler(method, path, fields, with_body):
            async def invoke(**kwargs):
                values = {}
                for key in fields:
                    value = kwargs[key]
                    if key not in {
                        "run_id",
                        "attempt_id",
                        "draft_id",
                        "export_id",
                    } and (type(value) is not int or value <= 0):
                        raise ValueError("Entity IDs must be positive integers")
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(value)):
                        raise ValueError("Invalid path identifier")
                    values[key] = quote(str(value), safe="")
                endpoint = "/api/v1" + path.format(**values)
                return await call_api(
                    method,
                    endpoint,
                    **({"json_body": kwargs["body"]} if with_body else {}),
                )

            return invoke

        fn = make_handler(method, path, fields, with_body)
        fn.__name__ = name
        fn.__signature__ = inspect.Signature(params, return_annotation=dict)
        description = f"Story {method} {path}. Read contracts_get(operation='{name}') for the exact body. "
        if name == "production_execute":
            description += "Requires displayed remaining item_ids, fresh fingerprints/revision and explicit confirm_api_billing. Run samples first; obtain the user's separate approval for every sample before production. Never invent approval or retry an uncertain call."
        elif name == "production_samples_review":
            description += "Record only the user's explicit decision after showing/playing this exact sample. Approval is not billing consent."
        elif name == "scene_development_generate":
            description += "Uses subscription text quota; generates a reviewable draft, no images or speech."
        else:
            description += "Preserves canonical validation, conflicts and provenance. No implicit media generation."
        mcp.add_tool(fn, name=name, description=description)

    @mcp.tool(
        name="contracts_get",
        description="Read the deployed Story request schema for one modern operation, including referenced definitions. No generation.",
    )
    async def contracts_get(operation: str) -> dict:
        entry = next((e for e in OPERATIONS if e[0] == operation), None)
        if entry is None:
            return {"operations": [e[0] for e in OPERATIONS]}
        spec = await call_api("GET", "/openapi.json")
        route = (
            spec.get("paths", {}).get("/api/v1" + entry[2], {}).get(entry[1].lower())
        )
        if not route:
            return {"available": False, "operation": operation}
        definitions = spec.get("components", {}).get("schemas", {})
        needed = {}

        def collect(value):
            if isinstance(value, dict):
                ref = value.get("$ref", "")
                if ref.startswith("#/components/schemas/"):
                    key = ref.rsplit("/", 1)[-1]
                    if key not in needed and key in definitions:
                        needed[key] = definitions[key]
                        collect(needed[key])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(route)
        return {
            "available": True,
            "operation": operation,
            "route": route,
            "components": {"schemas": needed},
        }
