"""Modern Story tools share REST contracts; no provider SDK or invented defaults.

Registration is isolated from the federation's other MCP domains so it can be
exercised against a fake transport without loading secrets or external clients.
"""

import inspect
import re
from urllib.parse import quote

# name, method, path. Bodies are the canonical schemas returned by contracts_get.
OPERATIONS = [
    ("text_review_speech_save", "PUT", "/projects/{project_id}/text-review/speech"),
    ("text_review_get", "GET", "/projects/{project_id}/text-review"),
    ("text_review_analyse", "POST", "/projects/{project_id}/text-review"),
    ("text_review_decide", "PUT", "/projects/{project_id}/text-review/{review_id}/decision"),
    ("image_feedback_list", "GET", "/projects/{project_id}/image-feedback"),
    ("image_feedback_get", "GET", "/media/{media_id}/feedback"),
    ("image_feedback_save", "PUT", "/media/{media_id}/feedback"),
    ("visual_development_get", "GET", "/projects/{project_id}/visual-development"),
    ("visual_development_analyse", "POST", "/projects/{project_id}/visual-development"),
    (
        "visual_development_image",
        "POST",
        "/projects/{project_id}/visual-development/{draft_id}/images",
    ),
    (
        "visual_development_recover",
        "POST",
        "/projects/{project_id}/visual-development/{draft_id}/recover",
    ),
    (
        "visual_development_activate",
        "POST",
        "/projects/{project_id}/visual-development/{draft_id}/activate",
    ),
    ("story_intent_get", "GET", "/projects/{project_id}/story-intent"),
    ("story_intent_save", "PUT", "/projects/{project_id}/story-intent"),
    ("workflow_get", "GET", "/workflow"),
    ("project_readiness", "GET", "/projects/{project_id}/readiness"),
    ("scenes_save", "PUT", "/scenes/{scene_id}"),
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
# Legacy tools retain their call signatures; expose their actual REST schemas too.
CRUD_OPERATIONS = [
    ("story_intent_get", "GET", "/projects/{project_id}/story-intent"),
    ("story_intent_save", "PUT", "/projects/{project_id}/story-intent"),
    ("projects_create", "POST", "/projects/"),
    ("projects_update", "PUT", "/projects/{project_id}"),
    ("projects_get", "GET", "/projects/{project_id}"),
    ("projects_list", "GET", "/projects/"),
    ("characters_create", "POST", "/projects/{project_id}/characters"),
    ("beats_create", "POST", "/projects/{project_id}/beats"),
    ("beats_update", "PUT", "/beats/{beat_id}"),
    ("scenes_create", "POST", "/beats/{beat_id}/scenes"),
    ("scenes_update", "PUT", "/scenes/{scene_id}"),
    ("shots_create", "POST", "/scenes/{scene_id}/shots"),
    ("shots_update", "PUT", "/shots/{shot_id}"),
]

NO_BODY = {
    "visual_development_recover",
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
                    if key in {"run_id", "attempt_id", "draft_id", "export_id", "review_id"}
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
                        "review_id",
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
        if name.startswith("text_review_"):
            description += "Review the complete ordered spoken text before full TTS: opening, introductions, chat references, question/answer logic, transitions, wording. GET sources/fingerprint; analyse creates one durable subscription-text review, never media or edits. Lost response: GET existing reviews, never auto-repeat. Show exact anchored findings/suggestions to the user. Use text_review_speech_save with expected_source_fingerprint and scene_id/speech_id/text for explicit user-accepted edits; media stays unchanged. After actual text edits analyse the new version; to consciously retain findings decide with an explicit reason for EVERY finding. Never invent a decision. Starting review blocks new full TTS until current text is explicitly accepted; existing media and voice auditions remain available. audio_checked=false: this is NOT listening or transcription. Portal optional."
        elif name.startswith("image_feedback_"):
            description += "Read/save user observations about one displayed image: rating undecided/like/needs_change, liked, change_requested. Read first for expected_revision and expected_media_fingerprint. Persists across sessions and feeds the next visual_development_analyse. Feedback is NEVER production/billing approval and never regenerates, replaces or activates media. Conflicts return409; preserve user text and reload explicitly."
        elif name.startswith("visual_development_"):
            description += "Two distinct scopes: to improve ONE displayed image, set target_media_id to its current ShotMedia.id in analyse; show only that image and its feedback. This creates exactly one replacement candidate and activation preserves the shot, every cut time and audio. Omit target_media_id ONLY for explicit whole-film picture direction. Never infer global scope from a single-image note. Parent proposals must have the same target. Refine pictures of an existing recorded film by spoken meaning. Get context first; analyse with coarse/balanced/fine detail and exact source fingerprint (subscription text only). Show frames with spoken_text, real times, reasons, reused thumbnails and new motif descriptions in chat. No equal-time buckets, audio changes or invented times. Image creates ONE requested new entry with explicit model/size/quality and billing consent, saving a durable result without changing the film. Same request never regenerates; recover only GETs provider status. Activate only after all new images exist and the user accepts the visual edit; atomically binds word cuts, retains original media/audio. Portal optional."
        elif name == "workflow_get":
            description = "START HERE for making stories/films entirely in this chat: samples -> one user decision -> remaining production -> MP4. Portal optional; never require UI clicks. Read the complete guide and field meanings. No generation."
        elif name == "story_intent_save":
            description = "Set explicit interview intent with question_character_id and answer_character_id (different project speakers). First story_intent_get for expected_fingerprint; contracts_get for schema. Does not generate text or media and never approves content."
        elif name == "project_readiness":
            description = "START HERE for an existing project: diagnose missing scenes, speech, voices, shots and references using real builders; returns blockers and next tools. Read-only, never authorizes production. See workflow_get for the complete recipe."
        elif name == "production_execute":
            description += "Requires displayed remaining item_ids, fresh fingerprints/revision and explicit confirm_api_billing. Run samples first; show sample_media URLs in chat. One explicit user decision can cover all displayed samples and the announced remaining scope; do not require separate user clicks per API call. Reuse already granted permission. Never invent approval or retry an uncertain call."
        elif name == "production_samples_review":
            description += "Record the user's decision after presenting this exact sample in chat. One group approval can cover all shown samples: call this tool for each with a fresh revision, without asking the user again. Quality approval alone is not billing consent; the same user message may explicitly authorize both."
        elif name in {"production_runs_get", "production_runs_create"}:
            description += "Returns sample_media with bound image/audio URLs for presentation in chat. Do not redirect the user to a portal. Run creation stores a plan and does not spend."
        elif name in {"film_export_create", "film_export_get"}:
            description += "Export existing film media via MCP; poll get until complete and return download_url as the actual MP4 link in chat. No portal required, no image or TTS generation. Null URL means not complete."
        elif name == "scene_development_generate":
            description += "Uses subscription text quota; generates a reviewable draft, no images or speech."
        else:
            description += "Preserves canonical validation, conflicts and provenance. No implicit media generation."
        mcp.add_tool(fn, name=name, description=description)

    @mcp.tool(
        name="contracts_get",
        description="Read the deployed Story schema and complete chat workflow for CRUD or production operations. If workflow_get is absent from your connector, call this tool with production_preview: workflow.guide and workflow.interaction provide the same instructions. No generation.",
    )
    async def contracts_get(operation: str) -> dict:
        entry = next(
            (e for e in OPERATIONS + CRUD_OPERATIONS if e[0] == operation), None
        )
        if entry is None:
            return {"operations": [e[0] for e in OPERATIONS + CRUD_OPERATIONS]}
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
        workflow = await call_api("GET", "/api/v1/workflow")
        return {
            "available": True,
            "operation": operation,
            "route": route,
            "components": {"schemas": needed},
            "field_usage": workflow.get("field_usage", {}),
            "workflow": {
                key: workflow[key]
                for key in ("version", "start_tool", "guide", "interaction")
                if key in workflow
            },
        }
