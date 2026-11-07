"""posts_create(draft_reviewer=...) (Post 5623): beim Anlegen genau zwei Rechte,
Schreiben fuer den freigebenden Menschen und Lesen fuer Comm (service:comm-api)."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "content")
import server  # noqa: E402


class DraftReviewerTests(unittest.TestCase):
    def test_setzt_zwei_grants(self):
        with patch.object(server, "call_content_api", new=AsyncMock(return_value={"id": 9})) as call:
            asyncio.run(server.content_posts_create("Mail", content="Hallo", collab_enabled=True,
                                                    visibility="restricted",
                                                    draft_reviewer=" A.Popovic@edera-safety.com "))
        body = call.call_args.kwargs["json_body"]
        self.assertEqual(body["grants"], [
            {"principal_type": "email", "principal_id": "a.popovic@edera-safety.com", "permission": "write"},
            {"principal_type": "service", "principal_id": "service:comm-api", "permission": "read"},
        ])

    def test_ohne_restricted_abgelehnt(self):
        with patch.object(server, "call_content_api", new=AsyncMock()) as call:
            out = asyncio.run(server.content_posts_create("Mail", draft_reviewer="a@b.at"))
        self.assertEqual(out["error"], "draft_reviewer_needs_restricted")
        call.assert_not_called()

    def test_ohne_reviewer_keine_grants(self):
        with patch.object(server, "call_content_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.content_posts_create("X"))
        self.assertNotIn("grants", call.call_args.kwargs["json_body"])
