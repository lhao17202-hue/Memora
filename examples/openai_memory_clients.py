"""OpenAI client adapters for Memora examples.

These adapters are intentionally kept outside the core package. Memora's core
contracts only require a provider object with `complete(messages) -> str`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["should_remember", "memories"],
    "properties": {
        "should_remember": {"type": "boolean"},
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "type",
                    "name",
                    "description",
                    "content",
                    "tags",
                    "confidence",
                    "weight",
                    "requires_confirmation",
                    "reason",
                ],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["preference", "project", "episodic", "reflective", "tool", "knowledge", "general"],
                    },
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "weight": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
                    "requires_confirmation": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


RELATION_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "confidence", "reason", "merged"],
    "properties": {
        "kind": {"type": "string", "enum": ["none", "duplicate", "merge", "conflict"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "merged": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "description", "content", "tags"],
            "properties": {
                "name": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "content": {"type": ["string", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


class OpenAIJSONClient:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=_normalize_messages(messages),
            text={
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "schema": self.schema,
                    "strict": True,
                }
            },
            store=False,
        )
        return response.output_text


class OpenAIExtractionClient(OpenAIJSONClient):
    schema_name = "memora_memory_extraction"
    schema = EXTRACTION_JSON_SCHEMA


class OpenAIRelationClient(OpenAIJSONClient):
    schema_name = "memora_relation_decision"
    schema = RELATION_DECISION_JSON_SCHEMA


def _normalize_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role") or "user"),
            "content": str(message.get("content") or ""),
        }
        for message in messages
    ]
