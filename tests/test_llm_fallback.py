import asyncio

from spatial_memory.llm.base import MemoryDraft, QueryDraft
from spatial_memory.llm.fallback import FallbackInterpreter
from spatial_memory.llm.rules import RuleBasedInterpreter


class FakePrimary:
    async def memory_from_observation(self, transcript, detected_label):
        return MemoryDraft(
            name="robot",
            description="LLM description",
            tags=["machine"],
            aliases=[],
            metadata={"llm": "yes"},
        )

    async def query(self, text):
        return QueryDraft(search_text=text)


def test_explicit_human_name_overrides_primary_llm_name() -> None:
    interpreter = FallbackInterpreter(
        primary=FakePrimary(),
        fallback=RuleBasedInterpreter(),
    )
    draft = asyncio.run(
        interpreter.memory_from_observation(
            "これはロボットです",
            "robot",
        )
    )

    assert draft.name == "ロボット"
    assert "robot" in draft.aliases
    assert "machine" in draft.tags
    assert draft.metadata["human_explicit_name"] == "ロボット"
    assert draft.metadata["llm"] == "yes"


def test_non_explicit_text_keeps_primary_llm_name() -> None:
    interpreter = FallbackInterpreter(
        primary=FakePrimary(),
        fallback=RuleBasedInterpreter(),
    )
    draft = asyncio.run(
        interpreter.memory_from_observation(
            "机の近くに置いた",
            "robot",
        )
    )
    assert draft.name == "robot"
