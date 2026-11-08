from cc_agent.context import compact_messages, message_chars, truncate_middle


def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD" + "x" * 1000 + "TAIL"
    result = truncate_middle(text, 100)
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert "characters omitted" in result


def test_truncate_middle_leaves_short_text_alone():
    assert truncate_middle("short", 100) == "short"


def _conversation(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for index in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{index}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{index}", "content": "y" * 3000})
    return messages


def test_compaction_shrinks_old_tool_outputs_without_dropping_messages():
    messages = _conversation(10)
    compacted, count = compact_messages(messages, budget_chars=15000, keep_recent=4)
    assert count > 0
    assert len(compacted) == len(messages)
    # assistant tool calls stay paired with their tool results
    assert [m.get("tool_call_id") for m in compacted] == [m.get("tool_call_id") for m in messages]
    assert compacted[-1]["content"] == "y" * 3000  # recent results are untouched
    assert message_chars(compacted) < message_chars(messages)
    assert messages[3]["content"] == "y" * 3000  # the input list is not mutated


def test_compaction_is_a_noop_under_budget():
    messages = [{"role": "user", "content": "hi"}]
    assert compact_messages(messages, budget_chars=1000) == (messages, 0)
