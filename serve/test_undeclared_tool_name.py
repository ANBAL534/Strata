"""serve/test_undeclared_tool_name.py - a call naming a tool the request did not declare is repaired, not emitted.

Seen live through OpenCodex (Codex CLI -> opencodex -> /v1/responses): the model wrapped its call in
<function=tools> - the template's own block name.  Strata emitted a function_call named "tools", the router
rejected it ("routed provider emitted undeclared client tool 'tools'; only request-declared tools may be
called") and the stream died mid-turn, cutting the conversation.  The parser now maps an undeclared name to the
declared tool it plainly meant (case, spacing, quotes, ns.tool for ns__tool, a long name's unique suffix), takes
a declared call nested inside a "tools" wrapper as the call, and text-ifies what it cannot map: the turn
completes and the agent loop goes on.

    python -m unittest serve.test_undeclared_tool_name -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import responses  # noqa: E402
from serve.frontend import Event, OutputParser, ToolCall, repair_call_name  # noqa: E402

SCHEMA = [{"name": "capture", "parameters": {"properties": {"content": {"type": "string"}}}},
          {"name": "mcp__node_repl__js", "parameters": {"properties": {"code": {"type": "string"}}}}]
INNER = "<function=capture>\n<parameter=content>\n42\n</parameter>\n</function>"
WRAP_OPEN = "<tool_call>\n<function=tools>\n" + INNER + "\n</tool_call>"        # the live shape: no outer close
WRAP_CLOSED = "<tool_call>\n<function=tools>\n" + INNER + "\n</function>\n</tool_call>"
THINKED = "Done.</think>\n\n"                              # the call is written after the thinking


def events(text, tools=SCHEMA, repair=True, stream_tools=True):
    p = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools, repair_names=repair)
    evs = []
    for i in range(0, len(text), 3):                   # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + 3])
    evs += p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class NestedWrapper(unittest.TestCase):
    def test_wrapped_call_after_the_thinking_is_the_call_the_model_meant(self):
        for text in (WRAP_OPEN, WRAP_CLOSED):
            for st in (False, True):
                with self.subTest(closed=text is WRAP_CLOSED, stream_tools=st):
                    self.assertEqual(calls(events(THINKED + text, stream_tools=st)),
                                     [("capture", {"content": "42"})])

    def test_wrapped_call_inside_the_thinking_is_rescued_the_same_way(self):
        for text in (WRAP_OPEN, WRAP_CLOSED):
            for st in (False, True):
                with self.subTest(closed=text is WRAP_CLOSED, stream_tools=st):
                    self.assertEqual(calls(events(text, stream_tools=st)), [("capture", {"content": "42"})])

    def test_nothing_undeclared_is_announced_while_streaming(self):
        evs = events(THINKED + WRAP_OPEN)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], [])
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_args"], [])
        self.assertEqual(calls(evs), [("capture", {"content": "42"})])


class SloppyNames(unittest.TestCase):
    def test_the_name_written_with_case_spaces_or_quotes_finds_its_tool(self):
        for name in (" Capture ", "Capture", "\"capture\""):
            text = THINKED + "<tool_call>\n<function=" + name + ">\n<parameter=content>\n42\n</parameter>\n</function>\n</tool_call>"
            for st in (False, True):
                with self.subTest(name=name, stream_tools=st):
                    self.assertEqual(calls(events(text, stream_tools=st)), [("capture", {"content": "42"})])

    def test_the_namespace_written_with_dots_finds_its_tool(self):
        text = THINKED + "<tool_call>\n<function=mcp.node_repl.js>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls(events(text)), [("mcp__node_repl__js", {"code": "x"})])

    def test_a_long_namespaced_name_written_as_its_unique_suffix_finds_its_tool(self):
        text = THINKED + "<tool_call>\n<function=node_repl__js>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls(events(text)), [("mcp__node_repl__js", {"code": "x"})])

    def test_one_declared_tool_and_no_name_written_is_that_call(self):
        text = THINKED + "<tool_call>\n<function=>\n<parameter=content>\n42\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls(events(text, tools=[SCHEMA[0]])), [("capture", {"content": "42"})])

    def test_the_block_word_written_as_a_prefix_finds_its_tool(self):
        for name in ("tools/write_stdin", "tools.write_stdin", "tools__write_stdin", "tools:write_stdin",
                     "Tools/Write_Stdin"):
            text = (THINKED + "<tool_call>\n<function=" + name + ">\n<parameter=cmd>\nls\n</parameter>\n</function>\n"
                    "</tool_call>")
            for st in (False, True):
                with self.subTest(name=name, stream_tools=st):
                    self.assertEqual(calls(events(text, tools=[{"name": "write_stdin",
                                                                "parameters": {"properties": {"cmd": {"type": "string"}}}}],
                                                  stream_tools=st)), [("write_stdin", {"cmd": "ls"})])


class UnmappableNames(unittest.TestCase):
    def test_a_format_word_with_no_declared_call_inside_is_the_text_it_is(self):
        text = THINKED + "<tool_call>\n<function=tools>\n<parameter=content>\n42\n</parameter>\n</function>\n</tool_call>"
        for st in (False, True):
            with self.subTest(stream_tools=st):
                evs = events(text, stream_tools=st)
                self.assertEqual(calls(evs), [])
                self.assertIn("<function=tools>", "".join(e.text for e in evs if e.kind == "content"))

    def test_a_short_ambiguous_suffix_is_not_guessed(self):
        text = THINKED + "<tool_call>\n<function=js>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls(events(text)), [])

    def test_repair_off_keeps_the_old_behaviour(self):
        self.assertEqual(calls(events(THINKED + WRAP_OPEN, repair=False)), [("tools", {"content": 42})])

    def test_a_call_cut_off_by_max_tokens_stays_reasoning_text(self):
        p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True)
        evs = p.feed("<tool_call>\n<function=tools>\n<function=capture>\n<parameter=content>\n42")
        evs += p.finish("length")                       # #1058: a cut turn delivers no calls
        self.assertEqual(calls(evs), [])


class DeclaredUnchanged(unittest.TestCase):
    def test_a_declared_call_streams_as_before(self):
        text = THINKED + "<tool_call>\n<function=capture>\n<parameter=content>\n42\n</parameter>\n</function>\n</tool_call>"
        evs = events(text)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], ["capture"])
        self.assertEqual(calls(evs), [("capture", {"content": "42"})])


class RepairName(unittest.TestCase):
    def test_the_map(self):
        self.assertEqual(repair_call_name("Capture", {"capture": {}}), "capture")
        self.assertEqual(repair_call_name("mcp.a.b", {"mcp__a__b": {}}), "mcp__a__b")
        self.assertEqual(repair_call_name("a__b", {"mcp__a__b": {}}), "mcp__a__b")
        self.assertEqual(repair_call_name("tools/write_stdin", {"write_stdin": {}}), "write_stdin")
        self.assertEqual(repair_call_name("tools/capture", {"capture": {}, "x": {}}), "capture")
        self.assertIsNone(repair_call_name("tools/zzz", {"capture": {}}))
        self.assertIsNone(repair_call_name("tools", {"capture": {}}))
        self.assertIsNone(repair_call_name("js", {"mcp__a__js": {}, "mcp__b__js": {}}))   # two tools end with it
        self.assertIsNone(repair_call_name("zzz", {"capture": {}}))
        self.assertIsNone(repair_call_name("capture", {}))                                # no tools declared


class ResponsesPath(unittest.TestCase):
    def test_the_responses_answer_completes_with_the_declared_call(self):
        req = {"input": "hi", "tools": [{"type": "function", "name": "capture",
                                         "parameters": {"type": "object",
                                                        "properties": {"content": {"type": "string"}}}}]}
        tools, names, _ = responses.request_tools(req)
        asm = responses.Assembler(req, "m", 0, names, False, json_mode=False)
        out = asm.start()
        for e in events(THINKED + WRAP_OPEN, tools=tools):
            out += asm.feed(e)
        out += asm.finish({"finish": "stop", "completion_tokens": 5})
        self.assertEqual(out[-1]["type"], "response.completed")
        fcs = [i for i in out[-1]["response"]["output"] if i["type"] == "function_call"]
        self.assertEqual([(i["name"], i["arguments"]) for i in fcs], [("capture", '{"content": "42"}')])
        self.assertEqual([e.get("name") for e in out if e["type"] == "response.function_call_arguments.done"],
                         ["capture"])

    def test_a_sloppy_name_maps_through_the_requests_spellings(self):
        req = {"input": "hi", "tools": [{"type": "namespace", "name": "mcp__node",
                                         "tools": [{"type": "function", "name": "js",
                                                    "parameters": {"type": "object",
                                                                   "properties": {"code": {"type": "string"}}}}]}]}
        _, names, _ = responses.request_tools(req)
        asm = responses.Assembler(req, "m", 0, names, False)
        item = asm._call_item(Event("tool_call", call=ToolCall(name="node__js", arguments={"code": "x"})))
        self.assertEqual((item["type"], item["name"], item["namespace"]), ("function_call", "js", "mcp__node"))


class UnknownToolRetryParser(unittest.TestCase):
    def test_unknown_complete_call_becomes_internal_error_only_when_enabled(self):
        text = "</think>\n\n<tool_call>\n<function=not_a_tool>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>"
        evs = events(text, repair=True)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=not_a_tool>", "".join(e.text for e in evs if e.kind == "content"))

        parser = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, repair_names=True,
                              retry_unknown_tools=True)
        got = []
        for i in range(0, len(text), 3):
            got += parser.feed(text[i:i + 3])
        self.assertEqual([e.kind for e in got if e.kind in ("tool_start", "tool_args", "tool_call")], [])
        errors = [e for e in got if e.kind == "tool_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual((errors[0].call.name, errors[0].call.arguments), ("not_a_tool", {"code": "x"}))

    def test_call_in_reasoning_is_held_until_the_model_stops(self):
        text = "<tool_call>\n<function=not_a_tool>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call></think>\n\n"
        parser = OutputParser(thinking=True, tools=SCHEMA, retry_unknown_tools=True)
        got = parser.feed(text)
        self.assertEqual([e.kind for e in got if e.kind == "tool_error"], ["tool_error"])

    def test_unterminated_call_never_becomes_an_error(self):
        parser = OutputParser(thinking=True, tools=SCHEMA, retry_unknown_tools=True)
        got = parser.feed("</think>\n\n<tool_call>\n<function=not_a_tool>\n<parameter=code>\nx")
        got += parser.finish("length")
        self.assertFalse([e for e in got if e.kind == "tool_error"])


if __name__ == "__main__":
    unittest.main()

