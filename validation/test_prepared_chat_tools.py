"""Acceptance checks for the released-run evidence consumer, not model evidence."""

import copy
import unittest

from validation.prepared_chat_tools import validate_events


def complete_call():
    return [
        {"ToolCallStart": {"index": 0, "id": "call_0", "name": "reading"}},
        {"ToolArgumentsDelta": {"index": 0, "json_fragment": '{"value":'}},
        {"ToolArgumentsDelta": {"index": 0, "json_fragment": '17}'}},
        "ToolCallEnd",
        {"Finished": {"reason": "grammar_complete"}},
    ]


class SemanticEvidenceTests(unittest.TestCase):
    def test_combines_fragmented_arguments_and_preserves_actual_output(self):
        events = [{"TextDelta": "Calling the tool."}, *complete_call()]
        result = validate_events(events)
        self.assertEqual(result["tool_calls"][0]["arguments"], {"value": 17})
        self.assertEqual(result["visible_text"], "Calling the tool.")
        self.assertEqual(result["finish_reason"], "grammar_complete")

    def test_rejects_incomplete_wrong_or_ambiguous_calls(self):
        invalid = [complete_call()[:-1], complete_call()[1:]]
        case = complete_call()
        case[-1]["Finished"]["reason"] = "max_tokens"
        invalid.append(case)
        case = complete_call()
        case[2]["ToolArgumentsDelta"]["index"] = 1
        invalid.append(case)
        case = complete_call()
        case[2]["ToolArgumentsDelta"]["json_fragment"] = "18}"
        invalid.append(case)
        case = complete_call()
        case[2]["ToolArgumentsDelta"]["json_fragment"] = '16,"value":17}'
        invalid.append(case)
        case = complete_call()
        case[2]["ToolArgumentsDelta"]["json_fragment"] = '17,"extra":true}'
        invalid.append(case)
        case = complete_call()
        case[0]["ToolCallStart"]["name"] = "other"
        invalid.append(case)
        invalid.append([*complete_call(), {"TextDelta": "after finish"}])
        case = complete_call()
        case.insert(1, {"TextDelta": "inside a call"})
        invalid.append(case)
        second = copy.deepcopy(complete_call()[:-1])
        second[0]["ToolCallStart"]["index"] = 1
        for event in second[1:3]:
            event["ToolArgumentsDelta"]["index"] = 1
        invalid.append([*complete_call()[:-1], *second, complete_call()[-1]])
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                validate_events(events)


if __name__ == "__main__":
    unittest.main()
