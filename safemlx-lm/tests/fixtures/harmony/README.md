# OpenAI Harmony protocol fixtures

These text fixtures are copied from the authoritative `openai/harmony`
repository at revision `abd677f7ac962629c808197caa1feb9e3e95d2b0`.
They are raw Harmony text, not locally reconstructed token decodes.

| Fixture | Upstream source |
| --- | --- |
| `reasoning-function-call-abd677f7.txt` | `docs/format.md`, “Receiving tool calls” example |
| `preamble-function-call-abd677f7.txt` | `docs/format.md`, “Preambles” example |
| `prior-call-result-abd677f7.txt` | `test-data/test_does_not_drop_if_ongoing_analysis.txt` |

The official examples demonstrate why Harmony is outside the bounded
declarative dialect: a function call can follow analysis and visible
commentary messages; recipient metadata can occur on either side of the
channel metadata; and the `<|call|>` action terminator is also a sampling stop.
