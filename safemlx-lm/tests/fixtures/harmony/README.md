# Harmony protocol fixtures

These fixed text fixtures exercise OpenAI Harmony reasoning, commentary,
recipient metadata, function calls, prior call results, and action terminators.
They are raw protocol text rather than locally reconstructed token decodes.

The cases verify why Harmony needs a stateful channel-aware parser: a function
call may follow reasoning or visible commentary, recipient metadata can appear
on either side of channel metadata, and the call terminator is also a sampling
stop.
