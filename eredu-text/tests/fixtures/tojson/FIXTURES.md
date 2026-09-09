# Hugging Face JSON filter fixtures

`python-json-dumps.json` contains reference outputs generated with CPython
`json.dumps(value, ensure_ascii=False, **options)`. Each entry includes its input,
the equivalent Hugging Face Jinja filter invocation, and the expected text.
The positional argument order is `ensure_ascii, indent, separators, sort_keys`.

The fixtures cover recursive map ordering, default and explicit separators,
numeric/string/boolean indentation, Unicode and HTML characters, string escaping,
empty containers, scalar values, and floating-point notation. Tests consume the
stored outputs without requiring Python, a checkpoint, or a native backend.
