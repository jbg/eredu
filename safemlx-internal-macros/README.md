# safemlx-internal-macros

`safemlx-internal-macros` contains procedural macros used to implement
`safemlx`, including builder generation and internal test expansion.

This crate is published so released versions of `safemlx` can depend on it. It
is an implementation detail and does not provide a stable application-facing
API. Applications should depend on
[`safemlx`](https://github.com/jbg/eredu/tree/main/safemlx) instead.

## License

Licensed under either Apache-2.0 or MIT.
