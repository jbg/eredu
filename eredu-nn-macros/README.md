# eredu-nn-macros

`eredu-nn-macros` provides derive macros used by `eredu-nn`. Its
`Parameterized` derive implements backend-neutral parameter traversal for
neural modules while allowing fields to be skipped explicitly.

This is a support crate. Most users should depend on
[`eredu-nn`](https://github.com/jbg/eredu/tree/main/eredu-nn), which re-exports
the derive macro alongside the traits it implements.

## License

Licensed under either Apache-2.0 or MIT.
