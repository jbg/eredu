# safemlx-macros

`safemlx-macros` provides the public derive macros used by `safemlx` neural
modules. It implements parameter traversal with `ModuleParameters` and module
quantization traversal with `Quantizable`.

This is a support crate. Most users should depend on
[`safemlx`](https://github.com/jbg/eredu/tree/main/safemlx), which exposes the
traits and types consumed by the generated implementations.

## License

Licensed under either Apache-2.0 or MIT.
