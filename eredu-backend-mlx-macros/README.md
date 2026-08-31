# eredu-backend-mlx-macros

Backend-internal derives for MLX module parameter traversal. Struct derives
traverse fields marked with `#[param]`; enum derives delegate traversal through
each single-field tuple variant. Direct use is normally unnecessary; depend on
`eredu-backend-mlx` for MLX backend facilities.

Licensed under either Apache-2.0 or MIT.
