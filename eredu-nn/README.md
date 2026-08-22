# eredu-nn

`eredu-nn` is Eredu's backend-neutral neural-compute boundary. Architecture
crates are generic over the `Tensor` trait, while each backend implements that
trait for its own opaque tensor handle and execution context.

The boundary does not exchange tensor data. Architecture calls are statically
dispatched after monomorphization, and operations such as linear projection,
layer normalization, rotary encoding, and scaled dot-product attention remain
backend fusion points. A backend therefore retains control of storage, device
placement, graph construction, laziness, synchronization, and kernel fusion.

`eredu-nn` has no concrete-backend features or accelerator dependencies.
`eredu-backend-mlx` implements the contract with its local `MlxTensor` newtype,
a transparent, zero-copy wrapper around the native MLX array handle. This
keeps the neutral contract independent while satisfying Rust's orphan rules.
Future backends implement the same contract once; models in backend-neutral
architecture crates can then be reused without being ported.
