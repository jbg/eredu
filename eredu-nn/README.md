# eredu-nn

`eredu-nn` is Eredu's backend-neutral neural-compute boundary. Architecture
crates are generic over the `NeuralBackend` trait. Each backend supplies its
opaque tensor handle as the trait's associated `Tensor` type; that handle
implements `Tensor` and identifies its execution context.

The boundary does not exchange tensor data. Architecture calls are statically
dispatched after monomorphization, and operations such as linear projection,
layer normalization, rotary encoding, and scaled dot-product attention remain
backend fusion points. A backend therefore retains control of storage, device
placement, graph construction, laziness, synchronization, and kernel fusion.

`eredu-nn` has no concrete-backend features or accelerator dependencies. Each
backend implements `NeuralBackend` for a local backend type and `Tensor` for a
local tensor newtype, which keeps the neutral contracts independent while
satisfying Rust's orphan rules. A new backend implements these contracts once;
models in backend-neutral architecture crates can then be reused without being
ported.
