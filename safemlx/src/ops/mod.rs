//! Operations

mod arithmetic;
mod conversion;
mod convolution;
mod cumulative;
mod factory;
#[cfg(target_vendor = "apple")]
mod host_projection;
mod io;
mod logical;
mod other;
mod quantization;
mod reduction;
mod shapes;
mod sort;

pub mod indexing;

pub use arithmetic::*;
pub use conversion::*;
pub use convolution::*;
pub use cumulative::*;
pub use factory::*;
#[cfg(target_vendor = "apple")]
pub use host_projection::*;
pub use logical::*;
pub use other::*;
pub use quantization::*;
pub use reduction::*;
pub use shapes::*;
pub use sort::*;
