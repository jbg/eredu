//! Operations

mod arithmetic;
mod conversion;
mod convolution;
mod cumulative;
mod factory;
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
pub use logical::*;
pub use other::*;
pub use quantization::*;
pub use reduction::*;
pub use shapes::*;
pub use sort::*;

mod graph_rows;
pub use graph_rows::{OriginalArrayRows, OriginalArrayRowsLayout, OriginalCopyWorkerLayout,
    reshape_like_prefix, reshape_like_prefix_control_bytes};

mod ordinary_recipe;
pub use ordinary_recipe::{OrdinaryRecipeCall, OrdinaryRecipeWrapperControls};
pub use ordinary_recipe::ordinary_array_result_guard_control_bytes;
