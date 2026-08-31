extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod module_parameters;
mod util;

/// Derive the backend's `PhysicalParameters` traversal implementation.
#[proc_macro_derive(PhysicalParameters, attributes(module, param))]
pub fn derive_physical_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    module_parameters::expand_physical_parameters(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
