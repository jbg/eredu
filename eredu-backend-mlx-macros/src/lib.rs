extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod module_parameters;
mod util;

/// Derive the backend's `ModuleParameters` traversal implementation.
#[proc_macro_derive(ModuleParameters, attributes(module, param))]
pub fn derive_module_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    module_parameters::expand_module_parameters(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
