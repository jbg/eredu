extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemEnum, ItemFn};

mod generate_macro;

#[doc(hidden)]
#[proc_macro]
pub fn generate_test_cases(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemEnum);
    let name = &input.ident;

    let tests = quote! {
        /// MLX's rules for promoting two dtypes.
        #[rustfmt::skip]
        const TYPE_RULES: [[Dtype; 14]; 14] = [
            // bool             uint8               uint16              uint32              uint64              int8                int16               int32               int64               float16             float32             float64,           bfloat16            complex64
            [Dtype::Bool,       Dtype::Uint8,       Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int8,        Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // bool
            [Dtype::Uint8,      Dtype::Uint8,       Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int16,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint8
            [Dtype::Uint16,     Dtype::Uint16,      Dtype::Uint16,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int32,       Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint16
            [Dtype::Uint32,     Dtype::Uint32,      Dtype::Uint32,      Dtype::Uint32,      Dtype::Uint64,      Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint32
            [Dtype::Uint64,     Dtype::Uint64,      Dtype::Uint64,      Dtype::Uint64,      Dtype::Uint64,      Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // uint64
            [Dtype::Int8,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int8,        Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int8
            [Dtype::Int16,      Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int16,       Dtype::Int16,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int16
            [Dtype::Int32,      Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float32,     Dtype::Int32,       Dtype::Int32,       Dtype::Int32,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int32
            [Dtype::Int64,      Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float32,     Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Int64,       Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // int64
            [Dtype::Float16,    Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float16,     Dtype::Float32,     Dtype::Float64,    Dtype::Float32,     Dtype::Complex64], // float16
            [Dtype::Float32,    Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float32,     Dtype::Float64,    Dtype::Float32,     Dtype::Complex64], // float32
            [Dtype::Float64,    Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,     Dtype::Float64,    Dtype::Float64,     Dtype::Complex64], // Dtype::Float64
            [Dtype::Bfloat16,   Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Bfloat16,    Dtype::Float32,     Dtype::Float32,     Dtype::Float64,    Dtype::Bfloat16,    Dtype::Complex64], // bfloat16
            [Dtype::Complex64,  Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,   Dtype::Complex64,  Dtype::Complex64,   Dtype::Complex64], // complex64
        ];

        #[cfg(test)]
        mod generated_tests {
            use super::*;
            use strum::IntoEnumIterator;
            use pretty_assertions::assert_eq;

            #[test]
            fn test_all_combinations() {
                for a in #name::iter() {
                    for b in #name::iter() {
                        let result = a.promote_with(b);
                        let expected = TYPE_RULES[a as usize][b as usize];
                        assert_eq!(result, expected, "{}", format!("Failed promotion test for {:?} and {:?}", a, b));
                    }
                }
            }
        }
    };

    TokenStream::from(quote! {
        #input
        #tests
    })
}

/// Generate a macro that expands to the given function for ergonomic purposes.
///
/// See `safemlx-tests/tests/test_generate_macro.rs` for more usage examples.
///
/// ```rust,ignore
/// #![allow(unused_variables)]
///
/// use safemlx_internal_macros::generate_macro;
/// use safemlx::Stream;
///
/// /// Test macro generation.
/// #[generate_macro(customize(root = "$crate"))] // Default is `$crate::ops`
/// fn foo(
///     a: i32, // Mandatory argument
///     b: i32, // Mandatory argument
///     #[optional] c: Option<i32>, // Optional argument
///     #[optional] d: impl Into<Option<i32>>, // Optional argument but impl Trait
///     #[optional] stream: impl AsRef<Stream>, // stream always optional and placed at the end
/// ) -> i32 {
///     a + b + c.unwrap_or(0) + d.into().unwrap_or(0)
/// }
///
/// let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
///
/// assert_eq!(foo!(1, 2, stream = &stream), 3);
/// assert_eq!(foo!(1, 2, c = Some(3), stream = &stream), 6);
/// assert_eq!(foo!(1, 2, d = Some(4), stream = &stream), 7);
/// assert_eq!(foo!(1, 2, c = Some(3), d = Some(4), stream = &stream), 10);
/// ```
#[doc(hidden)]
#[proc_macro_attribute]
pub fn generate_macro(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = if !attr.is_empty() {
        let meta = syn::parse_macro_input!(attr as syn::Meta);
        Some(meta)
    } else {
        None
    };
    let item = parse_macro_input!(item as ItemFn);
    generate_macro::expand_generate_macro(attr, item).unwrap()
}
