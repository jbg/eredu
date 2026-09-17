use super::*;

fn derive_field(attributes: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    expand_parameterized(syn::parse2(quote! {
        #[parameterized(tensor = "i32")]
        struct Fixture { #attributes value: i32 }
    })?)
}

#[test]
fn retained_annotations_require_skipping_editable_topology() {
    for attributes in [
        quote!(#[parameter(metadata)]),
        quote!(#[parameter(retained_value)]),
        quote!(#[parameter(retained_optional_value)]),
    ] {
        assert_eq!(
            derive_field(attributes).unwrap_err().to_string(),
            "parameter retained-value annotations require skip"
        );
    }
}

#[test]
fn retained_annotations_are_mutually_exclusive_across_attributes() {
    for attributes in [
        quote!(#[parameter(skip, metadata, retained_value)]),
        quote!(#[parameter(skip, metadata, retained_optional_value)]),
        quote!(#[parameter(skip, retained_value, retained_optional_value)]),
        quote!(#[parameter(skip, metadata)] #[parameter(retained_value)]),
    ] {
        assert_eq!(
            derive_field(attributes).unwrap_err().to_string(),
            "parameter metadata, retained_value and retained_optional_value are mutually exclusive"
        );
    }
}

#[test]
fn retained_annotation_validation_also_applies_to_enum_fields() {
    let input = syn::parse_quote! {
        #[parameterized(tensor = "i32")]
        enum Fixture { Empty, Value(#[parameter(retained_value)] i32) }
    };
    assert_eq!(
        expand_parameterized(input).unwrap_err().to_string(),
        "parameter retained-value annotations require skip"
    );
}
