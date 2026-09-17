//! Derive support for backend-neutral neural parameter traversal.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, parse_quote, Data, DeriveInput, Fields, LitStr, Type, WherePredicate,
};

/// Derives `eredu_nn::Parameterized` by recursively visiting every field.
///
/// The container must declare its tensor type with
/// `#[parameterized(tensor = "B::Tensor")]`. Individual fields may opt out
/// with `#[parameter(skip)]`, which leaves retained-value coverage unknown.
/// Add `metadata` for payload-free descriptions, `retained_value` for one raw
/// tensor, or `retained_optional_value` for an `Option` of raw tensors. These
/// annotations do not add editable parameter slots.
#[proc_macro_derive(Parameterized, attributes(parameterized, parameter))]
pub fn derive_parameterized(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_parameterized(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_parameterized(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let tensor = tensor_type(&input)?;
    let name = &input.ident;
    let mut generics = input.generics.clone();
    let fields = all_included_field_types(&input.data)?;
    {
        let where_clause = generics.make_where_clause();
        where_clause.predicates.push(parse_quote!(#tensor: 'static));
        for field in fields {
            let predicate: WherePredicate =
                parse_quote!(#field: ::eredu_nn::Parameterized<#tensor>);
            where_clause.predicates.push(predicate);
        }
    }
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let immutable = traversal(&input.data, Traversal::Immutable)?;
    let sources = traversal(&input.data, Traversal::Sources)?;
    let mutable = traversal(&input.data, Traversal::Mutable)?;
    let trainable = traversal(&input.data, Traversal::Trainable)?;
    let retained = traversal(&input.data, Traversal::Retained)?;
    let bound = traversal(&input.data, Traversal::Bound(&tensor))?;

    Ok(quote! {
        impl #impl_generics ::eredu_nn::Parameterized<#tensor> for #name #type_generics
        #where_clause
        {
            fn visit_parameter_sources<'__eredu, __EreduVisitor>(
                &'__eredu self, visitor: &mut __EreduVisitor,
            ) -> Result<(), ::eredu_nn::ParameterSourceError>
            where __EreduVisitor: ::eredu_nn::ParameterSourceVisitor<'__eredu, #tensor> {
                let mut __eredu_source_result = Ok(());
                #sources
                __eredu_source_result
            }

            fn visit_parameters<'__eredu, __EreduVisitor>(
                &'__eredu self,
                visitor: &mut __EreduVisitor,
            ) where
                __EreduVisitor: ::eredu_nn::ParameterVisitor<'__eredu, #tensor>,
            {
                #immutable
            }

            fn visit_parameters_mut<'__eredu, __EreduVisitor>(
                &'__eredu mut self,
                visitor: &mut __EreduVisitor,
            ) where
                __EreduVisitor: ::eredu_nn::ParameterVisitorMut<'__eredu, #tensor>,
            {
                #mutable
            }

            fn set_trainable(&mut self, trainable: bool) {
                #trainable
            }

            fn retained_value_slot_bound(&self) -> Option<usize> {
                let mut __eredu_bound = Some(0usize);
                #bound
                __eredu_bound
            }

            fn visit_retained_values(&self, visitor: &mut dyn FnMut(&#tensor)) -> bool {
                let mut __eredu_complete = true;
                #retained
                __eredu_complete
            }
        }
    })
}

fn tensor_type(input: &DeriveInput) -> syn::Result<Type> {
    let mut tensor = None;
    for attribute in &input.attrs {
        if !attribute.path().is_ident("parameterized") {
            continue;
        }
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("tensor") {
                let literal: LitStr = meta.value()?.parse()?;
                tensor = Some(literal.parse()?);
                Ok(())
            } else {
                Err(meta.error("unsupported parameterized option"))
            }
        })?;
    }
    tensor.ok_or_else(|| {
        syn::Error::new_spanned(
            &input.ident,
            "Parameterized derive requires #[parameterized(tensor = \"...\")]",
        )
    })
}

#[derive(Clone, Copy, Default)]
struct FieldOptions {
    skip: bool,
    metadata: bool,
    retained_value: bool,
    retained_optional_value: bool,
}

fn field_options(field: &syn::Field) -> syn::Result<FieldOptions> {
    let mut options = FieldOptions::default();
    for attribute in &field.attrs {
        if !attribute.path().is_ident("parameter") {
            continue;
        }
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                options.skip = true;
                Ok(())
            } else if meta.path.is_ident("metadata") {
                options.metadata = true;
                Ok(())
            } else if meta.path.is_ident("retained_value") {
                options.retained_value = true;
                Ok(())
            } else if meta.path.is_ident("retained_optional_value") {
                options.retained_optional_value = true;
                Ok(())
            } else {
                Err(meta.error("unsupported parameter option"))
            }
        })?;
    }
    let retained_annotations = usize::from(options.metadata)
        + usize::from(options.retained_value)
        + usize::from(options.retained_optional_value);
    if retained_annotations > 1 {
        return Err(syn::Error::new_spanned(
            field,
            "parameter metadata, retained_value and retained_optional_value are mutually exclusive",
        ));
    }
    if retained_annotations != 0 && !options.skip {
        return Err(syn::Error::new_spanned(
            field,
            "parameter retained-value annotations require skip",
        ));
    }
    Ok(options)
}

fn all_included_field_types(data: &Data) -> syn::Result<Vec<&Type>> {
    let mut types = Vec::new();
    let variants: Vec<&Fields> = match data {
        Data::Struct(data) => vec![&data.fields],
        Data::Enum(data) => data
            .variants
            .iter()
            .map(|variant| &variant.fields)
            .collect(),
        Data::Union(data) => {
            return Err(syn::Error::new_spanned(
                data.union_token,
                "Parameterized cannot be derived for unions",
            ))
        }
    };
    for fields in variants {
        for field in fields {
            if !field_options(field)?.skip {
                types.push(&field.ty);
            }
        }
    }
    Ok(types)
}

#[derive(Clone, Copy)]
enum Traversal<'a> {
    Immutable,
    Sources,
    Mutable,
    Trainable,
    Retained,
    Bound(&'a Type),
}

fn field_call(
    field_type: &Type,
    receiver: proc_macro2::TokenStream,
    traversal: Traversal<'_>,
) -> proc_macro2::TokenStream {
    match traversal {
        Traversal::Sources => quote! {
            __eredu_source_result = __eredu_source_result.and(
                <#field_type as ::eredu_nn::Parameterized<_>>::visit_parameter_sources(#receiver, visitor));
        },
        Traversal::Immutable => quote! {
            <#field_type as ::eredu_nn::Parameterized<_>>::visit_parameters(#receiver, visitor);
        },
        Traversal::Mutable => quote! {
            <#field_type as ::eredu_nn::Parameterized<_>>::visit_parameters_mut(#receiver, visitor);
        },
        Traversal::Trainable => quote! {
            <#field_type as ::eredu_nn::Parameterized<_>>::set_trainable(#receiver, trainable);
        },
        Traversal::Bound(tensor) => quote! {
            __eredu_bound = __eredu_bound.and_then(|sum| {
                sum.checked_add(<#field_type as ::eredu_nn::Parameterized<#tensor>>::retained_value_slot_bound(#receiver)?)
            });
        },
        Traversal::Retained => quote! {
            __eredu_complete &= <#field_type as ::eredu_nn::Parameterized<_>>::visit_retained_values(#receiver, visitor);
        },
    }
}

fn skipped_retained_call(
    options: FieldOptions,
    receiver: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if options.metadata {
        quote!()
    } else if options.retained_value {
        quote!(visitor(#receiver);)
    } else if options.retained_optional_value {
        quote! {
            if let Some(__eredu_value) = #receiver {
                visitor(__eredu_value);
            }
        }
    } else {
        quote!(__eredu_complete = false;)
    }
}

fn skipped_source_call(
    options: FieldOptions,
    receiver: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if options.metadata {
        quote!()
    } else if options.retained_value {
        quote!(visitor.retained(#receiver);)
    } else if options.retained_optional_value {
        quote! { if let Some(value) = #receiver { visitor.retained(value); } }
    } else {
        quote! { __eredu_source_result = __eredu_source_result.and(Err(::eredu_nn::ParameterSourceError::UnclassifiedRetainedField)); }
    }
}

fn skipped_bound_call(options: FieldOptions) -> proc_macro2::TokenStream {
    if options.metadata {
        quote!()
    } else if options.retained_value || options.retained_optional_value {
        // An explicit tensor/optional tensor field always needs at most one slot.
        quote!(__eredu_bound = __eredu_bound.and_then(|sum| sum.checked_add(1));)
    } else {
        quote!(__eredu_bound = None;)
    }
}

fn traversal(data: &Data, traversal: Traversal<'_>) -> syn::Result<proc_macro2::TokenStream> {
    match data {
        Data::Struct(data) => struct_traversal(&data.fields, traversal),
        Data::Enum(data) => {
            let mut arms = Vec::new();
            for variant in &data.variants {
                let variant_name = &variant.ident;
                let (pattern, calls) = enum_variant(&variant.fields, traversal)?;
                arms.push(quote! { Self::#variant_name #pattern => { #calls } });
            }
            Ok(quote! { match self { #(#arms),* } })
        }
        Data::Union(data) => Err(syn::Error::new_spanned(
            data.union_token,
            "Parameterized cannot be derived for unions",
        )),
    }
}

fn struct_traversal(
    fields: &Fields,
    traversal: Traversal<'_>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut calls = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let options = field_options(field)?;
        let member = field
            .ident
            .clone()
            .map(syn::Member::Named)
            .unwrap_or_else(|| syn::Member::Unnamed(index.into()));
        let receiver = match traversal {
            Traversal::Immutable
            | Traversal::Sources
            | Traversal::Retained
            | Traversal::Bound(_) => {
                quote!(&self.#member)
            }
            Traversal::Mutable | Traversal::Trainable => quote!(&mut self.#member),
        };
        if options.skip {
            if matches!(traversal, Traversal::Sources) {
                calls.push(skipped_source_call(options, receiver));
            } else if matches!(traversal, Traversal::Retained) {
                calls.push(skipped_retained_call(options, receiver));
            } else if matches!(traversal, Traversal::Bound(_)) {
                calls.push(skipped_bound_call(options));
            }
        } else {
            calls.push(field_call(&field.ty, receiver, traversal));
        }
    }
    Ok(quote! { #(#calls)* })
}

fn enum_variant(
    fields: &Fields,
    traversal: Traversal<'_>,
) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let mut bindings = Vec::new();
    let mut calls = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let binding = format_ident!("__eredu_field_{index}");
        let options = field_options(field)?;
        let is_skipped = options.skip
            && !(matches!(traversal, Traversal::Retained | Traversal::Sources)
                && (options.retained_value || options.retained_optional_value));
        if options.skip {
            if matches!(traversal, Traversal::Sources) {
                calls.push(skipped_source_call(options, quote!(#binding)));
            } else if matches!(traversal, Traversal::Retained) {
                calls.push(skipped_retained_call(options, quote!(#binding)));
            } else if matches!(traversal, Traversal::Bound(_)) {
                calls.push(skipped_bound_call(options));
            }
        } else {
            calls.push(field_call(&field.ty, quote!(#binding), traversal));
        }
        bindings.push((field, binding, is_skipped));
    }
    let pattern = match fields {
        Fields::Named(_) => {
            let entries = bindings.iter().map(|(field, binding, skipped)| {
                let name = field.ident.as_ref().expect("named field");
                if *skipped {
                    quote!(#name: _)
                } else {
                    quote!(#name: #binding)
                }
            });
            quote!({ #(#entries),* })
        }
        Fields::Unnamed(_) => {
            let entries = bindings.iter().map(|(_, binding, skipped)| {
                if *skipped {
                    quote!(_)
                } else {
                    quote!(#binding)
                }
            });
            quote!(( #(#entries),* ))
        }
        Fields::Unit => quote!(),
    };
    Ok((pattern, quote! { #(#calls)* }))
}

#[cfg(test)]
mod tests;
