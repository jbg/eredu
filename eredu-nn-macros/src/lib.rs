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
/// with `#[parameter(skip)]`.
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
    let mutable = traversal(&input.data, Traversal::Mutable)?;
    let trainable = traversal(&input.data, Traversal::Trainable)?;

    Ok(quote! {
        impl #impl_generics ::eredu_nn::Parameterized<#tensor> for #name #type_generics
        #where_clause
        {
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

fn skipped(field: &syn::Field) -> syn::Result<bool> {
    let mut skip = false;
    for attribute in &field.attrs {
        if !attribute.path().is_ident("parameter") {
            continue;
        }
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(meta.error("unsupported parameter option"))
            }
        })?;
    }
    Ok(skip)
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
                &data.union_token,
                "Parameterized cannot be derived for unions",
            ))
        }
    };
    for fields in variants {
        for field in fields {
            if !skipped(field)? {
                types.push(&field.ty);
            }
        }
    }
    Ok(types)
}

#[derive(Clone, Copy)]
enum Traversal {
    Immutable,
    Mutable,
    Trainable,
}

fn field_call(
    field_type: &Type,
    receiver: proc_macro2::TokenStream,
    traversal: Traversal,
) -> proc_macro2::TokenStream {
    match traversal {
        Traversal::Immutable => quote! {
            <#field_type as ::eredu_nn::Parameterized<_>>::visit_parameters(#receiver, visitor);
        },
        Traversal::Mutable => quote! {
            <#field_type as ::eredu_nn::Parameterized<_>>::visit_parameters_mut(#receiver, visitor);
        },
        Traversal::Trainable => quote! {
            <#field_type as ::eredu_nn::Parameterized<_>>::set_trainable(#receiver, trainable);
        },
    }
}

fn traversal(data: &Data, traversal: Traversal) -> syn::Result<proc_macro2::TokenStream> {
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
            &data.union_token,
            "Parameterized cannot be derived for unions",
        )),
    }
}

fn struct_traversal(
    fields: &Fields,
    traversal: Traversal,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut calls = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        if skipped(field)? {
            continue;
        }
        let member = field
            .ident
            .clone()
            .map(syn::Member::Named)
            .unwrap_or_else(|| syn::Member::Unnamed(index.into()));
        let receiver = match traversal {
            Traversal::Immutable => quote!(&self.#member),
            Traversal::Mutable | Traversal::Trainable => quote!(&mut self.#member),
        };
        calls.push(field_call(&field.ty, receiver, traversal));
    }
    Ok(quote! { #(#calls)* })
}

fn enum_variant(
    fields: &Fields,
    traversal: Traversal,
) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let mut bindings = Vec::new();
    let mut calls = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let binding = format_ident!("__eredu_field_{index}");
        let is_skipped = skipped(field)?;
        if !is_skipped {
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
