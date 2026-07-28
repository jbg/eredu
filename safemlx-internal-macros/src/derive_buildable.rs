use quote::quote;
use syn::DeriveInput;

use crate::shared::{PathOrIdent, Result};

#[allow(dead_code)]
pub(crate) struct StructProperty {
    pub ident: syn::Ident,

    /// Generate builder if None
    pub builder: Option<syn::Path>,

    /// Rename `safemlx` if Some(_)
    pub root: Option<syn::Path>,
}

impl StructProperty {
    pub(crate) fn from_derive_input(input: &DeriveInput) -> syn::Result<Self> {
        let mut builder = None;
        let mut root = None;

        for attr in input
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("buildable"))
        {
            if matches!(attr.meta, syn::Meta::Path(_)) {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                let slot = if meta.path.is_ident("builder") {
                    &mut builder
                } else if meta.path.is_ident("root") {
                    &mut root
                } else {
                    return Err(meta.error("unsupported `buildable` option"));
                };
                if slot.is_some() {
                    return Err(meta.error("duplicate `buildable` option"));
                }
                *slot = Some(meta.value()?.parse()?);
                Ok(())
            })?;
        }

        Ok(Self {
            ident: input.ident.clone(),
            builder,
            root,
        })
    }
}

pub(crate) fn expand_derive_buildable(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    let struct_prop = StructProperty::from_derive_input(&input)?;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let struct_ident = &struct_prop.ident;
    let builder_ident = syn::Ident::new(&format!("{struct_ident}Builder"), struct_ident.span());
    let root = match struct_prop.root {
        Some(path) => path,
        None => syn::parse_quote!(::safemlx),
    };

    let struct_builder_ident = match &struct_prop.builder {
        Some(path) => PathOrIdent::Path(path.clone()),
        None => PathOrIdent::Ident(builder_ident),
    };

    let impl_buildable = quote! {
        impl #impl_generics #root::builder::Buildable for #struct_ident #type_generics #where_clause {
            type Builder = #struct_builder_ident #type_generics;
        }
    };

    Ok(quote! {
        #impl_buildable
    })
}
