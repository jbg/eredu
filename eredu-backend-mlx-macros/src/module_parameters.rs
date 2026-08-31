use syn::{DataEnum, DataStruct, DeriveInput, Fields, Generics, Ident};

use crate::util::{filter_fields_with_attr, parse_root_attribute};

pub(crate) fn expand_physical_parameters(
    input: &DeriveInput,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let root = parse_root_attribute(&input.attrs, "module")?;
    let struct_ident = &input.ident;
    let generics = &input.generics;
    match &input.data {
        syn::Data::Struct(data) => {
            expand_physical_parameters_for_struct(struct_ident, generics, data, root)
        }
        syn::Data::Enum(data) => {
            expand_physical_parameters_for_enum(struct_ident, generics, data, root)
        }
        syn::Data::Union(_) => Err(syn::Error::new_spanned(
            input,
            "PhysicalParameters cannot be derived for unions",
        )),
    }
}

fn expand_physical_parameters_for_enum(
    ident: &Ident,
    generics: &Generics,
    data: &DataEnum,
    root: Option<syn::Path>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let variants = data
        .variants
        .iter()
        .map(|variant| match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => Ok(&variant.ident),
            _ => Err(syn::Error::new_spanned(
                variant,
                "PhysicalParameters enum variants must contain exactly one unnamed field",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(impl_physical_parameters_for_enum(
        ident, generics, variants, root,
    ))
}

fn impl_physical_parameters_for_enum(
    ident: &Ident,
    generics: &Generics,
    variants: Vec<&Ident>,
    root: Option<syn::Path>,
) -> proc_macro2::TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let root = root
        .map(|root| quote::quote! { #root })
        .unwrap_or_else(|| quote::quote! { crate });

    quote::quote! {
        const _: () = {
            impl #impl_generics #root::module::PhysicalParameters for #ident #ty_generics #where_clause {
                fn freeze_parameters(&mut self, recursive: bool) {
                    match self {
                        #(Self::#variants(module) => #root::module::PhysicalParameters::freeze_parameters(module, recursive),)*
                    }
                }

                fn unfreeze_parameters(&mut self, recursive: bool) {
                    match self {
                        #(Self::#variants(module) => #root::module::PhysicalParameters::unfreeze_parameters(module, recursive),)*
                    }
                }

                fn parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    match self {
                        #(Self::#variants(module) => #root::module::PhysicalParameters::parameters(module),)*
                    }
                }

                fn parameters_mut(&mut self) -> #root::module::ModuleParamMut<'_> {
                    match self {
                        #(Self::#variants(module) => #root::module::PhysicalParameters::parameters_mut(module),)*
                    }
                }

                fn trainable_parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    match self {
                        #(Self::#variants(module) => #root::module::PhysicalParameters::trainable_parameters(module),)*
                    }
                }
            }
        };
    }
}

fn expand_physical_parameters_for_struct(
    ident: &Ident,
    generics: &Generics,
    data: &DataStruct,
    root: Option<syn::Path>,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let fields = filter_fields_with_attr(&data.fields, "param")?;

    Ok(impl_physical_parameters_for_struct(
        ident,
        generics,
        fields.filtered,
        root,
    ))
}

fn impl_physical_parameters_for_struct(
    ident: &Ident,
    generics: &Generics,
    fields: Vec<&syn::Field>,
    root: Option<syn::Path>,
) -> proc_macro2::TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let field_names: Vec<_> = fields.iter().map(|field| &field.ident).collect();

    let (extern_import, root) = match root {
        Some(root) => (quote::quote! {}, quote::quote! { #root }),
        None => (quote::quote! {}, quote::quote! { crate }),
    };

    quote::quote! {
        const _: () = {
            #extern_import
            impl #impl_generics #root::module::PhysicalParameters for #ident #ty_generics #where_clause {
                fn freeze_parameters(&mut self, recursive: bool) {
                    use #root::module::PhysicalParameter;
                    #(self.#field_names.freeze(recursive);)*
                }

                fn unfreeze_parameters(&mut self, recursive: bool) {
                    use #root::module::PhysicalParameter;
                    #(self.#field_names.unfreeze(recursive);)*
                }

                fn parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(parameters.insert(std::rc::Rc::from(stringify!(#field_names)), #root::module::PhysicalParameter::as_nested_value(&self.#field_names));)*
                    parameters
                }

                fn parameters_mut(&mut self) -> #root::module::ModuleParamMut<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(parameters.insert(std::rc::Rc::from(stringify!(#field_names)), #root::module::PhysicalParameter::as_nested_value_mut(&mut self.#field_names));)*
                    parameters
                }

                fn trainable_parameters(&self) -> #root::module::ModuleParamRef<'_> {
                    let mut parameters = #root::nested::NestedHashMap::new();
                    #(
                        if let Some(field) = #root::module::PhysicalParameter::as_trainable_nested_value(&self.#field_names) {
                            parameters.insert(std::rc::Rc::from(stringify!(#field_names)), field);
                        }
                    )*
                    parameters
                }
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::expand_physical_parameters;
    use syn::{parse_quote, DeriveInput};

    #[test]
    fn accepts_single_field_enum_variants() {
        let input: DeriveInput = parse_quote! {
            enum ModuleChoice<T> {
                First(T),
                Second(T),
            }
        };

        assert!(expand_physical_parameters(&input).is_ok());
    }

    #[test]
    fn rejects_enum_variants_that_cannot_delegate() {
        let input: DeriveInput = parse_quote! {
            enum ModuleChoice<T> {
                Missing,
                Multiple(T, T),
            }
        };

        let error = expand_physical_parameters(&input).unwrap_err();
        assert_eq!(
            error.to_string(),
            "PhysicalParameters enum variants must contain exactly one unnamed field"
        );
    }
}
