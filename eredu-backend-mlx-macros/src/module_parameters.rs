use syn::{DataStruct, DeriveInput, Generics, Ident};

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
        _ => Err(syn::Error::new_spanned(
            input,
            "PhysicalParameters can only be derived for structs",
        )),
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
