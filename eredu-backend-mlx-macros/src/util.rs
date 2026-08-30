pub(crate) struct FilteredFields<'a> {
    pub filtered: Vec<&'a syn::Field>,
}

pub(crate) fn filter_fields_with_attr<'a>(
    fields: &'a syn::Fields,
    attr_name: &str,
) -> Result<FilteredFields<'a>, syn::Error> {
    let filtered = match fields {
        syn::Fields::Named(fields) => fields
            .named
            .iter()
            .filter(|field| {
                field
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident(attr_name))
            })
            .collect(),
        syn::Fields::Unit => Vec::new(),
        syn::Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                fields,
                "structs with unnamed fields are unsupported",
            ));
        }
    };
    Ok(FilteredFields { filtered })
}

pub(crate) fn parse_root_attribute(
    attrs: &[syn::Attribute],
    attr_name: &str,
) -> Result<Option<syn::Path>, syn::Error> {
    let mut root = None;
    for attr in attrs.iter().filter(|attr| attr.path().is_ident(attr_name)) {
        if matches!(attr.meta, syn::Meta::Path(_)) {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("root") {
                return Err(meta.error(format!("unsupported `{attr_name}` option")));
            }
            if root.is_some() {
                return Err(meta.error("duplicate `root` option"));
            }
            root = Some(meta.value()?.parse()?);
            Ok(())
        })?;
    }
    Ok(root)
}
