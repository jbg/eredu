pub(crate) struct FilteredFields<'a> {
    pub filtered: Vec<&'a syn::Field>,
    pub other_fields: Vec<&'a syn::Field>,
}

pub(crate) fn filter_fields_with_attr<'a>(
    fields: &'a syn::Fields,
    attr_name: &str,
) -> Result<FilteredFields<'a>, syn::Error> {
    let mut filtered = Vec::new();
    let mut other_fields = Vec::new();

    match fields {
        syn::Fields::Named(fields) => {
            for field in &fields.named {
                if field
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident(attr_name))
                {
                    filtered.push(field);
                } else {
                    other_fields.push(field);
                }
            }
        }
        syn::Fields::Unit => {}
        syn::Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                fields,
                "Struct with unnamed fields is not supported".to_string(),
            ));
        }
    }

    Ok(FilteredFields {
        filtered,
        other_fields,
    })
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
