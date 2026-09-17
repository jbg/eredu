use super::*;
use eredu_nn::{ParameterNameError, ParameterNameView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DestinationError<'a> {
    Construction(StaticConstructionError),
    Declaration(StaticDeclarationError<'a>),
    Name(ParameterNameError),
    Rows,
    Bytes,
    Overflow,
}
impl From<StaticConstructionError> for DestinationError<'_> {
    fn from(e: StaticConstructionError) -> Self {
        Self::Construction(e)
    }
}
impl<'a> From<StaticDeclarationError<'a>> for DestinationError<'a> {
    fn from(e: StaticDeclarationError<'a>) -> Self {
        Self::Declaration(e)
    }
}
impl From<ParameterNameError> for DestinationError<'_> {
    fn from(e: ParameterNameError) -> Self {
        Self::Name(e)
    }
}

// Closed test producer for only the declaration destinations. These exact
// borrowed rows and byte buffers are not a native module/storage/Q certificate.
struct Rows<'d, 'a> {
    rows: Option<&'d mut [Option<StaticParameterDeclaration<'a>>]>,
    bytes: Option<&'d mut [u8]>,
    used_rows: usize,
    used_bytes: usize,
}
impl<'d, 'a> Rows<'d, 'a> {
    fn count() -> Self {
        Self {
            rows: None,
            bytes: None,
            used_rows: 0,
            used_bytes: 0,
        }
    }
    fn push(&mut self, row: StaticParameterDeclaration<'a>) -> Result<(), DestinationError<'a>> {
        let names = [
            Some(row.name),
            row.alias_of,
            row.group,
            row.linear_companion_of,
        ];
        let mut bytes = 0usize;
        for name in names.into_iter().flatten() {
            bytes = bytes
                .checked_add(name.byte_len().ok_or(DestinationError::Overflow)?)
                .ok_or(DestinationError::Overflow)?;
        }
        let next_rows = self
            .used_rows
            .checked_add(1)
            .ok_or(DestinationError::Overflow)?;
        let next_bytes = self
            .used_bytes
            .checked_add(bytes)
            .ok_or(DestinationError::Overflow)?;
        if self.rows.as_ref().is_some_and(|r| next_rows > r.len()) {
            return Err(DestinationError::Rows);
        }
        if self.bytes.as_ref().is_some_and(|b| next_bytes > b.len()) {
            return Err(DestinationError::Bytes);
        }
        if let Some(destination) = self.bytes.as_deref_mut() {
            let mut position = self.used_bytes;
            for name in names.into_iter().flatten() {
                for part in name.parts() {
                    let end = position + part.len();
                    destination[position..end].copy_from_slice(part.as_bytes());
                    position = end;
                }
            }
            assert_eq!(position, next_bytes);
        }
        if let Some(destination) = self.rows.as_deref_mut() {
            destination[self.used_rows] = Some(row);
        }
        self.used_rows = next_rows;
        self.used_bytes = next_bytes;
        Ok(())
    }
    fn matrix(
        &mut self,
        weight: &'a str,
        format: LinearFormat,
    ) -> Result<(), DestinationError<'a>> {
        let primary = StaticParameterDeclaration::trainable(ParameterNameView::new(weight))?;
        let physical = static_linear_format(weight, format)?;
        self.push(primary)?;
        for row in physical.scale.into_iter().chain(physical.affine_bias) {
            self.push(row)?;
        }
        Ok(())
    }
}
impl<'a> StaticModuleConstruction<'a> for Rows<'_, 'a> {
    type Embedding = ();
    type Normalization = ();
    type Linear = ();
    type Error = DestinationError<'a>;
    fn embedding(&mut self, d: StaticEmbeddingDeclaration<'a>) -> Result<(), Self::Error> {
        self.matrix(d.weight, d.format)
    }
    fn normalization(&mut self, d: StaticNormalizationDeclaration<'a>) -> Result<(), Self::Error> {
        self.push(StaticParameterDeclaration::trainable(
            ParameterNameView::new(d.weight),
        )?)
    }
    fn head(&mut self, d: StaticHeadDeclaration<'a>) -> Result<(), Self::Error> {
        self.matrix(d.weight, d.format)
    }
}

#[test]
fn actual_static_declarations_fill_exact_rows_and_bytes_with_short_prefix_retention() {
    let mut spec = source(false);
    spec.embedding_quantization = Some(WeightQuantization::Affine(Default::default()));
    spec.head_format = LinearFormat::Affine(Default::default());
    let view = spec.borrowed();
    let mut count = Rows::count();
    let _ = view
        .construct_with(StaticModulePlacement::Replicated, &mut count)
        .unwrap();
    assert_eq!(count.used_rows, 7);
    let expected = [
        ("model.embed_tokens.weight", None),
        (
            "model.embed_tokens.scales",
            Some("model.embed_tokens.weight"),
        ),
        (
            "model.embed_tokens.biases",
            Some("model.embed_tokens.weight"),
        ),
        ("model.norm.weight", None),
        ("lm_head.weight", None),
        ("lm_head.scales", Some("lm_head.weight")),
        ("lm_head.biases", Some("lm_head.weight")),
    ];
    let exact = expected
        .iter()
        .map(|(name, group)| name.len() + group.map_or(0, str::len))
        .sum::<usize>();
    assert_eq!(count.used_bytes, exact);
    assert!(
        std::mem::size_of::<StaticParameterDeclaration<'_>>()
            > std::mem::size_of::<ParameterNameView<'_>>()
    );
    let mut row_destination = [None; 7];
    let mut byte_destination = vec![0xa5; exact];
    let mut fill = Rows {
        rows: Some(&mut row_destination),
        bytes: Some(&mut byte_destination),
        used_rows: 0,
        used_bytes: 0,
    };
    let _ = view
        .construct_with(StaticModulePlacement::Replicated, &mut fill)
        .unwrap();
    assert_eq!((fill.used_rows, fill.used_bytes), (7, exact));
    drop(fill);
    let mut flat = Vec::new();
    for (index, ((name, group), row)) in expected.into_iter().zip(row_destination).enumerate() {
        let row = row.unwrap();
        assert_eq!(row.name.to_string(), name);
        assert_eq!(row.group.map(|g| g.to_string()), group.map(str::to_owned));
        assert!(row.trainable);
        assert!(row.alias_of.is_none());
        assert!(row.linear_companion_of.is_none());
        assert_eq!(row.linear_row_layout, eredu_nn::LinearRowLayout::Contiguous);
        let role = match index {
            1 | 5 => Some(eredu_nn::LinearCompanionRole::Scale),
            2 | 6 => Some(eredu_nn::LinearCompanionRole::AffineBias),
            _ => None,
        };
        assert_eq!(row.linear_companion, role);
        flat.extend_from_slice(name.as_bytes());
        if let Some(group) = group {
            flat.extend_from_slice(group.as_bytes());
        }
    }
    assert_eq!(byte_destination, flat);
    assert_eq!(
        row_destination[0].unwrap().name.parts()[0].as_ptr(),
        spec.embedding_weight.as_ptr()
    );
    assert_eq!(
        row_destination[1].unwrap().name.parts()[0].as_ptr(),
        spec.embedding_weight.as_ptr()
    );
    for short_rows in [true, false] {
        let mut rows = vec![None; if short_rows { 6 } else { 7 }];
        let mut bytes = vec![0xa5; if short_rows { exact } else { exact - 1 }];
        let mut fill = Rows {
            rows: Some(&mut rows),
            bytes: Some(&mut bytes),
            used_rows: 0,
            used_bytes: 0,
        };
        let error = view
            .construct_with(StaticModulePlacement::Replicated, &mut fill)
            .unwrap_err();
        assert_eq!(
            error,
            if short_rows {
                DestinationError::Rows
            } else {
                DestinationError::Bytes
            }
        );
        assert_eq!(fill.used_rows, 6);
        let prefix = fill.used_bytes;
        drop(fill);
        assert!(rows[..6].iter().all(Option::is_some));
        assert!(rows.get(6).is_none_or(Option::is_none));
        assert_eq!(&bytes[..prefix], &flat[..prefix]);
        assert!(bytes[prefix..].iter().all(|b| *b == 0xa5));
    }
}
