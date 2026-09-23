use super::*;

impl MechanismInvocation {
    /// Validates ordinary invocation geometry and returns logical values.
    ///
    /// Input and parameter storage must not be charged again as workspace. In
    /// particular, packed weights are deliberately absent: their physical bytes
    /// and sharing are described by the ordinary prepared parameter contracts.
    pub fn logical_values(&self) -> Result<Vec<LogicalValue>, Error> {
        use LogicalValueKind::{Input, Output, State};
        let mut values = Vec::new();
        let mut add = |name: &str, shape: &[u64], element, kind| -> Result<(), Error> {
            if shape.contains(&0) {
                return Err(Error::backend(
                    "mechanism invocation requires positive value dimensions",
                ));
            }
            let value = LogicalValue {
                name: name.into(),
                shape: shape.to_vec(),
                element,
                kind,
            };
            value.logical_bytes()?;
            values.push(value);
            Ok(())
        };
        match *self {
            Self::Projection {
                rows,
                input,
                output,
                format,
                element,
                ..
            } => {
                format.validate().map_err(Error::backend)?;
                add("input", &[rows, input], element, Input)?;
                add("output", &[rows, output], element, Output)?;
            }
            Self::Attention {
                batch,
                query_heads,
                kv_heads,
                queries,
                keys,
                key_width,
                value_width,
                element,
                ..
            } => {
                if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
                    return Err(Error::backend(
                        "query heads must be a positive multiple of key/value heads",
                    ));
                }
                add(
                    "queries",
                    &[batch, query_heads, queries, key_width],
                    element,
                    Input,
                )?;
                add("keys", &[batch, kv_heads, keys, key_width], element, Input)?;
                add(
                    "values",
                    &[batch, kv_heads, keys, value_width],
                    element,
                    Input,
                )?;
                add(
                    "output",
                    &[batch, query_heads, queries, value_width],
                    element,
                    Output,
                )?;
            }
            Self::Convolution {
                batch,
                tokens,
                channels,
                kernel,
                element,
            } => {
                if kernel == 0 {
                    return Err(Error::backend("convolution kernel must be positive"));
                }
                add("input", &[batch, tokens, channels], element, Input)?;
                add("output", &[batch, tokens, channels], element, Output)?;
                if kernel > 1 {
                    add("history", &[batch, kernel - 1, channels], element, State)?;
                }
            }
            Self::Recurrent {
                kind,
                batch,
                tokens,
                heads,
                value_width,
                state_width,
                element,
                chunk_size,
            } => {
                if chunk_size == Some(0) {
                    return Err(Error::backend("scan chunk size must be positive"));
                }
                add(
                    "input",
                    &[batch, tokens, heads, value_width],
                    element,
                    Input,
                )?;
                add(
                    "output",
                    &[batch, tokens, heads, value_width],
                    element,
                    Output,
                )?;
                let state = match kind {
                    RecurrentKind::GatedDelta => [batch, heads, state_width, value_width],
                    RecurrentKind::SelectiveStateSpace => [batch, heads, value_width, state_width],
                };
                add("state", &state, TensorElementType::F32, State)?;
            }
            Self::ExpertDispatch {
                rows,
                experts,
                selected,
                input,
                output,
                format,
                element,
            } => {
                if selected > experts {
                    return Err(Error::backend("selected routes exceed local expert count"));
                }
                format.validate().map_err(Error::backend)?;
                add("input", &[rows, input], element, Input)?;
                add("indices", &[rows, selected], TensorElementType::U32, Input)?;
                add("coefficients", &[rows, selected], element, Input)?;
                add("output", &[rows, output], element, Output)?;
            }
            Self::CacheUpdate {
                batch,
                heads,
                previous,
                appended,
                key_width,
                value_width,
                element,
            } => {
                let visible = previous
                    .checked_add(appended)
                    .ok_or_else(|| Error::backend("cache append endpoint overflowed"))?;
                add(
                    "new_keys",
                    &[batch, heads, appended, key_width],
                    element,
                    Input,
                )?;
                add(
                    "new_values",
                    &[batch, heads, appended, value_width],
                    element,
                    Input,
                )?;
                add(
                    "visible_keys",
                    &[batch, heads, visible, key_width],
                    element,
                    Output,
                )?;
                add(
                    "visible_values",
                    &[batch, heads, visible, value_width],
                    element,
                    Output,
                )?;
            }
            Self::Sampling {
                rows,
                vocabulary,
                element,
                mode,
                top_k,
                ..
            } => {
                if top_k > vocabulary {
                    return Err(Error::backend("sampling top-k exceeds vocabulary"));
                }
                add("logits", &[rows, vocabulary], element, Input)?;
                add("tokens", &[rows], TensorElementType::U32, Output)?;
                if mode == SamplingMode::MirostatV2 {
                    if rows != 1 {
                        return Err(Error::backend("Mirostat V2 requires one distribution"));
                    }
                    add("adaptive_state", &[1], TensorElementType::F32, State)?;
                }
            }
        }
        Ok(values)
    }
}
