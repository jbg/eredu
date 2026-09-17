use super::*;

/// Failure to encode a canonical input identity or deliver its next word.
#[derive(Debug, thiserror::Error)]
pub enum InputWordWriteError<E> {
    /// An existing descriptor cannot be represented by the canonical wire format.
    #[error("{0}")]
    Encoding(#[from] PreparedInputError),
    /// The word consumer rejected delivery; no later word is visited.
    #[error("input identity word consumer failed: {0}")]
    Sink(#[source] E),
}

fn emit<E>(
    sink: &mut impl FnMut(u32) -> Result<(), E>,
    word: u32,
) -> Result<(), InputWordWriteError<E>> {
    sink(word).map_err(InputWordWriteError::Sink)
}

impl InputTensorIdentity {
    fn visit_words<E>(
        &self,
        sink: &mut impl FnMut(u32) -> Result<(), E>,
    ) -> Result<(), InputWordWriteError<E>> {
        emit(sink, dtype_wire_tag(&self.dtype)?)?;
        emit(
            sink,
            u32::try_from(self.shape.len())
                .map_err(|_| PreparedInputError::WireValueOverflow("tensor rank"))?,
        )?;
        for dimension in &self.shape {
            emit(
                sink,
                u32::try_from(*dimension)
                    .map_err(|_| PreparedInputError::WireValueOverflow("tensor dimension"))?,
            )?;
        }
        Ok(())
    }
}

impl InputExtent {
    fn visit_words<E>(
        self,
        sink: &mut impl FnMut(u32) -> Result<(), E>,
    ) -> Result<(), InputWordWriteError<E>> {
        emit(sink, self.wire_tag())?;
        let mut dimension = |value| {
            emit(
                sink,
                u32::try_from(value)
                    .map_err(|_| PreparedInputError::WireValueOverflow("input extent"))?,
            )
        };
        match self {
            Self::PatchGrid {
                time,
                height,
                width,
            } => {
                dimension(time)?;
                dimension(height)?;
                dimension(width)?;
            }
            Self::AudioValidFrames(frames) => dimension(frames)?,
        }
        Ok(())
    }
}

impl PreparedInputIdentity {
    /// Streams the existing canonical wire words without creating an encoded
    /// vector. A sink error stops immediately and preserves that exact cause.
    /// Descriptor errors can follow an already delivered prefix, so a consumer
    /// must discard provisional state on failure. No native data is inspected.
    pub fn visit_encoded_words<E>(
        &self,
        mut sink: impl FnMut(u32) -> Result<(), E>,
    ) -> Result<(), InputWordWriteError<E>> {
        emit(
            &mut sink,
            u32::try_from(self.parts.len())
                .map_err(|_| PreparedInputError::WireValueOverflow("part count"))?,
        )?;
        for part in &self.parts {
            emit(&mut sink, part.modality.wire_tag())?;
            emit(&mut sink, part.payload_kind.wire_tag())?;
            part.payload.visit_words(&mut sink)?;
            emit(
                &mut sink,
                u32::try_from(part.metadata.len())
                    .map_err(|_| PreparedInputError::WireValueOverflow("metadata count"))?,
            )?;
            for (key, identity) in &part.metadata {
                emit(&mut sink, key.wire_tag())?;
                identity.visit_words(&mut sink)?;
            }
            emit(
                &mut sink,
                u32::try_from(part.extents.len())
                    .map_err(|_| PreparedInputError::WireValueOverflow("extent count"))?,
            )?;
            for extent in part.extents.values().copied() {
                extent.visit_words(&mut sink)?;
            }
        }
        Ok(())
    }

    /// Counts canonical wire words with checked arithmetic, without allocating
    /// a descriptor vector or cloning its payload.
    pub fn encoded_word_count(&self) -> Result<usize, PreparedInputError> {
        let mut count = 0usize;
        self.visit_encoded_words(|_| {
            count = count
                .checked_add(1)
                .ok_or(PreparedInputError::WireValueOverflow("encoded word count"))?;
            Ok::<_, PreparedInputError>(())
        })
        .map_err(|error| match error {
            InputWordWriteError::Encoding(error) | InputWordWriteError::Sink(error) => error,
        })?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
