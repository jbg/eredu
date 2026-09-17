//! Owning policy and mutable stream metadata for the shared pooling worker.
use super::*;

pub(super) fn policy(
    source: &LayerCachePolicy,
    context: &WorkspaceContext,
) -> Result<Rc<LayerCachePolicy>, Error> {
    let value = match source {
        LayerCachePolicy::KeyOnly {
            attention,
            num_key_heads,
            head_dim,
        } => LayerCachePolicy::KeyOnly {
            attention: *attention,
            num_key_heads: *num_key_heads,
            head_dim: *head_dim,
        },
        LayerCachePolicy::KeyOnlyWithFixedState {
            attention,
            num_key_heads,
            head_dim,
            tensors,
        } => {
            let mut copied = context.metadata_vec(tensors.len())?;
            for tensor in tensors {
                let mut shape = context.metadata_vec(tensor.shape.len())?;
                shape.extend(tensor.shape.iter().copied());
                copied.push(StateTensorPolicy {
                    role: tensor.role,
                    shape,
                    dtype: tensor.dtype,
                    residency: tensor.residency,
                    presence: tensor.presence,
                });
            }
            LayerCachePolicy::KeyOnlyWithFixedState {
                attention: *attention,
                num_key_heads: *num_key_heads,
                head_dim: *head_dim,
                tensors: copied,
            }
        }
        _ => unreachable!("shared geometry validates the key-only mechanism"),
    };
    context.metadata_rc(value).map_err(Into::into)
}

impl WorkspacePoolingLayerState {
    pub(super) fn streams_mut(&mut self) -> Result<&mut Vec<WorkspacePoolingStream>, Error> {
        if Rc::get_mut(&mut self.streams).is_none() {
            // Each stream clone only retains immutable declarations and tensor
            // identities. Both new owning extents are admitted before replacing
            // the checkpoint-shared stream array.
            let mut streams = self.context.metadata_vec(self.streams.len())?;
            streams.extend(self.streams.iter().cloned());
            let streams = self.context.metadata_rc(streams)?;
            self.streams = streams;
        }
        Ok(Rc::get_mut(&mut self.streams).expect("unique prepared stream rows"))
    }
}
