//! Same borrowed layout clone, admitted before its first allocation.
use super::*;
use crate::working_memory::{OriginalHostMetadataCustody, qualified_shared_bytes};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::alloc::Layout;

// This private recipe is tied to one immutable source. It is neither a public
// byte certificate nor authority to construct a different layout or native state.
struct ClonePlan<'a> {
    source: &'a StateLayout,
    bytes: usize,
}

impl<'a> ClonePlan<'a> {
    fn new(source: &'a StateLayout) -> Result<Self, WorkspaceMetadataError> {
        let payload = clone_payload_bytes(source)?;
        let shared = |bytes: Result<u64, crate::working_memory::WorkingMemoryError>| {
            let bytes = bytes.map_err(|_| WorkspaceMetadataError::Unqualified)?;
            usize::try_from(bytes).map_err(|_| WorkspaceMetadataError::Overflow)
        };
        // The existing qualified compiler/liballoc clone worker requests each
        // Vec/String/boxed-slice extent at len, not at the source's spare capacity.
        // Its closed built-in elements have no user Clone callbacks. Clone remains
        // infallible after admission, including its existing allocator-OOM policy.
        let parts = [
            payload,
            shared(qualified_shared_bytes::<LayoutInner>())?,
            shared(qualified_shared_bytes::<()>())?,
            MetadataCustody::text_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
            shared(OriginalHostMetadataCustody::initialized_mutex_bytes())?,
            size_of::<StateLayout>(),
            size_of::<ClonePlan<'a>>(),
            size_of::<Result<ClonePlan<'a>, WorkspaceMetadataError>>(),
            size_of::<SharedStateLayout>(),
            size_of::<Arc<LayoutInner>>(),
            size_of::<Result<SharedStateLayout, Error>>(),
            size_of::<WorkspaceMetadataError>(),
            size_of::<Error>(),
            size_of::<(&StateLayout, &WorkspaceContext)>(),
            size_of::<Result<(), WorkspaceMetadataError>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(Self { source, bytes })
    }

    fn construct(self, context: &WorkspaceContext) -> Result<SharedStateLayout, Error> {
        // Rejection precedes the entire nested clone, identity, shared shell and
        // private PAL initialization. No partial source is relabelled as admitted.
        context.charge_metadata(self.bytes)?;
        let layout = self.source.clone();
        let custody = MetadataCustody::new_text();
        Ok(SharedStateLayout::from_payload(layout, custody))
    }
}

impl SharedStateLayout {
    /// Copies the existing validated layout under the context metadata producer.
    /// All cloned declarations and names keep their exact ordinary semantics.
    pub(crate) fn copy_workspace(
        source: &StateLayout,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if !context.uses_checked_metadata() {
            return Ok(Self::new(source.clone()));
        }
        ClonePlan::new(source)?.construct(context)
    }
}

fn array_bytes<T>(count: usize) -> Result<usize, WorkspaceMetadataError> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| WorkspaceMetadataError::Overflow)
}
fn add(total: &mut usize, bytes: usize) -> Result<(), WorkspaceMetadataError> {
    *total = total
        .checked_add(bytes)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    Ok(())
}

fn clone_payload_bytes(source: &StateLayout) -> Result<usize, WorkspaceMetadataError> {
    let StateLayout {
        layers,
        components,
        segments,
    } = source;
    // The stored StateLayout value itself is included in LayoutInner's Arc.
    let mut total = clone_layer_payload_bytes(layers)?;
    add(
        &mut total,
        array_bytes::<Vec<StateComponentPolicy>>(components.len())?,
    )?;
    add(&mut total, array_bytes::<StateSegmentSpec>(segments.len())?)?;
    for group in components {
        add(
            &mut total,
            array_bytes::<StateComponentPolicy>(group.len())?,
        )?;
        for component in group {
            add(
                &mut total,
                array_bytes::<StateTensorDimension>(component.shape().len())?,
            )?;
        }
    }
    for segment in segments {
        let StateSegmentSpec {
            id: StateSegmentId(name),
            layers: _,
            lifetime: _,
            processed_token_offset: _,
        } = segment;
        add(&mut total, array_bytes::<u8>(name.len())?)?;
    }
    Ok(total)
}

fn clone_layer_payload_bytes(
    layers: &eredu_core::LayerSchedule<LayerCachePolicy>,
) -> Result<usize, WorkspaceMetadataError> {
    let mut total = array_bytes::<LayerCachePolicy>(layers.len())?;
    for layer in layers.iter() {
        add(&mut total, StateLayout::policy_clone_payload_bytes(layer)?)?;
    }
    Ok(total)
}

impl StateLayout {
    // Same exhaustive closed policy clone query used by full-layout cloning
    // and actual sliced policy destinations. The outer policy slot is separate.
    pub(in crate::state) fn policy_clone_payload_bytes(
        layer: &LayerCachePolicy,
    ) -> Result<usize, WorkspaceMetadataError> {
        let mut total = 0usize;
        let tensors = match layer {
            LayerCachePolicy::NoState
            | LayerCachePolicy::KeyValue {
                attention: _,
                num_key_value_heads: _,
                head_dim: _,
            }
            | LayerCachePolicy::KeyOnly {
                attention: _,
                num_key_heads: _,
                head_dim: _,
            }
            | LayerCachePolicy::CompressedLatentRotary {
                attention: _,
                latent_dim: _,
                rotary_dim: _,
            } => return Ok(0),
            LayerCachePolicy::FixedState { tensors }
            | LayerCachePolicy::KeyValueWithFixedState {
                attention: _,
                num_key_value_heads: _,
                head_dim: _,
                tensors,
            }
            | LayerCachePolicy::KeyOnlyWithFixedState {
                attention: _,
                num_key_heads: _,
                head_dim: _,
                tensors,
            } => tensors,
        };
        add(&mut total, array_bytes::<StateTensorPolicy>(tensors.len())?)?;
        for tensor in tensors {
            let StateTensorPolicy {
                role: _,
                shape,
                dtype: _,
                residency: _,
                presence: _,
            } = tensor;
            add(
                &mut total,
                array_bytes::<StateTensorDimension>(shape.len())?,
            )?;
        }
        Ok(total)
    }
}

fn owned_clone_bytes<T>(payload: usize) -> Result<usize, WorkspaceMetadataError> {
    let parts = [
        payload,
        size_of::<T>(),
        size_of::<Result<T, Error>>(),
        size_of::<(&T, &WorkspaceContext)>(),
        size_of::<Result<(), WorkspaceMetadataError>>(),
        size_of::<WorkspaceMetadataError>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
        .ok_or(WorkspaceMetadataError::Overflow)
}

impl StateLayout {
    /// Same closed nested Clone, charged before the first allocation. No shared
    /// layout identity, custody shell or duplicate validation is constructed.
    pub fn clone_workspace(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        if context.uses_checked_metadata() {
            context.charge_metadata(owned_clone_bytes::<Self>(clone_payload_bytes(self)?)?)?;
        }
        Ok(self.clone())
    }

    pub(crate) fn schedule_clone_metadata_bytes(
        source: &eredu_core::LayerSchedule<LayerCachePolicy>,
    ) -> Result<usize, WorkspaceMetadataError> {
        owned_clone_bytes::<eredu_core::LayerSchedule<LayerCachePolicy>>(clone_layer_payload_bytes(
            source,
        )?)
    }

    /// Copies only the actual policy schedule used by persistence identity.
    pub(crate) fn clone_layers_workspace(
        &self,
        context: &WorkspaceContext,
    ) -> Result<eredu_core::LayerSchedule<LayerCachePolicy>, Error> {
        if context.uses_checked_metadata() {
            context.charge_metadata(owned_clone_bytes::<
                eredu_core::LayerSchedule<LayerCachePolicy>,
            >(clone_layer_payload_bytes(&self.layers)?)?)?;
        }
        Ok(self.layers.clone())
    }
}

impl StateLayout {
    /// Copies one actual retained cache policy through the existing exhaustive
    /// layout-policy census. The caller retains the metadata destination.
    pub fn clone_policy_workspace(
        policy: &LayerCachePolicy,
        context: &WorkspaceContext,
    ) -> Result<LayerCachePolicy, Error> {
        if context.uses_checked_metadata() {
            context.charge_metadata(owned_clone_bytes::<LayerCachePolicy>(
                Self::policy_clone_payload_bytes(policy)?,
            )?)?;
        }
        Ok(policy.clone())
    }
}
