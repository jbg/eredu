//! Fallible table construction for the participating metadata execution worker.
use super::*;
use crate::working_memory::WorkspaceResidentLayerState;
use eredu_nn::{
    Error,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceMetadataError},
};

impl<L> DeviceState<WorkspaceBackend, L> {
    /// Metadata requests for this exact typed layer table. Nested layer state
    /// and the separately retained immutable layout require their own census.
    /// Includes a possible boxed-slice replacement when the allocator supplies
    /// spare vector capacity. This query grants no allocation authority.
    pub fn workspace_construction_bytes(layer_count: usize) -> Option<usize> {
        [
            WorkspaceContext::metadata_vec_bytes::<L>(layer_count)?,
            std::alloc::Layout::array::<L>(layer_count).ok()?.size(),
            crate::HostSlotMetadata::workspace_control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Realizes metadata layers while sharing the already-owned immutable layout.
    /// The callback owns its nested metadata accounting; this constructor covers
    /// the actual layer vector, boxed table and initialized ownership controls.
    pub fn create_workspace_with_shared_layout(
        layout: SharedStateLayout,
        context: &WorkspaceContext,
        create: impl FnMut(usize, &LayerCachePolicy) -> Result<L, Error>,
    ) -> Result<Self, Error> {
        Self::create_workspace_with_shared_layout_result(layout, context, create)
    }

    /// Runs the same metadata constructor while preserving the caller's typed
    /// policy failures. Only metadata allocation failures convert through E.
    pub fn create_workspace_with_shared_layout_result<E: From<Error>>(
        layout: SharedStateLayout,
        context: &WorkspaceContext,
        mut create: impl FnMut(usize, &LayerCachePolicy) -> Result<L, E>,
    ) -> Result<Self, E> {
        let mut layers = context
            .metadata_vec(layout.layout().len())
            .map_err(E::from)?;
        for (index, policy) in layout.layout().layers().iter().enumerate() {
            layers.push(create(index, policy)?);
        }
        Self::workspace_from_layers(Some(layout), Some(layers), context).map_err(E::from)
    }

    fn workspace_from_layers(
        layout: Option<SharedStateLayout>,
        layers: Option<Vec<L>>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let layers = layers
            .map(|layers| {
                // Exact-capacity construction normally needs no shrink. If the
                // allocator supplied excess capacity, charge a possible replacement
                // before converting to the final boxed extent.
                if layers.capacity() != layers.len() && std::mem::size_of::<L>() != 0 {
                    context.charge_metadata(
                        std::alloc::Layout::array::<L>(layers.len())
                            .map_err(|_| WorkspaceMetadataError::Overflow)?
                            .size(),
                    )?;
                }
                crate::HostSlotTable::new_workspace(layers.into_boxed_slice(), context)
            })
            .transpose()?;
        Ok(Self {
            layout,
            layers,
            inference_retention: crate::working_memory::InferenceRetention::new(),
            backend: PhantomData,
        })
    }
}

impl DeviceState<WorkspaceBackend, WorkspaceResidentLayerState> {
    /// Validates each actual projected layer's trace, including empty layers.
    /// No state is copied, rebound or authorized for native execution.
    pub fn validate_workspace_context(&self, context: &WorkspaceContext) -> Result<(), Error> {
        for layer in self.as_ref() {
            layer.validate_workspace_context(context)?;
        }
        Ok(())
    }

    /// Makes a trace-state checkpoint with an explicitly priced table. Layer
    /// clones retain immutable shape/role storage; later mutable slot access
    /// performs its own fallible copy. Native request admission and revision
    /// authority are not inherited by this metadata-only quote branch.
    pub fn try_clone_workspace(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        let layers = self
            .layers
            .as_ref()
            .map(|table| {
                let mut layers = context.metadata_vec(table.len())?;
                for layer in table.slots() {
                    layer.validate_workspace_context(context)?;
                    layers.push(layer.clone());
                }
                Ok::<_, Error>(layers)
            })
            .transpose()?;
        Self::workspace_from_layers(self.layout.clone(), layers, context)
    }
}
