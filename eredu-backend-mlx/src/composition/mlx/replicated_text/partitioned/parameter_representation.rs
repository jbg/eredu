//! One source-bound descriptive table from actual idle resident parameter owners.
use super::*;
use eredu_nn::{
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType, WorkspaceMetadataError,
        WorkspaceParameterRepresentation, WorkspaceRepresentation,
    },
    ParameterId, ParameterMetadataView, ParameterSourceVisitor, Parameterized,
};
use std::mem::{size_of, size_of_val};

pub(in crate::composition::mlx::replicated_text) struct Collector<'context> {
    context: &'context WorkspaceContext,
    rows: Vec<WorkspaceParameterRepresentation>,
    failure: Option<eredu_nn::Error>,
}
impl Collector<'_> {
    /// Select only the exact static source when full resident coverage is
    /// unavailable. Partial unit rows are discarded without refund; actual
    /// unit facts still come from their checked layerwise source at acquire.
    pub(in crate::composition::mlx::replicated_text) fn static_source(
        &mut self,
        source: &(impl Parameterized<MlxTensor> + ?Sized),
    ) -> Result<bool, eredu_nn::Error> {
        self.context.charge_metadata(size_of::<(
            &mut Self, Result<bool, eredu_nn::Error>,
        )>().checked_add(size_of_val(&source)).ok_or(WorkspaceMetadataError::Overflow)?)?;
        // The enclosing installer must surface a real native/metadata failure;
        // a narrower source cannot turn such a failure into successful evidence.
        if self.failure.is_some() {
            return Ok(false);
        }
        self.rows.clear();
        self.source(source)
    }
    fn source(
        &mut self,
        source: &(impl Parameterized<MlxTensor> + ?Sized),
    ) -> Result<bool, eredu_nn::Error> {
        let visited = source.visit_parameter_sources(self);
        if let Some(cause) = self.failure.take() {
            return Err(cause);
        }
        match visited {
            Ok(()) => Ok(true),
            Err(
                eredu_nn::ParameterSourceError::Unavailable
                | eredu_nn::ParameterSourceError::UnclassifiedRetainedField,
            ) => Ok(false),
            Err(cause) => Err(self.context.metadata_source(cause)),
        }
    }
    fn parameter(
        &mut self,
        metadata: ParameterMetadataView<'_>,
        value: &MlxTensor,
    ) -> Result<(), eredu_nn::Error> {
        let controls = [
            size_of::<ParameterMetadataView<'_>>(),
            size_of::<ParameterId>(),
            size_of::<Result<ParameterId, eredu_nn::ParameterTopologyError>>(),
            size_of::<WorkspaceFloatingType>(),
            size_of::<Option<WorkspaceRepresentation>>(),
            size_of::<Option<bool>>(),
            size_of::<WorkspaceParameterRepresentation>(),
            size_of::<Result<(), eredu_nn::Error>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|bytes| bytes.checked_add(Array::descriptor_control_bytes()?))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.context.charge_metadata(bytes)?;
        let descriptor = value
            .as_array()
            .try_descriptor()
            .map_err(|cause| self.context.metadata_source(cause))?;
        let floating = match descriptor.facts().dtype() {
            Dtype::Float32 => WorkspaceFloatingType::Float32,
            Dtype::Float16 => WorkspaceFloatingType::Float16,
            Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
            // Quantized/integer slots retain their existing format facts.
            _ => return Ok(()),
        };
        self.context.reserve_metadata_vec(&mut self.rows, 1)?;
        let id = ParameterId::new(
            self.context
                .metadata_string(format_args!("{}", metadata.id()))?,
        )
        .map_err(|cause| self.context.metadata_source(cause))?;
        // Dtype belongs to the actual descriptor even when an unevaluated
        // view has no settled stride flags. False makes no contiguity claim.
        let representation = Some(WorkspaceRepresentation::new(
            floating,
            descriptor.row_contiguous().unwrap_or(false),
        ));
        let layout = self
            .context
            .layout(descriptor.shape(), WorkspaceDtype::Float32)?
            .with_representation(representation);
        self.rows
            .push(WorkspaceParameterRepresentation::new(id, layout));
        Ok(())
    }
}
impl<'source> ParameterSourceVisitor<'source, MlxTensor> for Collector<'_> {
    fn parameter(&mut self, metadata: ParameterMetadataView<'source>, value: &'source MlxTensor) {
        if self.failure.is_none() {
            self.failure = Collector::parameter(self, metadata, value).err();
        }
    }
    fn retained(&mut self, _: &'source MlxTensor) {}
}

/// One paid destination installed only after the exact strategy's complete
/// source visit. A callback failure retains the ordinary source/error custody;
/// partially populated rows never reach the workspace parameter table.
pub(in crate::composition::mlx::replicated_text) fn install<'context>(
    context: &'context WorkspaceContext,
    visit: impl FnOnce(&mut Collector<'context>) -> Result<bool, eredu_nn::Error>,
) -> Result<(), eredu_nn::Error> {
    let controls = [
        size_of::<Collector<'_>>(),
        size_of::<Result<bool, eredu_nn::Error>>(),
        size_of::<Result<(), eredu_nn::Error>>(),
        size_of::<(&WorkspaceContext,)>(),
        size_of_val(&visit),
    ];
    context.charge_metadata(
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let mut collector = Collector {
        context,
        rows: context.metadata_vec(0)?,
        failure: None,
    };
    let complete = visit(&mut collector)?;
    if let Some(cause) = collector.failure.take() {
        return Err(cause);
    }
    if complete {
        context.install_parameter_representations(collector.rows)?;
    }
    Ok(())
}

/// Uses the same retained execution's static source when its complete unit
/// source is unavailable. Both ordinary and composite session adapters call
/// this inside their immutable runtime fence. Partial rows never become facts.
pub(in crate::composition::mlx::replicated_text) fn install_with_static_fallback<'context>(
    context: &'context WorkspaceContext,
    static_modules: Option<&(impl Parameterized<MlxTensor> + ?Sized)>,
    visit: impl FnOnce(&mut Collector<'context>) -> Result<bool, eredu_nn::Error>,
) -> Result<(), eredu_nn::Error> {
    // install accounts the concrete callback, including its source/visit
    // captures; static_source accounts the fallback traversal before it starts.
    install(context, |collector| {
        if visit(collector)? {
            return Ok(true);
        }
        match static_modules {
            Some(source) => collector.static_source(source),
            None => Ok(false),
        }
    })
}

impl<U, P> MlxSelectedLayerwisePolicy<U, P>
where
    U: Parameterized<MlxTensor> + 'static,
{
    pub(in crate::composition::mlx::replicated_text) fn visit_parameter_sources_with_metadata<V>(
        &self,
        visitor: &mut V,
        context: &WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error>
    where
        V: for<'source> ParameterSourceVisitor<'source, MlxTensor>,
    {
        let controls = [
            size_of::<(&Self, &mut V, &WorkspaceContext)>(),
            size_of::<Result<bool, eredu_nn::Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<std::sync::MutexGuard<'_, MlxSelectedLayerwisePolicyInner<U, P>>>(),
            size_of::<
                Result<std::sync::MutexGuard<'_, MlxSelectedLayerwisePolicyInner<U, P>>, Error>,
            >(),
            size_of::<Result<(), eredu_nn::ParameterSourceError>>(),
            size_of::<(
                usize,
                eredu_runtime::ExecutionUnitAddress,
                eredu_runtime::ExecutionUnitAddress,
                &U,
            )>(),
            size_of::<
                std::iter::Enumerate<
                    std::slice::Iter<
                        '_,
                        (
                            eredu_runtime::ExecutionUnitAddress,
                            eredu_runtime::ExecutionUnitAddress,
                        ),
                    >,
                >,
            >(),
        ];
        context.charge_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        // Keep the actual policy transfer and every unit borrowed until the
        // callback returns. No residency acquire, native clone or completion.
        let selected = self
            .operation_policy()
            .map_err(|cause| context.metadata_source(cause))?;
        let MlxSelectedLayerwisePolicyInner::Resident(policy) = &*selected else {
            return Ok(false);
        };
        let observed = policy.parameter_sources();
        context.charge_metadata(size_of_val(&observed))?;
        let source = observed.map_err(|cause| context.metadata_source(cause))?;
        context.charge_metadata(size_of_val(&source))?;
        if source.layout().len() != self.parameter_locations.len() {
            return Err(context.metadata_error(format_args!(
                "resident parameter source differs from its retained partition locations"
            )));
        }
        for (ordinal, &(global, local)) in self.parameter_locations.iter().enumerate() {
            if source.layout().address(ordinal) != Some(local) || global.group() != local.group() {
                return Err(context.metadata_error(format_args!(
                    "resident parameter storage slot differs from its retained partition address"
                )));
            }
            // Modules were built from the retained global address. Their actual
            // ParameterIds already use that address; only storage uses local slots.
            let unit = source
                .unit(ordinal, local)
                .map_err(|cause| context.metadata_source(cause))?;
            match unit.visit_parameter_sources(visitor) {
                Ok(()) => {}
                Err(
                    eredu_nn::ParameterSourceError::Unavailable
                    | eredu_nn::ParameterSourceError::UnclassifiedRetainedField,
                ) => return Ok(false),
                Err(cause) => return Err(context.metadata_source(cause)),
            }
        }
        Ok(true)
    }

    pub(in crate::composition::mlx::replicated_text) fn install_workspace_parameter_representations(
        &self,
        static_modules: &(impl Parameterized<MlxTensor> + ?Sized),
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        install(context, |collector| {
            if !collector.source(static_modules)? {
                return Ok(false);
            }
            let frames=[size_of::<std::sync::MutexGuard<'_,MlxSelectedLayerwisePolicyInner<U,P>>>(),
                size_of::<Result<std::sync::MutexGuard<'_,MlxSelectedLayerwisePolicyInner<U,P>>,Error>>(),
                size_of::<bool>()];
            context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?)?;
            let bounded={
                let selected=self.operation_policy().map_err(|cause|context.metadata_source(cause))?;
                matches!(&*selected,MlxSelectedLayerwisePolicyInner::Bounded {..})
            };
            if bounded {
                // All static modules were visited above. Unit dtype facts are
                // supplied by the exact layerwise row source at checked acquire;
                // absent rows remain absent, with no scalar-type inference.
                Ok(true)
            } else {
                self.visit_parameter_sources_with_metadata(collector, context)
            }
        })
    }
}
