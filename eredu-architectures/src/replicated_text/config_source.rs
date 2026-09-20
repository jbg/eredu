//! Configuration loans from the same immutable prepared source used by dispatch.
use super::*;
use std::mem::{size_of, size_of_val};
use crate::prepared_sources::PreparedModelSources;
use eredu_nn::workspace::WorkspaceMetadataError;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::ArtifactArchitecturePlan {}
    impl Sealed for super::PreparedModelSources {}
    impl Sealed for eredu_core::ArtifactInspection<super::ArtifactArchitecturePlan> {}
    impl<T: Sealed + ?Sized> Sealed for &T {}
}

/// Architecture-owned source for the existing typed replicated constructor.
/// A prepared source lends its retained immutable configuration. A standalone
/// architecture plan preserves the existing owned-configuration constructor.
/// This loan grants no source readiness, allocation or execution authority.
pub trait ReplicatedTextConstructionSource: sealed::Sealed {
    /// Borrows the exact target architecture used for validation and dispatch.
    fn architecture_plan(&self) -> &ArtifactArchitecturePlan;
    /// Retains the admitted owner when this loan came from prepared sources.
    #[doc(hidden)]
    fn prepared_sources(&self) -> Option<&PreparedModelSources>;
}
impl ReplicatedTextConstructionSource for ArtifactArchitecturePlan {
    fn architecture_plan(&self) -> &ArtifactArchitecturePlan {
        self
    }
    fn prepared_sources(&self) -> Option<&PreparedModelSources> {
        None
    }
}
impl ReplicatedTextConstructionSource for eredu_core::ArtifactInspection<ArtifactArchitecturePlan> {
    fn architecture_plan(&self)->&ArtifactArchitecturePlan {self.architecture_plan()}
    fn prepared_sources(&self)->Option<&PreparedModelSources>{None}
}
impl ReplicatedTextConstructionSource for PreparedModelSources {
    fn architecture_plan(&self) -> &ArtifactArchitecturePlan {
        self.architecture()
    }
    fn prepared_sources(&self) -> Option<&PreparedModelSources> {
        Some(self)
    }
}
impl<T: ReplicatedTextConstructionSource + ?Sized> ReplicatedTextConstructionSource for &T {
    fn architecture_plan(&self) -> &ArtifactArchitecturePlan {
        T::architecture_plan(self)
    }
    fn prepared_sources(&self) -> Option<&PreparedModelSources> {
        T::prepared_sources(self)
    }
}

pub(crate) enum ConfigOwner<C> {
    Owned(C),
    Shared(SharedConfig<C>),
    Prepared(PreparedConfig<C>),
}
pub(crate) struct PreparedConfig<C> {
    project: fn(&ArtifactArchitecturePlan) -> Option<&C>,
    source: PreparedModelSources,
}
impl<C> std::ops::Deref for ConfigOwner<C> {
    type Target = C;
    fn deref(&self) -> &C {
        match self {
            Self::Owned(value) => value,
            Self::Shared(value) => value.value(),
            Self::Prepared(value) => (value.project)(value.source.architecture())
                .expect("the retained configuration projection is immutable"),
        }
    }
}
impl<C> From<C> for ConfigOwner<C> {
    fn from(value: C) -> Self {
        Self::Owned(value)
    }
}

pub(crate) fn source_config<B: NeuralBackend, C: Clone>(
    source: &impl ReplicatedTextConstructionSource,
    config: &C,
    project: fn(&ArtifactArchitecturePlan) -> Option<&C>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigOwner<C>, eredu_nn::Error> {
    let Some(retained) = source.prepared_sources() else {
        return Ok(ConfigOwner::Owned(config.clone()));
    };
    if !project(retained.architecture()).is_some_and(|actual| std::ptr::eq(actual, config)) {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    if let Some(metadata) = B::construction_metadata(context) {
        let bytes = std::mem::size_of::<ConfigOwner<C>>()
            .checked_add(std::mem::size_of::<PreparedConfig<C>>())
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<Result<ConfigOwner<C>, eredu_nn::Error>>())
            })
            .ok_or(WorkspaceMetadataError::Overflow)?;
        metadata.charge_metadata(bytes)?;
    }
    Ok(ConfigOwner::Prepared(PreparedConfig {
        project,
        source: retained.clone(),
    }))
}

pub(crate) fn selected_config<B: NeuralBackend, C: EffectiveConfigType, E>(
    source: &impl ReplicatedTextConstructionSource,
    config: &C,
    selected: &SelectedReplicatedTextRealization,
    project: fn(&ArtifactArchitecturePlan) -> Option<&C>,
    legacy: fn(&C, &SelectedReplicatedTextRealization) -> Result<C, String>,
    validate_borrowed: fn(&C) -> Result<(), String>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigOwner<C>, ReplicatedTextDispatchError<E>> {
    if let Some(owner) = retained_effective(source, selected, B::construction_metadata(context))
        .map_err(ReplicatedTextDispatchError::Metadata)?
    {
        return Ok(ConfigOwner::Shared(owner));
    }
    if source.prepared_sources().is_some()
        && selected_quantized_format_rows(selected).next().is_none()
    {
        validate_borrowed(config).map_err(ReplicatedTextDispatchError::Architecture)?;
        source_config::<B, C>(source, config, project, context)
            .map_err(ReplicatedTextDispatchError::Metadata)
    } else {
        construct_effective::<B, C, E>(source, config, selected, legacy, context)
    }
}

pub(crate) fn unchanged<C>(_: &C) -> Result<(), String> {
    Ok(())
}

macro_rules! projection {
    ($name:ident, $variant:ident, $ty:ty $(, $field:ident)?) => {
        pub(crate) fn $name(plan: &ArtifactArchitecturePlan) -> Option<&$ty> {
            // Selection owns execution eligibility; this loan only projects the
            // immutable configuration whose identity source_config verifies.
            match (
                plan.safetensors_architecture().map(|plan| plan.model()),
                plan.gguf_plan().map(|plan| plan.model()),
            ) {
                (Some(SafetensorsModelConfig::$variant(config)), None) => {
                    Some(&config$(.$field)?)
                }
                (None, Some(GgufModelConfig::$variant(config))) => {
                    Some(&config$(.$field)?)
                }
                _ => None,
            }
        }
    };
}
projection!(gemma2, Gemma2, crate::gemma2::ModelArgs);
projection!(llama, Llama, crate::llama::ModelArgs);
projection!(nanbeige, Nanbeige, crate::nanbeige::ModelArgs);
projection!(k2_horizon, K2Horizon, crate::k2_horizon::ModelArgs);
projection!(qwen, Qwen, crate::qwen::ModelArgs);
projection!(lfm2, Lfm2, crate::lfm2::ModelArgs);
projection!(nemotron_h, NemotronH, crate::nemotron_h::ModelArgs);
projection!(qwen_hybrid, QwenHybrid, crate::qwen::hybrid::HybridConfig, text);
projection!(kimi_linear, KimiLinear, crate::kimi_linear::ModelArgs);
projection!(deepseek_v3, DeepSeekV3, crate::deepseek::V3Args);

// Kimi's ordinary selected helper leaves the raw config unchanged with no
// quantized rows. DeepSeek's helper also rewrites matrix-format metadata, so it
// still owns the existing selected constructor until that overlay can be lent.
pub(super) fn selected_kimi_linear<B: NeuralBackend, E>(
    source: &impl ReplicatedTextConstructionSource,
    config: &crate::kimi_linear::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigOwner<crate::kimi_linear::ModelArgs>, ReplicatedTextDispatchError<E>> {
    selected_config::<B, _, E>(
        source,
        config,
        selected,
        kimi_linear,
        selected_kimi_linear_args,
        unchanged,
        context,
    )
}
pub(crate) fn selected_deepseek_v3<B: NeuralBackend, E>(
    source: &impl ReplicatedTextConstructionSource,
    config: &crate::deepseek::V3Args,
    selected: &SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigOwner<crate::deepseek::V3Args>, ReplicatedTextDispatchError<E>> {
    if let Some(owner) = retained_effective(source, selected, B::construction_metadata(context))
        .map_err(ReplicatedTextDispatchError::Metadata)?
    {
        return Ok(ConfigOwner::Shared(owner));
    }
    construct_effective::<B, _, E>(source, config, selected, selected_deepseek_v3_args, context)
}

pub(crate) fn deepseek_v4(plan:&ArtifactArchitecturePlan)->Option<&crate::deepseek::V4Args>{
    match (plan.safetensors_architecture(),plan.gguf_plan()) {
        (Some(plan),None)=>match plan.model(){crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)=>Some(args),_=>None},
        (None,Some(plan))=>match plan.model(){crate::configuration::GgufModelConfig::DeepSeekV4(args)=>Some(args),_=>None},
        _=>None,
    }
}
pub(crate) fn selected_deepseek_v4<B:NeuralBackend,E>(source:&impl ReplicatedTextConstructionSource,
    config:&crate::deepseek::V4Args,selected:&SelectedReplicatedTextRealization,
    context:&<B::Tensor as Tensor>::Context)->Result<ConfigOwner<crate::deepseek::V4Args>,ReplicatedTextDispatchError<E>>{
    if let Some(owner)=retained_effective(source,selected,B::construction_metadata(context))
        .map_err(ReplicatedTextDispatchError::Metadata)?{return Ok(ConfigOwner::Shared(owner));}
    construct_effective::<B,_,E>(source,config,selected,selected_deepseek_v4_args,context)
}

// Preserve the standalone constructor's public diagnostic adapter. Participating
// metadata contexts transport the existing NN cause without formatting it after
// a fixed metadata refusal.
pub(crate) fn constructor_error<B: NeuralBackend, E>(
    error: eredu_nn::Error,
    context: &<B::Tensor as Tensor>::Context,
) -> ReplicatedTextDispatchError<E> {
    if B::construction_metadata(context).is_some() {
        ReplicatedTextDispatchError::Metadata(error)
    } else {
        ReplicatedTextDispatchError::Architecture(error.to_string())
    }
}

/// A closed alias of the actual effective configuration owned by a model.
/// No raw Arc/Weak escapes, and the shared shell retires before C or its funding.
pub struct SharedConfig<C>(Option<std::sync::Arc<SharedConfigData<C>>>);
struct SharedConfigData<C> {
    value: SharedConfigValue<C>,
    _funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
// A projected owner keeps the exact immutable parent, not a copied child.
// This private erasure releases its Box before the parent alias can retire.
enum SharedConfigValue<C> {
    Owned(C),
    Projected(Option<Box<dyn ConfigProjection<C>>>),
}
trait ConfigProjection<C>: Send + Sync {
    fn value(&self) -> &C;
    fn retire(self: Box<Self>);
}
struct ProjectedConfig<P, C> {
    parent: Option<SharedConfig<P>>,
    project: fn(&P) -> &C,
}
impl<P: Send + Sync + 'static, C: 'static> ConfigProjection<C> for ProjectedConfig<P, C> {
    fn value(&self) -> &C {
        (self.project)(
            self.parent
                .as_ref()
                .expect("projected configuration is live"),
        )
    }
    fn retire(mut self: Box<Self>) {
        let parent = self.parent.take();
        drop(self);
        drop(parent);
    }
}
impl<C> Drop for SharedConfigValue<C> {
    fn drop(&mut self) {
        if let Self::Projected(projected) = self {
            if let Some(projected) = projected.take() {
                projected.retire();
            }
        }
    }
}

impl<C> std::ops::Deref for SharedConfig<C> {
    type Target = C;
    fn deref(&self) -> &C {
        self.value()
    }
}
impl<C: std::fmt::Debug> std::fmt::Debug for SharedConfig<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value().fmt(formatter)
    }
}
impl<C> Clone for SharedConfig<C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<C> Drop for SharedConfig<C> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(std::sync::Arc::into_inner(owner));
        }
    }
}
impl<C> SharedConfig<C> {
    fn value(&self) -> &C {
        match &self
            .0
            .as_deref()
            .expect("effective configuration is live")
            .value
        {
            SharedConfigValue::Owned(value) => value,
            SharedConfigValue::Projected(projected) => projected
                .as_ref()
                .expect("projected configuration is live")
                .value(),
        }
    }

    fn owner_bytes() -> Result<usize, eredu_nn::Error> {
        use std::{alloc::Layout, mem::size_of, sync::atomic::AtomicUsize};
        let shell = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<SharedConfigData<C>>())
            .map_err(|_| WorkspaceMetadataError::Overflow)?
            .0
            .pad_to_align();
        let controls = [
            shell.size(),
            size_of::<SharedConfigData<C>>(),
            size_of::<Self>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<Layout>(),
            size_of::<Option<usize>>(),
            size_of::<Result<usize, eredu_nn::Error>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| WorkspaceMetadataError::Overflow.into())
    }

    pub(crate) fn new(
        value: C,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self, eredu_nn::Error> {
        let bytes = Self::owner_bytes()?
            .checked_add(size_of::<C>())
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if let Some(metadata) = metadata {
            metadata.charge_metadata(bytes)?;
        }
        Ok(Self(Some(std::sync::Arc::new(SharedConfigData {
            value: SharedConfigValue::Owned(value),
            _funding: metadata.and_then(|metadata| metadata.metadata_funding()),
        }))))
    }

    /// Retains a borrowed immutable member of this actual parent configuration.
    /// Both new shells are paid before either allocation or parent alias birth.
    pub(crate) fn project<D: 'static>(
        &self,
        project: fn(&C) -> &D,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<SharedConfig<D>, eredu_nn::Error>
    where
        C: Send + Sync + 'static,
    {
        use std::alloc::Layout;
        let controls = [
            SharedConfig::<D>::owner_bytes()?,
            Layout::new::<ProjectedConfig<C, D>>().size(),
            size_of::<ProjectedConfig<C, D>>(),
            size_of::<Box<dyn ConfigProjection<D>>>(),
            size_of::<Box<ProjectedConfig<C, D>>>(),
            size_of::<SharedConfig<C>>(),
            size_of::<fn(&C) -> &D>(),
            size_of::<Layout>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if let Some(metadata) = metadata {
            metadata.charge_metadata(bytes)?;
        }
        let projected = Box::new(ProjectedConfig {
            parent: Some(self.clone()),
            project,
        });
        Ok(SharedConfig(Some(std::sync::Arc::new(SharedConfigData {
            value: SharedConfigValue::Projected(Some(projected)),
            _funding: metadata.and_then(|metadata| metadata.metadata_funding()),
        }))))
    }
}

// This is an alias of the effective C already used by the model, not a second
// configuration cache. The source's inline OnceLock owns no copied maps.
#[derive(Clone)]
pub(crate) enum EffectiveConfig {
    Gemma2(SharedConfig<crate::gemma2::ModelArgs>),
    Llama(SharedConfig<crate::llama::ModelArgs>),
    Nanbeige(SharedConfig<crate::nanbeige::ModelArgs>),
    K2Horizon(SharedConfig<crate::k2_horizon::ModelArgs>),
    Qwen(SharedConfig<crate::qwen::ModelArgs>),
    Lfm2(SharedConfig<crate::lfm2::ModelArgs>),
    NemotronH(SharedConfig<crate::nemotron_h::ModelArgs>),
    QwenHybrid(SharedConfig<crate::qwen::hybrid::HybridConfig>),
    KimiLinear(SharedConfig<crate::kimi_linear::ModelArgs>),
    DeepSeekV3(SharedConfig<crate::deepseek::V3Args>),
    DeepSeekV4(SharedConfig<crate::deepseek::V4Args>),
}
pub(crate) trait EffectiveConfigType: Clone {
    fn effective(owner: &EffectiveConfig) -> Option<&SharedConfig<Self>>;
    fn erase(owner: SharedConfig<Self>) -> EffectiveConfig;
}
macro_rules! effective_type {
    ($variant:ident, $ty:ty) => {
        impl EffectiveConfigType for $ty {
            fn effective(owner: &EffectiveConfig) -> Option<&SharedConfig<Self>> {
                match owner {
                    EffectiveConfig::$variant(owner) => Some(owner),
                    _ => None,
                }
            }
            fn erase(owner: SharedConfig<Self>) -> EffectiveConfig {
                EffectiveConfig::$variant(owner)
            }
        }
    };
}
effective_type!(Gemma2, crate::gemma2::ModelArgs);
effective_type!(Llama, crate::llama::ModelArgs);
effective_type!(Nanbeige, crate::nanbeige::ModelArgs);
effective_type!(K2Horizon, crate::k2_horizon::ModelArgs);
effective_type!(Qwen, crate::qwen::ModelArgs);
effective_type!(Lfm2, crate::lfm2::ModelArgs);
effective_type!(NemotronH, crate::nemotron_h::ModelArgs);
effective_type!(QwenHybrid, crate::qwen::hybrid::HybridConfig);
effective_type!(KimiLinear, crate::kimi_linear::ModelArgs);
effective_type!(DeepSeekV3, crate::deepseek::V3Args);
effective_type!(DeepSeekV4, crate::deepseek::V4Args);

pub(crate) fn exact_selection(
    source: &PreparedModelSources,
    selected: &SelectedReplicatedTextRealization,
) -> bool {
    // A requirements field is nonzero-sized and lives inside the selection's
    // closed Arc. Equality of its address therefore binds the actual owner,
    // including this selection's effective formats and lowering decisions.
    std::ptr::eq(
        source.selected().text_realization().requirements(),
        selected.requirements(),
    )
}

fn retained_effective<C: EffectiveConfigType>(
    source: &impl ReplicatedTextConstructionSource,
    selected: &SelectedReplicatedTextRealization,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<Option<SharedConfig<C>>, eredu_nn::Error> {
    let Some(source) = source.prepared_sources() else {
        return Ok(None);
    };
    if !exact_selection(source, selected) {
        return Err(WorkspaceMetadataError::Unqualified.into());
    }
    source
        .construction_semantics()
        .config
        .get()
        .map(|owner| {
            let config = C::effective(owner).ok_or(WorkspaceMetadataError::Unqualified)?;
            if let Some(metadata) = metadata {
                metadata.charge_metadata(
                    std::mem::size_of::<SharedConfig<C>>()
                        + std::mem::size_of::<ConfigOwner<C>>()
                        + std::mem::size_of::<Result<Option<SharedConfig<C>>, eredu_nn::Error>>(),
                )?;
            }
            Ok(config.clone())
        })
        .transpose()
}

fn construct_effective<B: NeuralBackend, C: EffectiveConfigType, E>(
    source: &impl ReplicatedTextConstructionSource,
    config: &C,
    selected: &SelectedReplicatedTextRealization,
    legacy: fn(&C, &SelectedReplicatedTextRealization) -> Result<C, String>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigOwner<C>, ReplicatedTextDispatchError<E>> {
    let metadata = B::construction_metadata(context);
    // Original reconstruction must borrow the completed producer, never call
    // the ordinary deep-copy producer after its metadata account was opened.
    // Public standalone/ordinary equation construction retains its prior worker.
    if source.prepared_sources().is_some()
        && metadata.is_some_and(|metadata| metadata.uses_checked_metadata())
    {
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    let config = legacy(config, selected).map_err(ReplicatedTextDispatchError::Architecture)?;
    if source.prepared_sources().is_some() && metadata.is_none() {
        SharedConfig::new(config, metadata)
            .map(ConfigOwner::Shared)
            .map_err(ReplicatedTextDispatchError::Metadata)
    } else {
        Ok(ConfigOwner::Owned(config))
    }
}

pub(crate) struct ConfigPublication {
    value: Option<EffectiveConfig>,
    source: Option<PreparedModelSources>,
}
impl ConfigPublication {
    pub(crate) fn commit<E>(
        self,
        selected: &SelectedReplicatedTextRealization,
    ) -> Result<(), ReplicatedTextDispatchError<E>> {
        let (Some(value), Some(source)) = (self.value, self.source) else {
            return Ok(());
        };
        if !exact_selection(&source, selected) {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        // Concurrent exact constructions may both finish. The first immutable
        // alias wins; the losing alias drops after set returns. Both models keep
        // their own actual owner and were validated against the same selection.
        let _ = source.construction_semantics().config.set(value);
        Ok(())
    }
}
pub(crate) fn publication<B: NeuralBackend, C: EffectiveConfigType, E>(
    source: &impl ReplicatedTextConstructionSource,
    config: &ConfigOwner<C>,
    selected: &SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigPublication, ReplicatedTextDispatchError<E>> {
    if source
        .prepared_sources()
        .is_some_and(|source| !exact_selection(source, selected))
    {
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    // Even the empty publication is an actual fixed constructor/result frame.
    if let Some(metadata) = B::construction_metadata(context) {
        metadata.charge_metadata(std::mem::size_of::<ConfigPublication>()
            + std::mem::size_of::<Result<ConfigPublication, ReplicatedTextDispatchError<E>>>()
            + std::mem::size_of::<EffectiveConfig>()
            + std::mem::size_of::<Result<(), EffectiveConfig>>()
            + std::mem::size_of::<Result<(), ReplicatedTextDispatchError<E>>>())
            .map_err(|cause| ReplicatedTextDispatchError::Metadata(cause.into()))?;
    }
    let (Some(source), ConfigOwner::Shared(config)) = (source.prepared_sources(), config) else {
        return Ok(ConfigPublication {
            value: None,
            source: None,
        });
    };
    Ok(ConfigPublication {
        value: Some(C::erase(config.clone())),
        source: Some(source.clone()),
    })
}

pub(super) fn selected_k2_horizon<B: NeuralBackend, E>(
    source: &impl ReplicatedTextConstructionSource,
    config: &crate::k2_horizon::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ConfigOwner<crate::k2_horizon::ModelArgs>, ReplicatedTextDispatchError<E>> {
    if let Some(owner) = retained_effective(source, selected, B::construction_metadata(context))
        .map_err(ReplicatedTextDispatchError::Metadata)?
    {
        return Ok(ConfigOwner::Shared(owner));
    }
    construct_effective::<B, _, E>(source, config, selected, selected_k2_horizon_args, context)
}

#[cfg(test)]
mod projection_tests {
    use super::SharedConfig;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn projected_configuration_retires_parent_with_final_alias() {
        struct Parent {
            value: usize,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for Parent {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let parent = SharedConfig::new(
            Parent {
                value: 17,
                drops: drops.clone(),
            },
            None,
        )
        .unwrap();
        let projected = parent.project(|parent| &parent.value, None).unwrap();
        let final_alias = projected.clone();
        drop(parent);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(projected);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(*final_alias, 17);
        drop(final_alias);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
