//! Shared backend-neutral decoder mechanics.
//!
//! Architecture families retain configuration, checkpoint naming, identity, and
//! policy while reusing these statically dispatched decoder operations.

pub(crate) mod identity;
pub(crate) mod parameter_metadata;
pub(crate) use parameter_metadata::static_groups as static_parallel_parameter_groups_with_metadata;
pub(crate) mod construction_specs;
#[cfg(test)]
mod instrumentation_tests;
mod module_metadata;
pub(crate) use module_metadata::ModuleMetadata;
mod prefill_observations;
/// Borrowed declarations and shared ordering for actual pinned-module construction.
pub mod static_construction;
pub(crate) use prefill_observations::{
    append_dense_component_prefill_observations, append_routed_prefill_observations,
    append_routed_prefill_path, media_prefill_observation_declarations,
    ordinary_prefill_observation_declarations,
};
pub(crate) use static_construction::StaticModuleSpecView;

use parameter_metadata::{DeclarationDestination, NormalizationName, ParameterGroupError};
use std::ops::Range;

pub(crate) mod attention_partition;
/// Shared-weight repeated stacks with independent state for each invocation.
pub(crate) mod repeated;

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::cache::LayerCachePolicy;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, AttentionRequest, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    Error, FusedProjectionLayout, FusedProjectionSegment, GatedProductPolicy, GroupedNeuralBackend,
    Index, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Parameter, ParameterSpec, RotaryOperator, RotaryPosition, RotarySpec,
    RotarySubspace, Tensor, VocabularyParallelRange,
};
use eredu_runtime::{
    module_parameter_group, partitioned_module_parameter_group, partitioned_projection_group,
    segmented_projection_group, ArchitectureParameterDescription, ExecutionGraph,
    ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    LayeredPartitionInput, LocalModelLayout, MemberSharding, OwnedParameterGroupSpec,
    ParallelLayeredArchitecture, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterRole, PartitionedLayeredArchitecture, ProjectionSharding, StateLayout,
    TensorPlacement,
};

/// Optional component instrumentation at the values actually consumed downstream.
/// An absent observer neither clones nor retains tensors and constructs no paths.
pub struct ComponentInstrumentation<'a, T> {
    observer: Option<&'a mut dyn eredu_runtime::ActivationObserver<T, Error>>,
    unit_path: &'a str,
}

pub mod unary;

impl<T: Clone> ComponentInstrumentation<'_, T> {
    /// Ordinary execution, without additional tensor work.
    pub fn disabled() -> Self {
        Self {
            observer: None,
            unit_path: "",
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.observer.is_some()
    }

    /// Read-only evidence from an existing value, without a second mutable hook.
    pub(crate) fn observe(&mut self, boundary: &str, value: &T) -> Result<(), Error> {
        if let Some(observer) = self.observer.as_deref_mut() {
            observer.observe(&format!("{}.{}", self.unit_path, boundary), value)?;
        }
        Ok(())
    }

    /// Borrows an already admitted observer for one architecture invocation.
    pub fn new<'a>(
        unit_path: &'a str,
        observer: &'a mut dyn eredu_runtime::ActivationObserver<T, Error>,
    ) -> ComponentInstrumentation<'a, T> {
        ComponentInstrumentation {
            observer: if observer.observes_activations() {
                Some(observer)
            } else {
                None
            },
            unit_path,
        }
    }

    /// Reborrows the same observer for nested architecture-owned routing seams.
    pub(crate) fn observer(
        &mut self,
    ) -> Option<&mut dyn eredu_runtime::ActivationObserver<T, Error>> {
        match &mut self.observer {
            Some(observer) => Some(&mut **observer),
            None => None,
        }
    }

    /// Reuses the same observer under a nested semantic invocation. Disabled
    /// instrumentation creates no path string and performs no tensor work.
    pub(crate) fn with_scope<R>(
        &mut self,
        suffix: &str,
        execute: impl FnOnce(&mut ComponentInstrumentation<'_, T>) -> R,
    ) -> R {
        match self.observer.as_deref_mut() {
            Some(observer) => {
                let path = format!("{}.{suffix}", self.unit_path);
                execute(&mut ComponentInstrumentation::new(&path, observer))
            }
            None => execute(&mut ComponentInstrumentation::disabled()),
        }
    }

    /// Records the original value, applies intervention, then records the effective
    /// value under a distinct read-only path. Only the effective value is returned.
    pub fn apply(&mut self, boundary: &str, value: T) -> Result<T, Error> {
        let Some(observer) = self.observer.as_mut() else {
            return Ok(value);
        };
        let path = format!("{}.{}", self.unit_path, boundary);
        observer.observe(&path, &value)?;
        let effective = observer.intervene(&path, &value)?.unwrap_or(value);
        observer.observe(&format!("{path}.effective"), &effective)?;
        Ok(effective)
    }
}

struct ComponentProjectionObserver<'a, T> {
    path: String,
    observer: &'a mut dyn eredu_runtime::ActivationObserver<T, Error>,
}
impl<T: Tensor> eredu_nn::ProjectionInputObserver<T> for ComponentProjectionObserver<'_, T> {
    fn observe(&mut self, value: &T) -> Result<(), Error> {
        self.observer.observe(&self.path, value)
    }
    fn observe_generated(
        &mut self,
        prototype: &T,
        source: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<T, Error>,
    ) -> Result<(), Error> {
        self.observer.observe_generated(
            &self.path,
            prototype,
            &eredu_runtime::capture::generated_capture_source(source),
            generate,
        )
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        prototype: &T,
        source: &eredu_nn::GeneratedTensorSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<T, Error>,
    ) -> Result<(), Error> {
        self.observer.observe_generated_retained(
            &self.path,
            prototype,
            &eredu_runtime::capture::generated_capture_source(source),
            factory,
        )
    }
}
impl<T: Tensor> ComponentInstrumentation<'_, T> {
    /// Attaches unit evidence to the exact expert request consumed by a provider.
    /// The disabled path preserves the request without constructing a scope path.
    pub(crate) fn routed<'data, P, R>(
        &mut self,
        boundary: &str,
        request: eredu_runtime::RoutedExpertRequest<'data, '_, T>,
        execute: impl for<'unit> FnOnce(
            eredu_runtime::RoutedExpertRequest<'data, 'unit, T>,
        ) -> Result<R, P>,
    ) -> Result<R, Error>
    where
        P: std::error::Error + Send + Sync + 'static,
    {
        match self.observer.as_deref_mut() {
            Some(observer) => eredu_runtime::with_routed_unit_observer(
                observer,
                &format!("{}.{}", self.unit_path, boundary),
                request,
                execute,
            )
            .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error),
            None => execute(request).map_err(Error::backend_retained_source),
        }
    }

    /// Executes the existing projection with read-only actual-input evidence.
    pub(crate) fn project<B: NeuralBackend<Tensor = T>>(
        &mut self,
        boundary: &str,
        projection: &mut B::Linear,
        input: &T,
        parallel: Option<&B::ParallelContext>,
        context: &T::Context,
    ) -> Result<T, Error> {
        let mut observer =
            self.observer
                .as_deref_mut()
                .map(|observer| ComponentProjectionObserver {
                    path: format!("{}.{}", self.unit_path, boundary),
                    observer,
                });
        let observer = observer
            .as_mut()
            .map(|observer| observer as &mut dyn eredu_nn::ProjectionInputObserver<T>);
        match parallel {
            Some(parallel) => B::row_parallel_linear_with_input_observer(
                projection, input, parallel, context, observer,
            ),
            None => projection.forward_with_input_observer(input, context, observer),
        }
    }
    /// Keeps grouped multiplication evidence in its actual rank-four geometry;
    /// generated evidence is admitted by the observer before materialization.
    pub(crate) fn project_grouped<B: eredu_nn::GroupedNeuralBackend<Tensor = T>>(
        &mut self,
        boundary: &str,
        projection: &mut B::Linear,
        input: &T,
        groups: i32,
        output_per_group: i32,
        context: &T::Context,
    ) -> Result<T, Error> {
        let mut observer =
            self.observer
                .as_deref_mut()
                .map(|observer| ComponentProjectionObserver {
                    path: format!("{}.{}", self.unit_path, boundary),
                    observer,
                });
        B::grouped_linear_with_input_observer(
            projection,
            input,
            groups,
            output_per_group,
            context,
            observer
                .as_mut()
                .map(|observer| observer as &mut dyn eredu_nn::ProjectionInputObserver<T>),
        )
    }

    /// Borrows the final collapse coefficients from the selected mechanism.
    pub(crate) fn collapse_streams<B: eredu_nn::HyperNeuralBackend<Tensor = T>>(
        &mut self,
        boundary: &str,
        head: &mut eredu_nn::HyperHead<B>,
        input: &T,
        context: &T::Context,
    ) -> Result<T, Error> {
        let mut observer =
            self.observer
                .as_deref_mut()
                .map(|observer| ComponentProjectionObserver {
                    path: format!("{}.{}", self.unit_path, boundary),
                    observer,
                });
        head.forward_with_coefficients_observer(
            input,
            context,
            observer
                .as_mut()
                .map(|o| o as &mut dyn eredu_nn::TensorValueObserver<T>),
        )
    }

    pub(crate) fn project_embedding<E: EmbeddingOperator<T>>(
        &mut self,
        boundary: &str,
        embedding: &mut E,
        input: &T,
        context: &T::Context,
    ) -> Result<T, Error> {
        let mut observer =
            self.observer
                .as_deref_mut()
                .map(|observer| ComponentProjectionObserver {
                    path: format!("{}.{}", self.unit_path, boundary),
                    observer,
                });
        embedding.as_linear_with_input_observer(
            input,
            context,
            observer
                .as_mut()
                .map(|observer| observer as &mut dyn eredu_nn::ProjectionInputObserver<T>),
        )
    }

    pub(crate) fn project_vocabulary<B: eredu_nn::DistributedNeuralBackend<Tensor = T>>(
        &mut self,
        boundary: &str,
        projection: &mut B::Linear,
        input: &T,
        parallel: &B::ParallelContext,
        context: &T::Context,
    ) -> Result<T, Error> {
        let mut observer =
            self.observer
                .as_deref_mut()
                .map(|observer| ComponentProjectionObserver {
                    path: format!("{}.{}", self.unit_path, boundary),
                    observer,
                });
        B::vocabulary_parallel_project_with_input_observer(
            projection,
            input,
            parallel,
            context,
            observer
                .as_mut()
                .map(|observer| observer as &mut dyn eredu_nn::ProjectionInputObserver<T>),
        )
    }

    pub(crate) fn project_vocabulary_embedding<
        B: eredu_nn::DistributedNeuralBackend<Tensor = T>,
    >(
        &mut self,
        boundary: &str,
        embedding: &mut B::Embedding,
        input: &T,
        parallel: &B::ParallelContext,
        context: &T::Context,
    ) -> Result<T, Error> {
        let mut observer =
            self.observer
                .as_deref_mut()
                .map(|observer| ComponentProjectionObserver {
                    path: format!("{}.{}", self.unit_path, boundary),
                    observer,
                });
        B::vocabulary_parallel_embedding_project_with_input_observer(
            embedding,
            input,
            parallel,
            context,
            observer
                .as_mut()
                .map(|observer| observer as &mut dyn eredu_nn::ProjectionInputObserver<T>),
        )
    }

    pub(crate) fn normalize_readout<N: NormalizationOperator<T>>(
        &mut self,
        hidden: &T,
        norm: &mut N,
        context: &T::Context,
    ) -> Result<T, Error> {
        let effective;
        let hidden = if self.observer.is_some() {
            effective = self.apply("residual", hidden.clone())?;
            &effective
        } else {
            hidden
        };
        let hidden = norm.forward(hidden, context)?;
        self.apply("normalized", hidden)
    }
}

/// Stable identity of the shared decoder target execution group.
pub const TARGET_EXECUTION_GROUP: &str = "target";
/// Stable identity of an ordinary one-group text decoder.
pub const TEXT_DECODER_EXECUTION_GROUP: &str = "text_decoder";

/// Canonical field segments used by one shared decoder block.
///
/// Architecture families can replace checkpoint vocabulary without replacing
/// the shared attention, residual, feed-forward, or parallel-placement logic.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BlockParameterFields<'a> {
    /// Self-attention module below one layer.
    pub attention: &'a str,
    /// Query projection below the attention module.
    pub attention_query: &'a str,
    /// Key projection below the attention module.
    pub attention_key: &'a str,
    /// Value projection below the attention module.
    pub attention_value: &'a str,
    /// Output projection below the attention module.
    pub attention_output: &'a str,
    /// Optional learned attention-sink parameter.
    pub attention_sinks: &'a str,
    /// Optional query normalization below the attention module.
    pub attention_query_norm: &'a str,
    /// Optional key normalization below the attention module.
    pub attention_key_norm: &'a str,
    /// Feed-forward module below one layer.
    pub feed_forward: &'a str,
    /// Gate projection below a split feed-forward module.
    pub feed_forward_gate: &'a str,
    /// Up projection below a split feed-forward module.
    pub feed_forward_up: &'a str,
    /// Output projection below the feed-forward module.
    pub feed_forward_output: &'a str,
    /// Pre-attention normalization below one layer.
    pub input_norm: &'a str,
    /// Pre-feed-forward normalization below one layer.
    pub post_attention_norm: &'a str,
}

impl Default for BlockParameterFields<'_> {
    fn default() -> Self {
        Self {
            attention: "self_attn",
            attention_query: "q_proj",
            attention_key: "k_proj",
            attention_value: "v_proj",
            attention_output: "o_proj",
            attention_sinks: "sinks",
            attention_query_norm: "q_norm",
            attention_key_norm: "k_norm",
            feed_forward: "mlp",
            feed_forward_gate: "gate_proj",
            feed_forward_up: "up_proj",
            feed_forward_output: "down_proj",
            input_norm: "input_layernorm",
            post_attention_norm: "post_attention_layernorm",
        }
    }
}

impl BlockParameterFields<'_> {
    fn validate(self) -> Result<Self, Error> {
        self.validate_with(|args| Error::backend(args))
    }

    fn validate_with<E>(
        self,
        mut invalid: impl FnMut(std::fmt::Arguments<'_>) -> E,
    ) -> Result<Self, E> {
        for (role, field) in [
            ("attention module", self.attention),
            ("attention query projection", self.attention_query),
            ("attention key projection", self.attention_key),
            ("attention value projection", self.attention_value),
            ("attention output projection", self.attention_output),
            ("attention sinks", self.attention_sinks),
            ("attention query norm", self.attention_query_norm),
            ("attention key norm", self.attention_key_norm),
            ("feed-forward module", self.feed_forward),
            ("feed-forward gate projection", self.feed_forward_gate),
            ("feed-forward up projection", self.feed_forward_up),
            ("feed-forward output projection", self.feed_forward_output),
            ("input norm", self.input_norm),
            ("post-attention norm", self.post_attention_norm),
        ] {
            if field.trim().is_empty() {
                return Err(invalid(format_args!(
                    "decoder block {role} field must not be empty"
                )));
            }
        }
        Ok(self)
    }
}

/// Geometry and policy required by the shared decoder mechanics.
pub trait Config: 'static {
    /// Stable architecture family used by persistence identity.
    fn model_family(&self) -> &'static str;
    /// Stable normalized model identity.
    fn model_identity(&self) -> &str;
    /// Stable identity of the complete normalized architecture policy.
    ///
    /// Implementations must bind every construction, equation, state, and
    /// encoding policy that can affect decoder or cache compatibility.
    fn architecture_fingerprint(&self) -> String;
    /// Constructs the same semantic fingerprint through the caller's metadata account.
    /// All intermediate strings and sort storage must use that account as well.
    fn architecture_fingerprint_with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<String, Error>;
    /// Canonical parameter namespace for this decoder body.
    fn parameter_root(&self) -> &str {
        "model"
    }
    /// Authoritative shared parameter identity for a logical invocation's slot.
    /// Returning an alias preserves separate execution/state ownership while
    /// binding parameter queries and edits to the shared value.
    fn parameter_alias(&self, _name: &str) -> Option<String> {
        None
    }
    /// Produces the same source alias through the checked construction account.
    fn parameter_alias_with_metadata(
        &self,
        _name: &str,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Canonical parameter fields used within each shared decoder block.
    fn block_parameter_fields(&self) -> BlockParameterFields<'_> {
        BlockParameterFields::default()
    }
    /// Returns the canonical routed-observation point for one decoder layer.
    fn routed_observation_points(
        &self,
        _unit_path: &str,
        _layer: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Error> {
        crate::decoder::identity::Metadata::new(metadata_context).controls::<(
            &Self,
            &str,
            usize,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            Option<eredu_runtime::RoutedObservationPoints>,
            Result<Option<eredu_runtime::RoutedObservationPoints>, eredu_nn::Error>,
        )>()?;
        Ok(None)
    }
    /// Validates architecture-owned configuration policy.
    fn validate_config(&self) -> Result<(), Error>;
    /// Validates the same configuration with a caller-owned diagnostic destination.
    /// Custom configurations must provide their actual checked producer.
    fn validate_config_with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), Error> {
        if context.uses_checked_metadata() {
            Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
        } else {
            self.validate_config()
        }
    }
    /// Transformer hidden size.
    fn hidden_size(&self) -> i32;
    /// Number of decoder layers.
    fn num_hidden_layers(&self) -> i32;
    /// Optional learned RMS normalization after a block's final residual.
    /// The returned canonical parameter belongs to that execution unit.
    fn block_output_normalization(&self, _layer: usize) -> Option<String> {
        None
    }
    /// SwiGLU intermediate width.
    fn intermediate_size(&self) -> i32;
    /// Number of query heads.
    fn num_attention_heads(&self) -> i32;
    /// Number of key/value heads.
    fn num_key_value_heads(&self) -> i32;
    /// Per-head width.
    fn head_dim(&self) -> i32;
    /// RMSNorm epsilon.
    fn rms_norm_epsilon(&self) -> f32;
    /// Offset added to learned RMS normalization scales.
    fn normalization_offset(&self) -> f32 {
        0.0
    }
    /// Independent float32 RMS reductions with a distinct scale per feature.
    fn normalization_groups(&self) -> Option<i32> {
        None
    }
    /// Whether Q/K normalization owns a different scale vector for each head.
    fn query_key_norm_per_head_weights(&self) -> bool {
        false
    }
    /// Separate projection and activation applied to attended values.
    fn attention_output_gate(&self) -> Option<(&str, OutputGateActivation)> {
        None
    }
    /// Paired dimensions rotated across the two halves of each complete head.
    fn rotary_pair_dimensions(&self) -> i32 {
        self.head_dim()
    }
    /// Optional normalization after attention, before its residual addition.
    fn attention_output_normalization(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Optional normalization after feed-forward, before its residual addition.
    fn feed_forward_output_normalization(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Multiplier applied once to input token embeddings.
    fn embedding_scale(&self) -> f32 {
        1.0
    }
    /// Optional positive tanh cap on final vocabulary logits.
    fn output_softcap(&self) -> Option<f32> {
        None
    }
    /// Scale applied to attention scores before any soft cap.
    fn attention_scale(&self) -> f32 {
        (self.head_dim() as f32).sqrt().recip()
    }
    /// Optional positive tanh cap on attention scores before masking.
    fn attention_softcap(&self) -> Option<f32> {
        None
    }
    /// Score and softmax rounding boundaries shared by every cache layout.
    fn attention_arithmetic(&self) -> eredu_nn::AttentionArithmetic {
        eredu_nn::AttentionArithmetic::Fused
    }
    /// Vocabulary size.
    fn vocabulary_size(&self) -> i32;
    /// Whether one attention projection owns a learned bias.
    fn attention_bias(&self, projection: AttentionProjection) -> bool;
    /// Physical construction of the query/key/value input projection.
    /// Whether this layer receives mixed values from its projection policy.
    fn external_attention_value(&self, _layer: usize) -> bool {
        false
    }

    /// Encoding of the projection that supplies this layer's key/value-head rows.
    /// Routed value providers override this with their declared bank encoding.
    fn attention_value_format(&self, layer: usize) -> LinearFormat {
        let name = parameter_metadata::default_attention_value_name(self, layer);
        self.linear_format(&name.to_string())
    }
    /// Builds the same selected value-format lookup through a metadata account.
    /// An overriding format policy must supply this companion explicitly.
    fn attention_value_format_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<LinearFormat, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Builds an optional block-output normalization name through the caller account.
    fn block_output_normalization_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Builds an actual post-attention normalization name through the caller account.
    fn attention_output_normalization_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Builds an actual post-feed-forward normalization name through the caller account.
    fn feed_forward_output_normalization_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Physical layout of the query, key, and optional ordinary value projections.
    fn attention_projection_layout(&self) -> AttentionProjectionLayout<'_> {
        AttentionProjectionLayout::Split
    }
    /// Whether each attention layer owns one learned logit per query head.
    fn learned_attention_sinks(&self) -> bool {
        false
    }
    /// Optional per-head Q/K RMS-normalization epsilon.
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        None
    }
    /// Whether projections own MLP biases.
    fn mlp_bias(&self) -> bool;
    /// Physical construction of the gate/up input projection.
    fn gated_projection_layout(&self) -> GatedProjectionLayout<'_> {
        GatedProjectionLayout::Split
    }
    /// Optional equation policy applied by each dense gated product.
    fn gated_product_policy(&self) -> Option<GatedProductPolicy> {
        None
    }
    /// Whether the language-model head is tied to input embeddings.
    fn tie_word_embeddings(&self) -> bool;
    /// Exact per-layer attention policy.
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy>;
    /// Physical encoding selected for one canonical checkpoint parameter.
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization>;
    /// Performs the same embedding encoding lookup using the caller's metadata destination.
    fn weight_quantization_with_metadata(
        &self,
        _name: &str,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<WeightQuantization>, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Complete matrix encoding, including block-FP8 companions.
    fn linear_format(&self, name: &str) -> eredu_checkpoint::LinearFormat {
        self.weight_quantization(name).into()
    }
    /// Performs the same format lookup using the caller's metadata destination.
    fn linear_format_with_metadata(
        &self,
        _name: &str,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<LinearFormat, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }
    /// Complete rotary-position construction specification.
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec;
    /// Normalizes the same rotary configuration without unaccounted scratch.
    fn rotary_spec_with_metadata(
        &self,
        _dimensions: i32,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<RotarySpec, Error> {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Whether this decoder stack applies rotary position encoding.
    fn rotary_enabled(&self) -> bool {
        true
    }
}

/// Additional neural mechanisms selected by shared decoder equation policy.
pub(crate) fn operator_requirements(config: &impl Config) -> eredu_nn::NeuralOperatorCapabilities {
    use eredu_nn::{GatedProductActivation, NeuralOperatorCapabilities as Caps};
    let mut required = Caps::NONE;
    if let Some((_, activation)) = config.attention_output_gate() {
        required = required.union(match activation {
            OutputGateActivation::Sigmoid => Caps::SIGMOID,
            OutputGateActivation::Softplus(_) => Caps::SOFTPLUS,
            OutputGateActivation::Silu => Caps::NONE,
        });
    }
    if config.learned_attention_sinks() {
        required = required.union(Caps::ATTENTION_SINKS);
    }
    if config.output_softcap().is_some() {
        required = required.union(Caps::TANH);
    }
    if config.attention_softcap().is_some() {
        required = required.union(Caps::ATTENTION_SOFTCAP);
    }
    if config
        .gated_product_policy()
        .is_some_and(|p| matches!(p.activation(), GatedProductActivation::GeluApproximate))
    {
        required = required.union(Caps::GELU_APPROXIMATE);
    }
    required
}

/// Configuration that can derive one tensor-parallel local block without
/// backend module construction.
pub trait PartitionedConfig: Config + Clone {
    /// Ordered routed invocations inside a logical decoder block.
    fn routed_bank_order(&self, _layer: usize) -> Vec<eredu_runtime::RoutedBankId> {
        vec![eredu_runtime::RoutedBankId::new(0)]
    }

    /// Tensor sums surrounding a routed invocation in execution order.
    fn routed_bank_tensor_reductions(
        &self,
        _layer: usize,
        _bank: eredu_runtime::RoutedBankId,
    ) -> Result<(usize, usize), Error> {
        Ok((1, 1))
    }
    /// Rewrites only the local attention-head and feed-forward geometry.
    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), Error>;

    /// Derives one rank-local block configuration from an exact physical layout.
    ///
    /// Dense families inherit the ordinary split-projection derivation. Routed
    /// families override this hook because their expert bank has an additional
    /// owner-local group axis that is not the dense MLP intermediate axis.
    fn local_block_config(&self, layer: usize, layout: &LocalModelLayout) -> Result<Self, Error> {
        dense_local_block_config(self, layer, layout)
    }

    /// Validates that a selected parameter description belongs to this family.
    ///
    /// The default remains the exact dense declaration. Routed families may
    /// validate their distinct router/bank topology without weakening dense
    /// construction.
    fn validate_partition_parameters(
        &self,
        parameters: &ArchitectureParameterDescription,
    ) -> Result<(), Error> {
        let expected = dense_parameter_description(self).map_err(Error::backend)?;
        if &expected != parameters {
            return Err(Error::backend(
                "decoder partition belongs to a different normalized parameter topology",
            ));
        }
        Ok(())
    }
}

/// Declares cache identity shared by ordinary layered decoder families.
pub fn state_identity<C: Config>(
    args: &C,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, Error> {
    state_identity_with(
        args,
        layout,
        global_layer_start,
        topology,
        identity::Metadata::new(None),
    )
}

fn state_identity_with<C: Config>(
    args: &C,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
    metadata: identity::Metadata<'_>,
) -> Result<eredu_runtime::ModelStateIdentity, Error> {
    metadata.controls::<eredu_runtime::ModelStateIdentity>()?;
    // Configuration validation remains the authoritative architecture producer.
    args.validate_config()?;
    topology.validate_with_diagnostic(|message| metadata.prompt_error(message))?;
    let layer_count =
        usize::try_from(args.num_hidden_layers()).map_err(|error| metadata.source(error))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| metadata.error(format_args!("decoder owned state range overflowed")))?;
    if global_layer_end > layer_count {
        return Err(metadata.error(format_args!(
            "{} owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers",
            args.model_family()
        )));
    }
    let fingerprint = metadata.configured(args)?;
    eredu_runtime::ModelStateIdentity::new_with_diagnostic(
        metadata.text(args.model_family())?,
        metadata.text(args.model_identity())?,
        fingerprint,
        layer_count,
        global_layer_start,
        0,
        topology,
        |message| metadata.prompt_error(message),
    )
}

/// Semantic attention projection selected by architecture policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AttentionProjection {
    /// Query projection.
    Query,
    /// Key projection.
    Key,
    /// Value projection.
    Value,
    /// Output projection.
    Output,
}

/// Architecture-selected physical query/key/value projection layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AttentionProjectionLayout<'a> {
    /// Independent query, key, and value affine projections.
    Split,
    /// One component-major query/key/value affine projection.
    Fused {
        /// Canonical projection field below the attention module.
        field: &'a str,
    },
}

/// Architecture-selected physical gate/up projection layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GatedProjectionLayout<'a> {
    /// Independent gate and up affine projections.
    Split,
    /// One component-major gate/up affine projection.
    Fused {
        /// Canonical projection field below the MLP module.
        field: &'a str,
    },
}

/// Construction policy for one named table in a deterministic embedding sum.
#[derive(Debug, Clone)]
pub struct NamedEmbeddingSpec {
    /// Stable semantic stream name used for validation diagnostics.
    pub name: String,
    /// Canonical embedding parameter and physical format.
    pub embedding: EmbeddingSpec,
    /// Strict or diagnostic zero-sentinel lookup behavior.
    pub lookup: EmbeddingLookupPolicy,
}

/// One backend-native named table participating in an embedding sum.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct NamedEmbedding<B: NeuralBackend> {
    /// Backend-native embedding operator.
    pub embedding: B::Embedding,
    #[parameter(skip, metadata)]
    name: String,
    #[parameter(skip, metadata)]
    lookup: EmbeddingLookupPolicy,
}

/// Ordered multi-stream embedding lookup and deterministic sum.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MultiTableEmbedding<B: NeuralBackend> {
    /// Tables in semantic stream order.
    pub tables: Vec<NamedEmbedding<B>>,
}

impl<B: NeuralBackend> MultiTableEmbedding<B> {
    /// Builds validated named tables without materializing checkpoint values.
    pub fn new(
        specs: impl IntoIterator<Item = NamedEmbeddingSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs = specs.into_iter().collect::<Vec<_>>();
        if specs.is_empty() {
            return Err(Error::backend(
                "multi-table embedding sum requires at least one table",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut dimensions = None;
        let mut tables = Vec::with_capacity(specs.len());
        for spec in specs {
            if spec.name.trim().is_empty() || !names.insert(spec.name.clone()) {
                return Err(Error::backend(format!(
                    "multi-table embedding name {:?} is empty or duplicated",
                    spec.name
                )));
            }
            spec.lookup.validate()?;
            if spec.embedding.vocabulary <= 0 || spec.embedding.dimensions <= 0 {
                return Err(Error::backend(format!(
                    "embedding table {:?} has invalid geometry vocabulary={} dimensions={}",
                    spec.name, spec.embedding.vocabulary, spec.embedding.dimensions
                )));
            }
            if dimensions
                .replace(spec.embedding.dimensions)
                .is_some_and(|expected| expected != spec.embedding.dimensions)
            {
                return Err(Error::backend(format!(
                    "embedding table {:?} width {} differs from preceding width {:?}",
                    spec.name, spec.embedding.dimensions, dimensions
                )));
            }
            tables.push(NamedEmbedding {
                embedding: B::embedding(spec.embedding, context)?,
                name: spec.name,
                lookup: spec.lookup,
            });
        }
        Ok(Self { tables })
    }

    /// Builds rank-local vocabulary tables with validated global ownership.
    pub fn new_vocabulary_parallel(
        specs: impl IntoIterator<Item = NamedEmbeddingSpec>,
        ranges: impl IntoIterator<Item = eredu_nn::VocabularyParallelRange>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let specs = specs.into_iter().collect::<Vec<_>>();
        let ranges = ranges.into_iter().collect::<Vec<_>>();
        if specs.is_empty() || specs.len() != ranges.len() {
            return Err(Error::backend(
                "parallel multi-table embeddings require one range per table",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut dimensions = None;
        let mut tables = Vec::with_capacity(specs.len());
        for (spec, range) in specs.into_iter().zip(ranges) {
            if spec.name.trim().is_empty() || !names.insert(spec.name.clone()) {
                return Err(Error::backend(
                    "parallel embedding name is empty or duplicated",
                ));
            }
            spec.lookup.validate()?;
            if dimensions
                .replace(spec.embedding.dimensions)
                .is_some_and(|expected| expected != spec.embedding.dimensions)
            {
                return Err(Error::backend("parallel embedding widths differ"));
            }
            tables.push(NamedEmbedding {
                embedding: B::vocabulary_parallel_embedding(spec.embedding, range, context)?,
                name: spec.name,
                lookup: spec.lookup,
            });
        }
        Ok(Self { tables })
    }

    /// Looks up every global-token stream and reduces rank-local contributions.
    pub fn forward_parallel(
        &mut self,
        inputs: &[&B::Tensor],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        if inputs.len() != self.tables.len() {
            return Err(Error::backend("parallel embedding input count drifted"));
        }
        let mut output: Option<B::Tensor> = None;
        for (table, input) in self.tables.iter_mut().zip(inputs) {
            let value = B::vocabulary_parallel_lookup(
                &mut table.embedding,
                input,
                table.lookup,
                parallel,
                context,
            )?;
            output = Some(match output {
                Some(output) => output.add(&value, context)?,
                None => value,
            });
        }
        output.ok_or_else(|| Error::backend("parallel embedding sum is empty"))
    }

    /// Looks up one token tensor per table and sums in declared stream order.
    pub fn forward(
        &mut self,
        tokens: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if tokens.len() != self.tables.len() {
            return Err(Error::backend(format!(
                "multi-table embedding received {} token streams, expected {}",
                tokens.len(),
                self.tables.len()
            )));
        }
        let expected_shape = tokens
            .first()
            .map(|tokens| tokens.shape().to_vec())
            .ok_or_else(|| Error::backend("multi-table embedding received no token streams"))?;
        let mut sum: Option<B::Tensor> = None;
        for (table, tokens) in self.tables.iter_mut().zip(tokens.iter().copied()) {
            if tokens.shape() != expected_shape {
                return Err(Error::backend(format!(
                    "embedding stream {:?} has token shape {:?}, expected {:?}",
                    table.name,
                    tokens.shape(),
                    expected_shape
                )));
            }
            let embedded = table.embedding.lookup(tokens, table.lookup, context)?;
            sum = Some(match sum {
                Some(sum) => sum.add(&embedded, context)?,
                None => embedded,
            });
        }
        sum.ok_or_else(|| Error::backend("multi-table embedding received no tables"))
    }

    /// Returns stable table names in deterministic stream order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|table| table.name.as_str())
    }
}

fn parameter_spec<C: Config>(
    config: &C,
    name: impl Into<String>,
) -> Result<ParameterSpec, eredu_nn::ParameterTopologyError> {
    let name = name.into();
    let alias = config
        .parameter_alias(&name)
        .map(eredu_nn::ParameterId::new)
        .transpose()?;
    let mut parameter = ParameterSpec::trainable(name)?;
    parameter.alias_of = alias;
    Ok(parameter)
}

/// Derives the canonical backend-neutral cache layout for this decoder.
pub fn cache_layout<C: Config>(config: &C) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    cache_layout_with_key_value_heads(
        config,
        std::iter::repeat_n(
            config.num_key_value_heads(),
            config.attention_schedule().len(),
        ),
    )
}

/// Declares the complete mutable-state geometry consumed by either resident
/// or bounded-residency execution.
pub fn state_layout<C: Config>(config: &C) -> Result<StateLayout, Error> {
    StateLayout::new(cache_layout(config)?).map_err(Error::backend)
}

/// Derives this actual decoder's mutable geometry in the metadata context.
/// Validation and policy construction are shared with the ordinary producer.
pub fn state_layout_with_metadata<C: Config>(
    config: &C,
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<StateLayout, Error> {
    if !context.uses_checked_metadata() {
        return state_layout(config);
    }
    use crate::state_geometry::Destination;
    let destination = crate::state_geometry::Counted::new(context, Error::backend_message);
    let schedule = cache_layout_destination(
        config,
        std::iter::repeat_n(
            config.num_key_value_heads(),
            config.attention_schedule().len(),
        ),
        &destination,
    )?;
    destination.layout(schedule)
}

/// Complete planner-derived construction geometry for one shared decoder rank.
///
/// The value is backend-neutral and is the single source of truth for local
/// unit construction, vocabulary ownership, and mutable cache geometry.
#[derive(Debug, Clone)]
pub struct LocalGeometry<C> {
    blocks: Vec<C>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl<C: Config> LocalGeometry<C> {
    /// Returns the rank-local configuration of one decoder block.
    pub fn block(&self, index: usize) -> Option<&C> {
        self.blocks.get(index)
    }

    /// Returns local decoder blocks in global execution order.
    pub fn blocks(&self) -> &[C] {
        &self.blocks
    }

    /// Returns this rank's input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns this rank's untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the cache layout derived from the same local block geometry.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Validates that this geometry was derived from this exact normalized
    /// model configuration and that its state/vocabulary views have not
    /// drifted from its local block geometry.
    pub fn validate_for(&self, config: &C) -> Result<(), ParallelPlanError> {
        let layers = usize::try_from(config.num_hidden_layers()).map_err(|_| {
            ParallelPlanError::InvalidGroup("decoder layer count exceeds usize".into())
        })?;
        if self.architecture_fingerprint != config.architecture_fingerprint()
            || self.blocks.len() != layers
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local decoder geometry belongs to a different normalized configuration"
                    .into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(config.vocabulary_size())
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (config.tie_word_embeddings(), &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(config.vocabulary_size())
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(ParallelPlanError::InvalidTensor(
                    "tied decoder output has a separate vocabulary range".into(),
                ));
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidTensor(
                    "untied decoder output has no vocabulary range".into(),
                ));
            }
        }
        let expected = StateLayout::new(
            cache_layout_with_key_value_heads(
                config,
                self.blocks.iter().map(Config::num_key_value_heads),
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        )
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local decoder state layout drifted from block geometry".into(),
            ));
        }
        Ok(())
    }
}

/// Derives one shared decoder's complete rank-local geometry from a typed plan.
pub fn local_geometry<C, F>(
    config: &C,
    layout: &LocalModelLayout,
    mut local_block: F,
) -> Result<LocalGeometry<C>, ParallelPlanError>
where
    C: Config,
    F: FnMut(&C, usize, &LocalModelLayout) -> Result<C, ParallelPlanError>,
{
    let layers = usize::try_from(config.num_hidden_layers())
        .map_err(|_| ParallelPlanError::InvalidGroup("decoder layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|index| local_block(config, index, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let key_value_heads = blocks.iter().map(Config::num_key_value_heads);
    let state_layout = StateLayout::new(
        cache_layout_with_key_value_heads(config, key_value_heads)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let vocabulary = usize::try_from(config.vocabulary_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder vocabulary size exceeds usize".into())
    })?;
    let embedding_name = format!("{}.embed_tokens", config.parameter_root());
    let embedding_range = vocabulary_range(layout, &embedding_name, vocabulary)?;
    let output_range = if config.tie_word_embeddings() {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: config.architecture_fingerprint(),
    };
    geometry.validate_for(config)?;
    Ok(geometry)
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    global_vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected: Option<std::ops::Range<usize>> = None;
    let mut found = false;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        found = true;
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..global_vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "vocabulary member {target} has non-row placement {placement:?}"
                )));
            }
        };
        if tensor.global_shape().first().copied() != Some(global_vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
                tensor.global_shape()
            )));
        }
        if selected.as_ref().is_some_and(|selected| selected != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Declares the complete mutable-state geometry consumed by resident or bounded execution.
pub fn cache_layout_with_key_value_heads<C: Config>(
    config: &C,
    key_value_heads: impl IntoIterator<Item = i32>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    cache_layout_destination(
        config,
        key_value_heads.into_iter(),
        &crate::state_geometry::Ordinary(Error::backend_message),
    )
}

pub(crate) fn cache_layout_destination<C: Config, D: crate::state_geometry::Destination>(
    config: &C,
    key_value_heads: impl Iterator<Item = i32>,
    destination: &D,
) -> Result<LayerSchedule<LayerCachePolicy>, D::Error> {
    destination.controls::<(LayerSchedule<LayerCachePolicy>, &C)>()?;
    let layers = usize::try_from(config.num_hidden_layers())
        .map_err(|cause| destination.error(format_args!("{cause}")))?;
    let key_value_heads = destination.collect_values(key_value_heads)?;
    if key_value_heads.len() != layers {
        return Err(destination.error(format_args!(
            "decoder cache geometry has {} layers, expected {layers}",
            key_value_heads.len(),
        )));
    }
    let policies =
        destination.collect(config.attention_schedule().iter().zip(key_value_heads).map(
            |(attention, heads)| destination.key_value(*attention, heads, config.head_dim()),
        ))?;
    destination.schedule(layers, policies)
}

/// Creates one concrete backend cache per decoder layer from the neutral policy.
///
/// Cache construction is outside inference. The closure is monomorphized and
/// returns the backend's native cache type without boxing or tensor conversion.
pub fn create_caches<C: Config, K>(
    config: &C,
    mut create: impl FnMut(usize, Option<i32>) -> K,
) -> Result<Vec<Option<K>>, Error> {
    validate_schedule(config)?;
    config
        .attention_schedule()
        .iter()
        .enumerate()
        .map(|(layer, policy)| {
            let window = policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(Error::backend)?;
            Ok(Some(create(layer, window)))
        })
        .collect()
}

/// Validates that concrete backend caches implement the architecture's policy.
pub fn validate_caches<B, C, K>(config: &C, caches: &[Option<K>]) -> Result<(), Error>
where
    B: NeuralBackend,
    C: Config,
    K: AttentionCache<B::Tensor>,
{
    validate_schedule(config)?;
    if caches.len() != config.attention_schedule().len() {
        return Err(Error::backend(format!(
            "decoder cache has {} layers, expected {}",
            caches.len(),
            config.attention_schedule().len()
        )));
    }
    for (layer, (cache, policy)) in caches
        .iter()
        .zip(config.attention_schedule().iter())
        .enumerate()
    {
        let cache = cache
            .as_ref()
            .ok_or_else(|| Error::backend(format!("decoder cache is missing layer {layer}")))?;
        let expected = policy
            .window()
            .map(|window| i32::try_from(window.get()))
            .transpose()
            .map_err(Error::backend)?;
        if cache.max_size() != expected {
            return Err(Error::backend(format!(
                "decoder cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}",
                cache.max_size()
            )));
        }
    }
    Ok(())
}

fn validate_schedule<C: Config>(config: &C) -> Result<(), Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    if config.attention_schedule().len() != layers {
        return Err(Error::backend(format!(
            "decoder attention schedule has {} layers, expected {layers}",
            config.attention_schedule().len()
        )));
    }
    Ok(())
}

/// Hidden-state input for one decoder block.
pub struct AttentionInput<'a, T, C> {
    /// Hidden states shaped `[batch, sequence, hidden]`.
    pub hidden: &'a T,
    /// Optional additive or boolean attention mask.
    pub mask: Option<&'a T>,
    /// Optional mutable layer cache.
    pub cache: Option<&'a mut C>,
    /// Whether the block may select its mask-free sliding prefill kernel.
    pub allow_sliding_prefill: bool,
    /// Optional caller-provided explicit rotary position data.
    pub rotary_position: Option<RotaryPosition<'a, T>>,
}

/// Shared grouped-query self attention.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct FusedAttentionProjection<B: NeuralBackend> {
    /// Component-major affine projection.
    pub projection: B::Linear,
    /// Validated query/key/value component geometry.
    #[parameter(skip, metadata)]
    pub layout: FusedProjectionLayout,
}

/// Split or fused physical query/key/value operators feeding one attention path.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum AttentionInputProjection<B: NeuralBackend> {
    /// Q/K projections paired with an architecture-owned routed value provider.
    ExternalValue {
        /// Query projection.
        query: B::Linear,
        /// Key projection.
        key: B::Linear,
    },
    /// Independent projections used by conventional decoder checkpoints.
    Split {
        /// Query projection.
        query: B::Linear,
        /// Key projection.
        key: B::Linear,
        /// Value projection.
        value: B::Linear,
    },
    /// One component-major fused projection.
    Fused(FusedAttentionProjection<B>),
}

/// Shared grouped-query self attention.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend> {
    /// Independent projection of the attention output gate.
    pub output_gate: Option<B::Linear>,
    /// Activation applied to either separately projected or fused-query gates.
    #[parameter(skip, metadata)]
    pub output_gate_activation: OutputGateActivation,
    /// Normalize complete Q/K projections using per-head scale vectors.
    #[parameter(skip, metadata)]
    pub query_key_norm_per_head_weights: bool,
    /// Rotated paired width; zero means the complete head.
    #[parameter(skip, metadata)]
    pub rotary_pair_dimensions: i32,
    /// Number of query heads.
    #[parameter(skip, metadata)]
    pub query_heads: i32,
    /// Number of key/value heads.
    #[parameter(skip, metadata)]
    pub key_value_heads: i32,
    /// Inverse square-root head scaling.
    #[parameter(skip, metadata)]
    pub scale: f32,
    /// Optional score cap applied before masking and softmax.
    #[parameter(skip, metadata)]
    pub softcap: Option<f32>,
    /// Selected score/probability arithmetic.
    #[parameter(skip, metadata)]
    pub arithmetic: eredu_nn::AttentionArithmetic,
    /// Split or fused query/key/value projections.
    pub input_projection: AttentionInputProjection<B>,
    /// Output projection.
    pub output: B::Linear,
    /// Optional learned logit participating in attention softmax for each query head.
    pub sinks: Option<Parameter<B::Tensor>>,
    /// Optional per-head query normalization.
    pub query_norm: Option<B::Normalization>,
    /// Optional per-head key normalization.
    pub key_norm: Option<B::Normalization>,
    /// Optional rotary-position operator for positioned attention families.
    pub rotary: Option<B::Rotary>,
    /// Layer-local sliding window.
    #[parameter(skip, metadata)]
    pub sliding_window: Option<i32>,
    /// Whether the query projection's second half gates attended values.
    #[parameter(skip, metadata)]
    pub query_output_gate: bool,
}

struct ProjectedAttention<T> {
    queries: T,
    keys: T,
    values: T,
    output_gate: Option<T>,
    batch: i32,
    sequence: i32,
}

/// Pointwise gate applied after attention and before the output projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputGateActivation {
    /// Logistic sigmoid.
    Sigmoid,
    /// SiLU.
    Silu,
    /// `softplus(beta * x) / beta` with finite positive beta.
    Softplus(f32),
}

impl OutputGateActivation {
    fn apply<B: NeuralBackend>(
        self,
        input: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Sigmoid => B::sigmoid(input, context),
            Self::Silu => B::silu(input, context),
            Self::Softplus(beta) if beta.is_finite() && beta > 0.0 => {
                B::softplus(input, beta, context)
            }
            Self::Softplus(_) => Err(Error::backend(
                "attention gate beta must be finite and positive",
            )),
        }
    }
}

impl<B: NeuralBackend> Attention<B> {
    /// Assembles grouped-query attention from architecture-named operators.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        query_heads: i32,
        key_value_heads: i32,
        head_dim: i32,
        query: B::Linear,
        key: B::Linear,
        value: B::Linear,
        output: B::Linear,
        query_norm: Option<B::Normalization>,
        key_norm: Option<B::Normalization>,
        rotary: Option<B::Rotary>,
        sliding_window: Option<i32>,
    ) -> Result<Self, Error> {
        Self::from_parts_with_query_gate(
            query_heads,
            key_value_heads,
            head_dim,
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
            rotary,
            sliding_window,
            false,
        )
    }

    /// Assembles grouped-query attention whose fused query projection carries
    /// an equally sized output gate in its second half.
    #[allow(clippy::too_many_arguments)]
    pub fn from_gated_parts(
        query_heads: i32,
        key_value_heads: i32,
        head_dim: i32,
        query: B::Linear,
        key: B::Linear,
        value: B::Linear,
        output: B::Linear,
        query_norm: Option<B::Normalization>,
        key_norm: Option<B::Normalization>,
        rotary: Option<B::Rotary>,
        sliding_window: Option<i32>,
    ) -> Result<Self, Error> {
        Self::from_parts_with_query_gate(
            query_heads,
            key_value_heads,
            head_dim,
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
            rotary,
            sliding_window,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts_with_query_gate(
        query_heads: i32,
        key_value_heads: i32,
        head_dim: i32,
        query: B::Linear,
        key: B::Linear,
        value: B::Linear,
        output: B::Linear,
        query_norm: Option<B::Normalization>,
        key_norm: Option<B::Normalization>,
        rotary: Option<B::Rotary>,
        sliding_window: Option<i32>,
        query_output_gate: bool,
    ) -> Result<Self, Error> {
        if query_heads <= 0
            || key_value_heads <= 0
            || head_dim <= 0
            || query_heads % key_value_heads != 0
            || sliding_window.is_some_and(|window| window <= 0)
        {
            return Err(Error::backend(format!(
                "invalid grouped-query attention geometry q={query_heads} kv={key_value_heads} dim={head_dim} window={sliding_window:?}"
            )));
        }
        Ok(Self {
            output_gate: None,
            output_gate_activation: OutputGateActivation::Sigmoid,
            query_key_norm_per_head_weights: false,
            rotary_pair_dimensions: 0,
            query_heads,
            key_value_heads,
            scale: (head_dim as f32).sqrt().recip(),
            softcap: None,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
            input_projection: AttentionInputProjection::Split { query, key, value },
            output,
            sinks: None,
            query_norm,
            key_norm,
            rotary,
            sliding_window,
            query_output_gate,
        })
    }

    /// Builds unloaded grouped-query attention for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(
            Self,
            LinearSpec,
            NormalizationConstructionSpec,
            RotarySpec,
            [FusedProjectionSegment; 3],
        )>()?;
        if config.learned_attention_sinks() {
            metadata.require::<B>(
                "shared decoder attention sinks",
                eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
            )?;
        }
        let fields = config
            .block_parameter_fields()
            .validate_with(|args| metadata.error(args))?;
        let prefix = metadata.text(format_args!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.attention
        ))?;
        let hidden = config.hidden_size();
        let head = config.head_dim();
        let query_heads = config.num_attention_heads();
        let key_value_heads = config.num_key_value_heads();
        let linear = |field: &str, input, output, bias: bool| {
            let weight_name = metadata.text(format_args!("{prefix}.{field}.weight"))?;
            let bias = bias
                .then(|| {
                    metadata.parameter(
                        config,
                        metadata.text(format_args!("{prefix}.{field}.bias"))?,
                    )
                })
                .transpose()?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: metadata
                        .parameter(config, metadata.text(format_args!("{weight_name}"))?)?,
                    bias,
                    format: metadata
                        .format(&weight_name, metadata.linear_format(config, &weight_name)?)?,
                },
                context,
            )
        };
        let policy = config.attention_schedule().get(layer).ok_or_else(|| {
            metadata.error(format_args!(
                "decoder attention schedule has no policy for layer {layer}"
            ))
        })?;
        let query_width = query_heads.checked_mul(head).ok_or_else(|| {
            metadata.error(format_args!("decoder query projection width overflowed"))
        })?;
        let key_value_width = key_value_heads.checked_mul(head).ok_or_else(|| {
            metadata.error(format_args!(
                "decoder key/value projection width overflowed"
            ))
        })?;
        let input_projection = match config.attention_projection_layout() {
            AttentionProjectionLayout::Split if config.external_attention_value(layer) => {
                AttentionInputProjection::ExternalValue {
                    query: linear(
                        fields.attention_query,
                        hidden,
                        query_width,
                        config.attention_bias(AttentionProjection::Query),
                    )?,
                    key: linear(
                        fields.attention_key,
                        hidden,
                        key_value_width,
                        config.attention_bias(AttentionProjection::Key),
                    )?,
                }
            }
            AttentionProjectionLayout::Split => AttentionInputProjection::Split {
                query: linear(
                    fields.attention_query,
                    hidden,
                    query_width,
                    config.attention_bias(AttentionProjection::Query),
                )?,
                key: linear(
                    fields.attention_key,
                    hidden,
                    key_value_width,
                    config.attention_bias(AttentionProjection::Key),
                )?,
                value: linear(
                    fields.attention_value,
                    hidden,
                    key_value_width,
                    config.attention_bias(AttentionProjection::Value),
                )?,
            },
            AttentionProjectionLayout::Fused { field } => {
                if field.trim().is_empty() {
                    return Err(metadata
                        .error(format_args!("fused QKV projection field must not be empty")));
                }
                let biases = [
                    config.attention_bias(AttentionProjection::Query),
                    config.attention_bias(AttentionProjection::Key),
                    config.attention_bias(AttentionProjection::Value),
                ];
                if biases.iter().any(|bias| *bias != biases[0]) {
                    return Err(metadata.error(format_args!(
                        "fused QKV projection requires identical query/key/value bias policy"
                    )));
                }
                let layout = metadata.fused([
                    metadata.segment("query", query_width)?,
                    metadata.segment("key", key_value_width)?,
                    metadata.segment("value", key_value_width)?,
                ])?;
                let projection = linear(field, hidden, layout.output_width(), biases[0])?;
                AttentionInputProjection::Fused(FusedAttentionProjection { projection, layout })
            }
        };
        Ok(Self {
            output_gate: config
                .attention_output_gate()
                .map(|(field, _)| linear(field, hidden, query_width, false))
                .transpose()?,
            output_gate_activation: config
                .attention_output_gate()
                .map_or(OutputGateActivation::Sigmoid, |(_, activation)| activation),
            query_key_norm_per_head_weights: config.query_key_norm_per_head_weights(),
            rotary_pair_dimensions: config.rotary_pair_dimensions(),
            query_heads,
            key_value_heads,
            scale: config.attention_scale(),
            softcap: config.attention_softcap(),
            arithmetic: config.attention_arithmetic(),
            input_projection,
            output: linear(
                fields.attention_output,
                query_width,
                hidden,
                config.attention_bias(AttentionProjection::Output),
            )?,
            sinks: config
                .learned_attention_sinks()
                .then(|| {
                    Parameter::unloaded(
                        metadata.parameter(
                            config,
                            metadata.text(format_args!("{prefix}.{}", fields.attention_sinks))?,
                        )?,
                        &[query_heads],
                        context,
                    )
                })
                .transpose()?,
            query_norm: config
                .query_key_norm_epsilon()
                .map(|epsilon| {
                    let mut spec = NormalizationConstructionSpec::learned(
                        if config.query_key_norm_per_head_weights() {
                            query_width
                        } else {
                            head
                        },
                        epsilon,
                        metadata.parameter(
                            config,
                            metadata.text(format_args!(
                                "{prefix}.{}.weight",
                                fields.attention_query_norm
                            ))?,
                        )?,
                    );
                    if config.query_key_norm_per_head_weights() {
                        spec = metadata.with_groups(spec, query_heads)?;
                    }
                    B::normalization(spec, context)
                })
                .transpose()?,
            key_norm: config
                .query_key_norm_epsilon()
                .map(|epsilon| {
                    let mut spec = NormalizationConstructionSpec::learned(
                        if config.query_key_norm_per_head_weights() {
                            key_value_width
                        } else {
                            head
                        },
                        epsilon,
                        metadata.parameter(
                            config,
                            metadata.text(format_args!(
                                "{prefix}.{}.weight",
                                fields.attention_key_norm
                            ))?,
                        )?,
                    );
                    if config.query_key_norm_per_head_weights() {
                        spec = metadata.with_groups(spec, key_value_heads)?;
                    }
                    B::normalization(spec, context)
                })
                .transpose()?,
            rotary: config
                .rotary_enabled()
                .then(|| B::rotary(metadata.rotary(config, head)?, context))
                .transpose()?,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(|cause| metadata.error(format_args!("{cause}")))?,
            query_output_gate: false,
        })
    }

    fn projections(
        &mut self,
        hidden: &B::Tensor,
        external_value: Option<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ProjectedAttention<B::Tensor>, Error> {
        if external_value.is_some()
            && !matches!(
                self.input_projection,
                AttentionInputProjection::ExternalValue { .. }
            )
        {
            return Err(Error::backend(
                "externally projected values require an external-value attention layout",
            ));
        }
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let reshape = |tensor: B::Tensor, heads| {
            tensor
                .reshape(&[batch, sequence, heads, -1], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let (mut query, mut key, value) = match &mut self.input_projection {
            AttentionInputProjection::ExternalValue { query, key } => (
                query.forward(hidden, context)?,
                key.forward(hidden, context)?,
                external_value.ok_or_else(|| {
                    Error::backend("attention requires a provider-projected value")
                })?,
            ),
            AttentionInputProjection::Split { query, key, value } => (
                query.forward(hidden, context)?,
                key.forward(hidden, context)?,
                value.forward(hidden, context)?,
            ),
            AttentionInputProjection::Fused(fused) => {
                let projected = fused.projection.forward(hidden, context)?;
                let mut components = fused.layout.split(&projected, context)?.into_iter();
                let query = components.next().ok_or_else(|| {
                    Error::backend("fused QKV projection is missing its query component")
                })?;
                let key = components.next().ok_or_else(|| {
                    Error::backend("fused QKV projection is missing its key component")
                })?;
                let value = components.next().ok_or_else(|| {
                    Error::backend("fused QKV projection is missing its value component")
                })?;
                if components.next().is_some() {
                    return Err(Error::backend(
                        "fused QKV projection produced unexpected extra components",
                    ));
                }
                (query, key, value)
            }
        };
        if matches!(
            self.input_projection,
            AttentionInputProjection::ExternalValue { .. }
        ) && value.shape() != key.shape()
        {
            return Err(Error::backend(format!(
                "provider-projected value shape {:?} differs from key projection {:?}",
                value.shape(),
                key.shape()
            )));
        }
        if self.query_key_norm_per_head_weights {
            if let Some(norm) = &mut self.query_norm {
                query = norm.forward(&query, context)?;
            }
            if let Some(norm) = &mut self.key_norm {
                key = norm.forward(&key, context)?;
            }
        }
        let query = query.reshape(&[batch, sequence, self.query_heads, -1], context)?;
        let (query, output_gate) = if self.query_output_gate {
            let projected = query.dim(3);
            if projected <= 0 || projected % 2 != 0 {
                return Err(Error::backend(format!(
                    "gated query projection has invalid final width {projected}"
                )));
            }
            let head = projected / 2;
            (
                query.index(
                    &[Index::Full, Index::Full, Index::Full, Index::Range(0, head)],
                    context,
                )?,
                Some(
                    query
                        .index(
                            &[
                                Index::Full,
                                Index::Full,
                                Index::Full,
                                Index::Range(head, projected),
                            ],
                            context,
                        )?
                        .reshape(&[batch, sequence, self.query_heads * head], context)?,
                ),
            )
        } else {
            let gate = self
                .output_gate
                .as_mut()
                .map(|gate| gate.forward(hidden, context))
                .transpose()?;
            (query, gate)
        };
        let mut queries = query.transpose_axes(&[0, 2, 1, 3], context)?;
        if !self.query_key_norm_per_head_weights {
            if let Some(norm) = &mut self.query_norm {
                queries = norm.forward(&queries, context)?;
            }
        }
        let mut keys = reshape(key, self.key_value_heads)?;
        if !self.query_key_norm_per_head_weights {
            if let Some(norm) = &mut self.key_norm {
                keys = norm.forward(&keys, context)?;
            }
        }
        let values = reshape(value, self.key_value_heads)?;
        Ok(ProjectedAttention {
            queries,
            keys,
            values,
            output_gate,
            batch,
            sequence,
        })
    }

    fn attend<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        mut cache: Option<&mut C>,
        allow_sliding_prefill: bool,
        rotary_position: Option<RotaryPosition<'_, B::Tensor>>,
        external_value: Option<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let ProjectedAttention {
            queries,
            keys,
            values,
            output_gate,
            batch,
            sequence,
        } = self.projections(hidden, external_value, context)?;
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys) = match &mut self.rotary {
            Some(rotary) => {
                let position = rotary_position.unwrap_or(RotaryPosition::Offset(offset));
                (
                    paired_rotary(
                        rotary,
                        &queries,
                        self.rotary_pair_dimensions,
                        position,
                        context,
                    )?,
                    paired_rotary(
                        rotary,
                        &keys,
                        self.rotary_pair_dimensions,
                        position,
                        context,
                    )?,
                )
            }
            None => (queries, keys),
        };
        let (keys, values) = match cache.as_mut() {
            Some(cache) => cache.update_for_attention(keys, values, context)?,
            None => (keys, values),
        };
        let sinks = self.sinks.as_ref().map(Parameter::as_ref);
        if let Some(window) = self.sliding_window.filter(|_| {
            allow_sliding_prefill
                && sequence > 1
                && !cache.as_ref().is_some_and(|c| c.uses_blockwise_attention())
        }) {
            let attended = B::sliding_window_attention_with_sinks(
                AttentionRequest {
                    arithmetic: self.arithmetic,
                    softcap: self.softcap,
                    queries,
                    keys,
                    values,
                    scale: self.scale,
                    mask,
                    sinks,
                },
                window,
                offset,
                context,
            )?;
            return match output_gate {
                Some(gate) => attended.multiply(
                    &self.output_gate_activation.apply::<B>(gate, context)?,
                    context,
                ),
                None => Ok(attended),
            };
        }
        let request = AttentionRequest {
            arithmetic: self.arithmetic,
            softcap: self.softcap,
            queries,
            keys,
            values,
            scale: self.scale,
            mask,
            sinks,
        };
        let attended = if let Some(cache) = cache {
            cache.attention(request, context)?
        } else {
            B::attention_with_sinks(request, context)?
        };
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)?;
        match output_gate {
            Some(gate) => attended.multiply(
                &self.output_gate_activation.apply::<B>(gate, context)?,
                context,
            ),
            None => Ok(attended),
        }
    }

    /// Executes grouped-query attention and its output projection.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_value(input, None, context)
    }

    /// Executes the actual post-aggregation channel boundary before projection.
    /// Callers own normalization, residual order and the invocation path.
    pub fn forward_instrumented<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            None,
            context,
        )?;
        let attended = instrumentation.apply("attention.channels", attended)?;
        instrumentation.project::<B>(
            "attention.write_input",
            &mut self.output,
            &attended,
            parallel,
            context,
        )
    }

    /// Executes ordinary attention with already mixed provider-projected values.
    /// Values enter the same KV cache exactly once, before attention.
    pub fn forward_with_value<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        value: Option<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            value,
            context,
        )?;
        self.output.forward(&attended, context)
    }

    /// Executes grouped-query attention with a row-parallel output projection.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel_with_value(input, None, parallel, context)
    }

    /// Executes head-local attention with mixed local value rows and reduces
    /// only the output projection; activated value rows are never all-summed.
    pub fn forward_parallel_with_value<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        value: Option<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            value,
            context,
        )?;
        B::row_parallel_linear(&mut self.output, &attended, parallel, context)
    }
}

fn paired_rotary<T: Tensor, R: RotaryOperator<T>>(
    rotary: &mut R,
    input: &T,
    dimensions: i32,
    position: RotaryPosition<'_, T>,
    context: &T::Context,
) -> Result<T, Error> {
    let head = *input
        .shape()
        .last()
        .ok_or_else(|| Error::backend("rotary input is scalar"))?;
    if dimensions == 0 || dimensions == head {
        return rotary.forward_subspace(input, RotarySubspace::Full, position, context);
    }
    if dimensions <= 0 || dimensions > head || dimensions % 2 != 0 || head % 2 != 0 {
        return Err(Error::backend("invalid partial paired rotary geometry"));
    }
    let slice = |value: &T, start, end| {
        let mut index = vec![Index::Full; value.shape().len()];
        *index.last_mut().expect("non-scalar rotary input") = Index::Range(start, end);
        value.index(&index, context)
    };
    let half = head / 2;
    let rotated_half = dimensions / 2;
    let selected = T::concatenate(
        &[
            slice(input, 0, rotated_half)?,
            slice(input, half, half + rotated_half)?,
        ],
        -1,
        context,
    )?;
    let rotated = rotary.forward_subspace(&selected, RotarySubspace::Full, position, context)?;
    T::concatenate(
        &[
            slice(&rotated, 0, rotated_half)?,
            slice(input, rotated_half, half)?,
            slice(&rotated, rotated_half, dimensions)?,
            slice(input, half + rotated_half, head)?,
        ],
        -1,
        context,
    )
}

/// One component-major fused gate/up affine projection.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct FusedGatedProjection<B: NeuralBackend> {
    /// Fused affine operator.
    pub projection: B::Linear,
    /// Validated gate/up component geometry.
    #[parameter(skip, metadata)]
    pub layout: FusedProjectionLayout,
}

/// Split or fused physical gate/up operators feeding one gated product.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum GatedInputProjection<B: NeuralBackend> {
    /// Independent gate and up projections.
    Split {
        /// Gate projection.
        gate: B::Linear,
        /// Up projection.
        up: B::Linear,
    },
    /// One component-major fused projection.
    Fused(FusedGatedProjection<B>),
}

/// Shared dense SwiGLU feed-forward network.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: NeuralBackend> {
    /// Split or fused gate/up input projection.
    pub input_projection: GatedInputProjection<B>,
    /// Down projection.
    pub down: B::Linear,
    /// Optional shared pre-activation bound.
    #[parameter(skip, metadata)]
    pub limit: Option<GatedProductPolicy>,
}

impl<B: NeuralBackend> Mlp<B> {
    /// Assembles a dense gated network from independent gate/up projections.
    pub fn from_parts(
        gate: B::Linear,
        up: B::Linear,
        down: B::Linear,
        policy: Option<GatedProductPolicy>,
    ) -> Self {
        Self {
            input_projection: GatedInputProjection::Split { gate, up },
            down,
            limit: policy,
        }
    }

    /// Builds an unloaded dense SwiGLU network for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, LinearSpec, [FusedProjectionSegment; 2])>()?;
        let fields = config
            .block_parameter_fields()
            .validate_with(|args| metadata.error(args))?;
        let prefix = metadata.text(format_args!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.feed_forward
        ))?;
        let build = |field: &str, input, output| {
            let weight_name = metadata.text(format_args!("{prefix}.{field}.weight"))?;
            let bias = config
                .mlp_bias()
                .then(|| {
                    metadata.parameter(
                        config,
                        metadata.text(format_args!("{prefix}.{field}.bias"))?,
                    )
                })
                .transpose()?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: metadata
                        .parameter(config, metadata.text(format_args!("{weight_name}"))?)?,
                    bias,
                    format: metadata
                        .format(&weight_name, metadata.linear_format(config, &weight_name)?)?,
                },
                context,
            )
        };
        let input_projection = match config.gated_projection_layout() {
            GatedProjectionLayout::Split => GatedInputProjection::Split {
                gate: build(
                    fields.feed_forward_gate,
                    config.hidden_size(),
                    config.intermediate_size(),
                )?,
                up: build(
                    fields.feed_forward_up,
                    config.hidden_size(),
                    config.intermediate_size(),
                )?,
            },
            GatedProjectionLayout::Fused { field } => {
                if field.trim().is_empty() {
                    return Err(metadata.error(format_args!(
                        "fused gate/up projection field must not be empty"
                    )));
                }
                let layout = metadata.fused([
                    metadata.segment("gate", config.intermediate_size())?,
                    metadata.segment("up", config.intermediate_size())?,
                ])?;
                GatedInputProjection::Fused(FusedGatedProjection {
                    projection: build(field, config.hidden_size(), layout.output_width())?,
                    layout,
                })
            }
        };
        Ok(Self {
            input_projection,
            down: build(
                fields.feed_forward_output,
                config.intermediate_size(),
                config.hidden_size(),
            )?,
            limit: config.gated_product_policy(),
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (gate, up) = match &mut self.input_projection {
            GatedInputProjection::Split { gate, up } => {
                (gate.forward(input, context)?, up.forward(input, context)?)
            }
            GatedInputProjection::Fused(fused) => {
                let projected = fused.projection.forward(input, context)?;
                let mut components = fused.layout.split(&projected, context)?.into_iter();
                let gate = components.next().ok_or_else(|| {
                    Error::backend("fused gate/up projection is missing its gate component")
                })?;
                let up = components.next().ok_or_else(|| {
                    Error::backend("fused gate/up projection is missing its up component")
                })?;
                if components.next().is_some() {
                    return Err(Error::backend(
                        "fused gate/up projection produced unexpected extra components",
                    ));
                }
                (gate, up)
            }
        };
        B::gated_product(gate, up, self.limit.unwrap_or_default(), context)
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        self.down.forward(&hidden, context)
    }

    fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        B::row_parallel_linear(&mut self.down, &hidden, parallel, context)
    }
}

/// Replaceable value and feed-forward projections inside the shared residual block.
pub trait DecoderProjectionOperator<B: NeuralBackend>: eredu_nn::Parameterized<B::Tensor> {
    /// Exact dense component hooks of this selected row-local projection worker.
    /// The enclosing block must separately establish causal prefix equivalence.
    /// Custom/routed workers publish only their own actual internal hooks.
    fn append_component_prefill_observations(
        _unit_path: &str,
        _declarations: &mut Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(), Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &str,
            usize,
            &mut Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            String,
            Result<(), Error>,
        )>()?;

        Ok(())
    }

    /// Optionally projects mixed values from normalized attention input. The
    /// result has `[batch, tokens, local_kv_heads * head_width]` geometry.
    fn project_values(
        &mut self,
        _input: &B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        Ok(None)
    }

    /// Preserves observations inside a custom value projection when the caller
    /// uses the direct decoder path without an external expert provider.
    fn project_values_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        _instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, Error> {
        self.project_values(input, context)
    }

    /// Complete residual contribution; routed providers retain their earlier output boundary.
    fn residual_observation(&self) -> &'static str {
        "feed_forward.output"
    }

    /// Executes replicated feed-forward computation.
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
    /// Executes with genuine internal component boundaries when implemented.
    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        _instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_feed_forward(input, context)
    }
}

/// Additive feed-forward execution for tensor-parallel realizations.
pub trait TensorParallelProjectionOperator<B: NeuralBackend>: DecoderProjectionOperator<B> {
    /// Executes tensor-parallel feed-forward computation.
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;

    /// Allows rank-local component boundaries before the output reduction.
    /// The default retains ordinary execution without internal observations;
    /// implementations with scalar components override it. The caller owns
    /// global component coordinates.
    fn forward_feed_forward_parallel_observed(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_feed_forward_parallel(input, parallel, context)
    }
}

impl<B: NeuralBackend> DecoderProjectionOperator<B> for Mlp<B> {
    fn append_component_prefill_observations(
        unit_path: &str,
        declarations: &mut Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(), Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &str,
            usize,
            &mut Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            String,
            Result<(), Error>,
        )>()?;

        // hidden() is the same per-row gate/up product in ordinary and TP;
        // the shared residual block emits its completed down-projection write.
        append_dense_component_prefill_observations(
            declarations,
            &metadata.format(format_args!("{unit_path}.feed_forward"))?,
            metadata_context,
        )?;

        Ok(())
    }
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward(input, context)
    }
    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        let hidden = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &hidden,
            None,
            context,
        )
    }
}

impl<B: NeuralBackend> TensorParallelProjectionOperator<B> for Mlp<B> {
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel(input, parallel, context)
    }

    fn forward_feed_forward_parallel_observed(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        let hidden = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &hidden,
            Some(parallel),
            context,
        )
    }
}

/// One RMS-pre-norm residual decoder block.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: NeuralBackend, F = Mlp<B>> {
    /// Self-attention operator.
    pub self_attention: Attention<B>,
    /// Feed-forward operator.
    pub mlp: F,
    /// Pre-attention RMSNorm.
    pub input_norm: B::Normalization,
    /// Pre-MLP RMSNorm.
    pub post_attention_norm: B::Normalization,
    /// Optional post-attention normalization before residual addition.
    pub attention_output_norm: Option<B::Normalization>,
    /// Optional post-feed-forward normalization before residual addition.
    pub feed_forward_output_norm: Option<B::Normalization>,
    /// Optional normalization of the completed block output.
    pub output_norm: Option<B::Normalization>,
}

impl<B: NeuralBackend> TransformerBlock<B, Mlp<B>> {
    /// Builds an unloaded block for one global layer index.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, NormalizationConstructionSpec)>()?;
        let fields = config
            .block_parameter_fields()
            .validate_with(|args| metadata.error(args))?;
        let norm = |name: String| {
            B::normalization(
                NormalizationConstructionSpec {
                    groups: config.normalization_groups(),
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    scale: normalization_scale(
                        metadata.parameter(config, name)?,
                        config.normalization_offset(),
                    ),
                },
                context,
            )
        };
        Ok(Self {
            attention_output_norm: metadata
                .normalization_name(
                    config,
                    layer,
                    parameter_metadata::NormalizationName::Attention,
                )?
                .map(&norm)
                .transpose()?,
            feed_forward_output_norm: metadata
                .normalization_name(
                    config,
                    layer,
                    parameter_metadata::NormalizationName::FeedForward,
                )?
                .map(&norm)
                .transpose()?,
            output_norm: metadata
                .normalization_name(config, layer, parameter_metadata::NormalizationName::Block)?
                .map(|name| {
                    B::normalization(
                        NormalizationConstructionSpec::learned(
                            config.hidden_size(),
                            config.rms_norm_epsilon(),
                            metadata.parameter(config, name)?,
                        ),
                        context,
                    )
                })
                .transpose()?,
            self_attention: Attention::new(config, layer, context)?,
            mlp: Mlp::new(config, layer, context)?,
            input_norm: norm(metadata.text(format_args!(
                "{}.layers.{layer}.{}.weight",
                config.parameter_root(),
                fields.input_norm
            ))?)?,
            post_attention_norm: norm(metadata.text(format_args!(
                "{}.layers.{layer}.{}.weight",
                config.parameter_root(),
                fields.post_attention_norm
            ))?)?,
        })
    }
}

impl<B, F> TransformerBlock<B, F>
where
    B: NeuralBackend,
    F: DecoderProjectionOperator<B>,
{
    fn normalize_output(
        &mut self,
        hidden: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &mut self.output_norm {
            Some(norm) => norm.forward(&hidden, context),
            None => Ok(hidden),
        }
    }
    /// Executes the shared normalization, attention, cache, and residual lifecycle
    /// with family-owned projection callbacks. The same provider is borrowed
    /// sequentially by value and feed-forward execution.
    fn forward_projections<C, D, V, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        state: &mut D,
        values: V,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        V: FnOnce(
            &mut F,
            &B::Tensor,
            &mut D,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<Option<B::Tensor>, Error>,
        H: FnOnce(
            &mut F,
            &B::Tensor,
            &mut D,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let value = values(&mut self.mlp, &normalized, state, context, instrumentation)?;
        let attention_input = AttentionInput {
            hidden: &normalized,
            mask: input.mask,
            cache: input.cache,
            allow_sliding_prefill: input.allow_sliding_prefill,
            rotary_position: input.rotary_position,
        };
        let attended = self.self_attention.attend(
            attention_input.hidden,
            attention_input.mask,
            attention_input.cache,
            attention_input.allow_sliding_prefill,
            attention_input.rotary_position,
            value,
            context,
        )?;
        let attended = instrumentation.apply("attention.channels", attended)?;
        let attention = instrumentation.project::<B>(
            "attention.write_input",
            &mut self.self_attention.output,
            &attended,
            parallel,
            context,
        )?;
        let attention = instrumentation.apply("attention.write", attention)?;
        let attention = match &mut self.attention_output_norm {
            Some(norm) => norm.forward(&attention, context)?,
            None => attention,
        };
        let attention = instrumentation.apply("attention.output", attention)?;
        let hidden = input.hidden.add(&attention, context)?;
        let hidden = instrumentation.apply("attention.residual", hidden)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let mlp = feed_forward(&mut self.mlp, &normalized, state, context, instrumentation)?;
        let mlp = instrumentation.apply("feed_forward.write", mlp)?;
        let mlp = match &mut self.feed_forward_output_norm {
            Some(norm) => norm.forward(&mlp, context)?,
            None => mlp,
        };
        let mlp = instrumentation.apply(self.mlp.residual_observation(), mlp)?;
        let hidden = instrumentation.apply("feed_forward.residual", hidden.add(&mlp, context)?)?;
        self.normalize_output(hidden, context)
    }

    /// Executes this block with replicated projections.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_projections(
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            &mut (),
            |policy, input, _, context, _| policy.project_values(input, context),
            |policy, input, _, context, instrumentation| {
                policy.forward_feed_forward_observed(input, context, instrumentation)
            },
        )
    }

    /// Executes the ordinary block equations with component capture/intervention.
    pub fn forward_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_projections(
            input,
            None,
            context,
            instrumentation,
            &mut (),
            |policy, input, _, context, instrumentation| {
                policy.project_values_observed(input, context, instrumentation)
            },
            |policy, input, _, context, instrumentation| {
                policy.forward_feed_forward_observed(input, context, instrumentation)
            },
        )
    }

    /// Delegates feed-forward computation while retaining the shared block lifecycle.
    pub fn forward_with_feed_forward<C, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_projections(
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            &mut (),
            |policy, input, _, context, _| policy.project_values(input, context),
            |policy, input, _, context, _| feed_forward(policy, input, context),
        )
    }

    /// Executes column partitions and reduced row projections.
    pub fn forward_tensor_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        F: TensorParallelProjectionOperator<B>,
    {
        self.forward_tensor_parallel_observed(
            input,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the shared block equations with rank-local channel/unit values
    /// and complete residual writes after their ordinary collective reductions.
    /// Global selection and record assembly belong to the partition driver.
    pub fn forward_tensor_parallel_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        F: TensorParallelProjectionOperator<B>,
    {
        self.forward_projections(
            input,
            Some(parallel),
            context,
            instrumentation,
            &mut (),
            |policy, input, _, context, instrumentation| {
                policy.project_values_observed(input, context, instrumentation)
            },
            |policy, input, _, context, instrumentation| {
                policy.forward_feed_forward_parallel_observed(
                    input,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
    }

    /// Delegates feed-forward computation for a tensor-parallel block.
    pub fn forward_tensor_parallel_with_feed_forward<C, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_projections(
            input,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
            &mut (),
            |policy, input, _, context, _| policy.project_values(input, context),
            |policy, input, _, context, _| feed_forward(policy, input, context),
        )
    }

    /// Routes both replaceable projection stages through one runtime provider.
    pub fn forward_routed<C, P>(
        &mut self,
        layer: usize,
        input: AttentionInput<'_, B::Tensor, C>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
        C: AttentionCache<B::Tensor>,
        F: RoutedProjectionOperator<B>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            provider,
            |policy, input, provider, context, _| {
                policy.project_values_with_provider(layer, input, pass, provider, context)
            },
            |policy, input, provider, context, _| {
                policy.forward_with_provider(layer, input, pass, provider, context)
            },
        )
    }

    /// Executes provider-backed projections at their actual component boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_routed_observed<C, P>(
        &mut self,
        layer: usize,
        input: AttentionInput<'_, B::Tensor, C>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
        C: AttentionCache<B::Tensor>,
        F: RoutedProjectionOperator<B>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            input,
            None,
            context,
            instrumentation,
            provider,
            |policy, input, provider, context, instrumentation| {
                policy.project_values_observed_with_provider(
                    layer,
                    input,
                    pass,
                    provider,
                    context,
                    instrumentation,
                    points.clone(),
                )
            },
            |policy, input, provider, context, instrumentation| {
                policy.forward_observed_with_provider(
                    layer,
                    input,
                    pass,
                    provider,
                    context,
                    instrumentation,
                    points.clone(),
                )
            },
        )
    }

    /// Routes both stages with owned value-output rows and reduced feed-forward rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_routed_parallel<C, P>(
        &mut self,
        layer: usize,
        input: AttentionInput<'_, B::Tensor, C>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
        C: AttentionCache<B::Tensor>,
        F: TensorParallelRoutedProjectionOperator<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            input,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
            provider,
            |policy, input, provider, context, _| {
                policy.project_values_with_provider(layer, input, pass, provider, context)
            },
            |policy, input, provider, context, _| {
                policy
                    .forward_parallel_with_provider(layer, input, pass, provider, parallel, context)
            },
        )
    }

    /// Executes provider-backed projections at their actual component boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_routed_parallel_observed<C, P>(
        &mut self,
        layer: usize,
        input: AttentionInput<'_, B::Tensor, C>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
        C: AttentionCache<B::Tensor>,
        F: TensorParallelRoutedProjectionOperator<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            input,
            Some(parallel),
            context,
            instrumentation,
            provider,
            |policy, input, provider, context, instrumentation| {
                policy.project_values_observed_with_provider(
                    layer,
                    input,
                    pass,
                    provider,
                    context,
                    instrumentation,
                    points.clone(),
                )
            },
            |policy, input, provider, context, instrumentation| {
                policy.forward_parallel_observed_with_provider(
                    layer,
                    input,
                    pass,
                    provider,
                    parallel,
                    context,
                    instrumentation,
                    points.clone(),
                )
            },
        )
    }
}

/// Declares attention and normalization groups shared by dense and routed blocks.
pub fn block_common_parallel_parameter_groups<B: NeuralBackend, F>(
    block: &TransformerBlock<B, F>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    block_common_parallel_parameter_groups_with(block, config, layer, DeclarationDestination(None))
        .map_err(ParameterGroupError::ordinary)
}
pub(crate) fn block_common_parallel_parameter_groups_with<B: NeuralBackend, F>(
    block: &TransformerBlock<B, F>,
    config: &impl Config,
    layer: usize,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls::<(
        Vec<ParameterGroupSpec>,
        attention_partition::AttentionPartition,
    )>()?;
    let prefix = destination.text(format_args!("{}.layers.{layer}", config.parameter_root()))?;
    let fields = config
        .block_parameter_fields()
        .validate_with(|args| destination.group_error(args))?;
    let attention_prefix = destination.text(format_args!("{prefix}.{}", fields.attention))?;
    let query_heads = usize::try_from(config.num_attention_heads()).map_err(|_| {
        destination.group_error(format_args!("decoder query-head count exceeds usize"))
    })?;
    let key_value_heads = usize::try_from(config.num_key_value_heads()).map_err(|_| {
        destination.group_error(format_args!("decoder key/value-head count exceeds usize"))
    })?;
    let head_dimension = usize::try_from(config.head_dim()).map_err(|_| {
        destination.group_error(format_args!("decoder head dimension exceeds usize"))
    })?;
    if head_dimension == 0 || key_value_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(destination.group_error(format_args!(
            "decoder attention geometry q={query_heads}, kv={key_value_heads}, dim={head_dimension} does not form positive integral GQA groups"
        )));
    }
    let head_partition =
        attention_partition::AttentionPartition::new_with(config, layer, destination)?;
    let attention_units = head_partition.preferred_units();
    let attention = match &block.self_attention.input_projection {
        AttentionInputProjection::ExternalValue { query, key } => {
            let fixed = [
                (query, ProjectionSharding::Column),
                (key, ProjectionSharding::Column),
                (&block.self_attention.output, ProjectionSharding::Row),
            ];
            let mut projections = destination
                .vector(fixed.len() + usize::from(block.self_attention.output_gate.is_some()))?;
            projections.extend_from_slice(&fixed);
            if let Some(gate) = &block.self_attention.output_gate {
                projections.push((gate, ProjectionSharding::Column));
            }
            destination.projections::<B::Tensor, B::Linear>(
                format_args!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &projections,
                attention_units,
            )?
        }
        AttentionInputProjection::Split { query, key, value } => {
            let fixed = [
                (query, ProjectionSharding::Column),
                (key, ProjectionSharding::Column),
                (value, ProjectionSharding::Column),
                (&block.self_attention.output, ProjectionSharding::Row),
            ];
            let mut projections = destination
                .vector(fixed.len() + usize::from(block.self_attention.output_gate.is_some()))?;
            projections.extend_from_slice(&fixed);
            if let Some(gate) = &block.self_attention.output_gate {
                projections.push((gate, ProjectionSharding::Column));
            }
            destination.projections::<B::Tensor, B::Linear>(
                format_args!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &projections,
                attention_units,
            )?
        }
        AttentionInputProjection::Fused(fused) => destination.segmented::<B::Tensor, B::Linear>(
            format_args!("{attention_prefix}.projections"),
            ParameterRole::AttentionHeads,
            &fused.projection,
            &block.self_attention.output,
            fused_projection_ranges_with(&fused.layout, destination)?,
            attention_units,
        )?,
    };

    let input_norm = destination.module::<B::Tensor, _>(
        format_args!("{prefix}.{}", fields.input_norm),
        ParameterRole::Replicated,
        &block.input_norm,
        |_| Ok(MemberSharding::Replicated),
    )?;
    let post_attention_norm = destination.module::<B::Tensor, _>(
        format_args!("{prefix}.{}", fields.post_attention_norm),
        ParameterRole::Replicated,
        &block.post_attention_norm,
        |_| Ok(MemberSharding::Replicated),
    )?;
    let count = 3
        + usize::from(block.output_norm.is_some())
        + usize::from(block.self_attention.sinks.is_some())
        + usize::from(block.self_attention.query_norm.is_some())
        + usize::from(block.self_attention.key_norm.is_some())
        + usize::from(block.attention_output_norm.is_some())
        + usize::from(block.feed_forward_output_norm.is_some());
    let mut groups = destination.vector(count)?;
    groups.push(attention);
    if let Some(norm) = &block.output_norm {
        let name = destination
            .normalization_name(config, layer, NormalizationName::Block, true)?
            .ok_or_else(|| {
                destination.group_error(format_args!(
                    "block output normalization is absent from configuration"
                ))
            })?;
        groups.push(destination.module::<B::Tensor, _>(
            format_args!("{}", name.trim_end_matches(".weight")),
            ParameterRole::Replicated,
            norm,
            |_| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(sinks) = &block.self_attention.sinks {
        groups.push(destination.partitioned_module::<B::Tensor, _>(
            format_args!("{attention_prefix}.{}", fields.attention_sinks),
            ParameterRole::AttentionHeads,
            query_heads,
            sinks,
            |shape| {
                if shape != [query_heads] {
                    return Err(destination.tensor_error(format_args!(
                        "decoder attention sinks have shape {shape:?}, expected [{query_heads}]"
                    )));
                }
                Ok(MemberSharding::Partitioned { axis: 0 })
            },
        )?);
    }
    for (norm, field, heads) in [
        (
            &block.self_attention.query_norm,
            fields.attention_query_norm,
            query_heads,
        ),
        (
            &block.self_attention.key_norm,
            fields.attention_key_norm,
            key_value_heads,
        ),
    ] {
        if let Some(norm) = norm {
            let name = destination.text(format_args!("{attention_prefix}.{field}"))?;
            groups.push(if config.query_key_norm_per_head_weights() {
                destination.partitioned_module::<B::Tensor, _>(
                    format_args!("{name}"),
                    ParameterRole::AttentionHeads,
                    heads,
                    norm,
                    |_| Ok(MemberSharding::Partitioned { axis: 0 }),
                )?
            } else {
                destination.module::<B::Tensor, _>(
                    format_args!("{name}"),
                    ParameterRole::Replicated,
                    norm,
                    |_| Ok(MemberSharding::Replicated),
                )?
            });
        }
    }
    for (name, norm) in [
        (
            destination.normalization_name(
                config,
                layer,
                NormalizationName::Attention,
                block.attention_output_norm.is_some(),
            )?,
            &block.attention_output_norm,
        ),
        (
            destination.normalization_name(
                config,
                layer,
                NormalizationName::FeedForward,
                block.feed_forward_output_norm.is_some(),
            )?,
            &block.feed_forward_output_norm,
        ),
    ] {
        if let (Some(name), Some(norm)) = (name, norm) {
            groups.push(destination.module::<B::Tensor, _>(
                format_args!("{}", name.trim_end_matches(".weight")),
                ParameterRole::Replicated,
                norm,
                |_| Ok(MemberSharding::Replicated),
            )?);
        }
    }
    groups.extend([input_norm, post_attention_norm]);
    let mut result = destination.vector(groups.len())?;
    for group in groups {
        result.push(head_partition.apply_with(
            group,
            |name| destination.linear_format(config, name),
            destination,
        )?);
    }
    Ok(result)
}

/// Declares the dense SwiGLU placement group shared by dense decoder families.
pub fn dense_mlp_parallel_parameter_group<B: NeuralBackend>(
    mlp: &Mlp<B>,
    config: &impl Config,
    layer: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    dense_mlp_parallel_parameter_group_with(mlp, config, layer, DeclarationDestination(None))
        .map_err(ParameterGroupError::ordinary)
}
pub(crate) fn dense_mlp_parallel_parameter_group_with<B: NeuralBackend>(
    mlp: &Mlp<B>,
    config: &impl Config,
    layer: usize,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    destination.controls::<ParameterGroupSpec>()?;
    let fields = config
        .block_parameter_fields()
        .validate_with(|args| destination.group_error(args))?;
    let prefix = destination.text(format_args!(
        "{}.layers.{layer}.{}",
        config.parameter_root(),
        fields.feed_forward
    ))?;
    let intermediate = usize::try_from(config.intermediate_size()).map_err(|_| {
        destination.group_error(format_args!("decoder feed-forward width exceeds usize"))
    })?;
    let output_format = destination.linear_format(
        config,
        &destination.text(format_args!(
            "{prefix}.{}.weight",
            fields.feed_forward_output
        ))?,
    )?;
    let units = crate::linear_format::input_partition_units_with(
        &prefix,
        intermediate,
        1,
        output_format,
        destination,
    )?;
    let group = match &mlp.input_projection {
        GatedInputProjection::Split { gate, up } => destination
            .projections::<B::Tensor, B::Linear>(
                format_args!("{prefix}.projections"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, ProjectionSharding::Column),
                    (up, ProjectionSharding::Column),
                    (&mlp.down, ProjectionSharding::Row),
                ],
                units,
            ),
        GatedInputProjection::Fused(fused) => destination.segmented::<B::Tensor, B::Linear>(
            format_args!("{prefix}.projections"),
            ParameterRole::FeedForwardIntermediate,
            &fused.projection,
            &mlp.down,
            fused_projection_ranges_with(&fused.layout, destination)?,
            units,
        ),
    }?;
    crate::linear_format::dense_ffn_partition_tail_with(
        group,
        intermediate,
        output_format,
        |name| destination.linear_format(config, name),
        destination,
    )
}

fn fused_projection_ranges_with(
    layout: &FusedProjectionLayout,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<Range<usize>>, ParameterGroupError> {
    let mut ranges = destination.vector(layout.segments().len())?;
    let mut start = 0usize;
    for segment in layout.segments() {
        let width = usize::try_from(segment.width()).map_err(|_| {
            destination.tensor_error(format_args!(
                "fused projection segment {} exceeds usize",
                segment.name()
            ))
        })?;
        let end = start.checked_add(width).ok_or_else(|| {
            destination.tensor_error(format_args!(
                "fused projection segment ranges overflowed usize"
            ))
        })?;
        ranges.push(start..end);
        start = end;
    }
    Ok(ranges)
}

/// Declares every rank-local placement group for one dense shared decoder block.
pub fn layer_parallel_parameter_groups<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    layer_parallel_parameter_groups_with(block, config, layer, DeclarationDestination(None))
        .map_err(ParameterGroupError::ordinary)
}
pub(crate) fn layer_parallel_parameter_groups_with_metadata<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<Vec<ParameterGroupSpec>, Error> {
    layer_parallel_parameter_groups_with(
        block,
        config,
        layer,
        DeclarationDestination(Some(context)),
    )
    .map_err(ParameterGroupError::into_neural)
}
fn layer_parallel_parameter_groups_with<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    let mut groups =
        block_common_parallel_parameter_groups_with(block, config, layer, destination)?;
    let group = dense_mlp_parallel_parameter_group_with(&block.mlp, config, layer, destination)?;
    destination.reserve(&mut groups, 1)?;
    groups.push(group);
    Ok(groups)
}

/// Derives the complete dense-decoder parameter topology from normalized
/// configuration without constructing backend modules.
pub fn dense_parameter_description(
    config: &impl Config,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    parameter_description(config, |_| true, |_| Ok(None))
}

/// Declares a split-projection decoder with optional provider-owned values and
/// replacement feed-forward groups, without constructing backend tensors.
pub(crate) fn parameter_description(
    config: &impl Config,
    ordinary_value_projection: impl Fn(usize) -> bool,
    replacement_groups: impl Fn(usize) -> Result<Option<Vec<ParameterGroupSpec>>, ParallelPlanError>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    config
        .validate_config()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if !matches!(
        config.attention_projection_layout(),
        AttentionProjectionLayout::Split
    ) || !matches!(
        config.gated_projection_layout(),
        GatedProjectionLayout::Split
    ) || config.learned_attention_sinks()
    {
        return Err(ParallelPlanError::InvalidGroup(
            "static dense parameter description requires split projections without attention sinks"
                .into(),
        ));
    }
    let dimension = |name: &str, value: i32| {
        usize::try_from(value)
            .map_err(|_| ParallelPlanError::InvalidTensor(format!("decoder {name} exceeds usize")))
    };
    let hidden = dimension("hidden width", config.hidden_size())?;
    let vocabulary = dimension("vocabulary", config.vocabulary_size())?;
    let layers = dimension("layer count", config.num_hidden_layers())?;
    let query_heads = dimension("query-head count", config.num_attention_heads())?;
    let key_value_heads = dimension("key/value-head count", config.num_key_value_heads())?;
    let head = dimension("head width", config.head_dim())?;
    let intermediate = dimension("feed-forward width", config.intermediate_size())?;
    let query_width = query_heads.checked_mul(head).ok_or_else(|| {
        ParallelPlanError::InvalidTensor("decoder query width overflowed usize".into())
    })?;
    let key_value_width = key_value_heads.checked_mul(head).ok_or_else(|| {
        ParallelPlanError::InvalidTensor("decoder key/value width overflowed usize".into())
    })?;
    if head == 0 || key_value_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(ParallelPlanError::InvalidGroup(
            "decoder attention geometry does not form positive integral GQA groups".into(),
        ));
    }

    let graph = ExecutionGraph::chain([TEXT_DECODER_EXECUTION_GROUP])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [layers])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let group_id = layout
        .group_id(0)
        .expect("dense decoder layout contains its text group")
        .clone();
    let root = config.parameter_root();
    let embedding = format!("{root}.embed_tokens.weight");
    let norm = format!("{root}.norm.weight");
    let mut owned = vec![
        OwnedParameterGroupSpec::new(
            if config.tie_word_embeddings() {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role("embedding")
            },
            ParameterGroupSpec::new(
                format!("{root}.embed_tokens"),
                ParameterRole::Vocabulary,
                [eredu_runtime::ParameterMemberSpec::new(
                    &embedding,
                    vec![vocabulary, hidden],
                    MemberSharding::Balanced { axis: 0 },
                )],
            )?,
        ),
        OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("norm"),
            ParameterGroupSpec::new(
                format!("{root}.norm"),
                ParameterRole::Replicated,
                [eredu_runtime::ParameterMemberSpec::new(
                    norm,
                    vec![hidden],
                    MemberSharding::Replicated,
                )],
            )?,
        ),
    ];
    if !config.tie_word_embeddings() {
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("output"),
            ParameterGroupSpec::new(
                "lm_head",
                ParameterRole::Vocabulary,
                [eredu_runtime::ParameterMemberSpec::new(
                    "lm_head.weight",
                    vec![vocabulary, hidden],
                    MemberSharding::Balanced { axis: 0 },
                )],
            )?,
        ));
    }

    for layer in 0..layers {
        let prefix = format!("{root}.layers.{layer}");
        let attention_prefix = format!("{prefix}.{}", fields.attention);
        let head_partition = attention_partition::AttentionPartition::new(config, layer)?;
        let attention_units = head_partition.preferred_units();
        let mut attention_members = Vec::new();
        let mut projection =
            |field: &str, shape: Vec<usize>, sharding: MemberSharding, bias: bool| {
                attention_members.push(eredu_runtime::ParameterMemberSpec::new(
                    format!("{attention_prefix}.{field}.weight"),
                    shape,
                    sharding.clone(),
                ));
                if bias {
                    let bias_sharding =
                        if matches!(sharding, MemberSharding::Partitioned { axis: 0 }) {
                            sharding
                        } else {
                            MemberSharding::Replicated
                        };
                    attention_members.push(eredu_runtime::ParameterMemberSpec::new(
                        format!("{attention_prefix}.{field}.bias"),
                        vec![if matches!(bias_sharding, MemberSharding::Replicated) {
                            hidden
                        } else if field == fields.attention_query {
                            query_width
                        } else {
                            key_value_width
                        }],
                        bias_sharding,
                    ));
                }
            };
        projection(
            fields.attention_query,
            vec![query_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
            config.attention_bias(AttentionProjection::Query),
        );
        projection(
            fields.attention_key,
            vec![key_value_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
            config.attention_bias(AttentionProjection::Key),
        );
        if ordinary_value_projection(layer) {
            projection(
                fields.attention_value,
                vec![key_value_width, hidden],
                MemberSharding::Partitioned { axis: 0 },
                config.attention_bias(AttentionProjection::Value),
            );
        }
        projection(
            fields.attention_output,
            vec![hidden, query_width],
            MemberSharding::Partitioned { axis: 1 },
            config.attention_bias(AttentionProjection::Output),
        );
        if let Some((field, _)) = config.attention_output_gate() {
            projection(
                field,
                vec![query_width, hidden],
                MemberSharding::Partitioned { axis: 0 },
                false,
            );
        }
        let attention = ParameterGroupSpec::partitioned(
            format!("{attention_prefix}.projections"),
            ParameterRole::AttentionHeads,
            attention_units,
            attention_members,
        )?;

        let feed_forward_prefix = format!("{prefix}.{}", fields.feed_forward);
        let feed_forward_units = crate::linear_format::input_partition_units(
            &feed_forward_prefix,
            intermediate,
            1,
            config.linear_format(&format!(
                "{feed_forward_prefix}.{}.weight",
                fields.feed_forward_output
            )),
        )?;
        let mut feed_forward_members = Vec::new();
        for field in [fields.feed_forward_gate, fields.feed_forward_up] {
            feed_forward_members.push(eredu_runtime::ParameterMemberSpec::new(
                format!("{feed_forward_prefix}.{field}.weight"),
                vec![intermediate, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ));
            if config.mlp_bias() {
                feed_forward_members.push(eredu_runtime::ParameterMemberSpec::new(
                    format!("{feed_forward_prefix}.{field}.bias"),
                    vec![intermediate],
                    MemberSharding::Partitioned { axis: 0 },
                ));
            }
        }
        feed_forward_members.push(eredu_runtime::ParameterMemberSpec::new(
            format!(
                "{feed_forward_prefix}.{}.weight",
                fields.feed_forward_output
            ),
            vec![hidden, intermediate],
            MemberSharding::Partitioned { axis: 1 },
        ));
        if config.mlp_bias() {
            feed_forward_members.push(eredu_runtime::ParameterMemberSpec::new(
                format!("{feed_forward_prefix}.{}.bias", fields.feed_forward_output),
                vec![hidden],
                MemberSharding::Replicated,
            ));
        }
        let feed_forward = ParameterGroupSpec::partitioned(
            format!("{feed_forward_prefix}.projections"),
            ParameterRole::FeedForwardIntermediate,
            feed_forward_units,
            feed_forward_members,
        )?;
        let feed_forward = crate::linear_format::dense_ffn_partition_tail(
            feed_forward,
            intermediate,
            config.linear_format(&format!(
                "{feed_forward_prefix}.{}.weight",
                fields.feed_forward_output
            )),
            |name| config.linear_format(name),
        )?;
        let input_norm = ParameterGroupSpec::new(
            format!("{prefix}.{}", fields.input_norm),
            ParameterRole::Replicated,
            [eredu_runtime::ParameterMemberSpec::new(
                format!("{prefix}.{}.weight", fields.input_norm),
                vec![hidden],
                MemberSharding::Replicated,
            )],
        )?;
        let post_attention_norm = ParameterGroupSpec::new(
            format!("{prefix}.{}", fields.post_attention_norm),
            ParameterRole::Replicated,
            [eredu_runtime::ParameterMemberSpec::new(
                format!("{prefix}.{}.weight", fields.post_attention_norm),
                vec![hidden],
                MemberSharding::Replicated,
            )],
        )?;
        let mut unit_groups = vec![attention];
        for name in [
            config.block_output_normalization(layer),
            config.attention_output_normalization(layer),
            config.feed_forward_output_normalization(layer),
        ]
        .into_iter()
        .flatten()
        {
            unit_groups.push(ParameterGroupSpec::new(
                name.trim_end_matches(".weight"),
                ParameterRole::Replicated,
                [eredu_runtime::ParameterMemberSpec::new(
                    &name,
                    vec![hidden],
                    MemberSharding::Replicated,
                )],
            )?);
        }
        if config.query_key_norm_epsilon().is_some() {
            for (field, heads) in [
                (fields.attention_query_norm, query_heads),
                (fields.attention_key_norm, key_value_heads),
            ] {
                let name = format!("{attention_prefix}.{field}");
                let owned = config.query_key_norm_per_head_weights();
                let members = [eredu_runtime::ParameterMemberSpec::new(
                    format!("{name}.weight"),
                    vec![if owned { heads * head } else { head }],
                    if owned {
                        MemberSharding::Partitioned { axis: 0 }
                    } else {
                        MemberSharding::Replicated
                    },
                )];
                unit_groups.push(if owned {
                    ParameterGroupSpec::partitioned(
                        name,
                        ParameterRole::AttentionHeads,
                        heads,
                        members,
                    )?
                } else {
                    ParameterGroupSpec::new(name, ParameterRole::Replicated, members)?
                });
            }
        }
        unit_groups.extend([input_norm, post_attention_norm]);
        if let Some(groups) = replacement_groups(layer)? {
            unit_groups.extend(groups);
        } else {
            unit_groups.push(feed_forward);
        }
        for group in unit_groups {
            let group = head_partition.apply(group, |name| config.linear_format(name))?;
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(group_id.clone(), layer),
                group,
            ));
        }
    }

    let mut expanded = Vec::with_capacity(owned.len());
    for tagged in owned {
        let owner = tagged.owner().clone();
        let [group] = eredu_runtime::expand_linear_format_parameter_groups(
            vec![tagged.into_group()],
            |member| {
                crate::linear_format::standard_parallel_linear_format(
                    member,
                    config.linear_format(member.target()),
                )
            },
        )?
        .try_into()
        .expect("one parameter group expands to one parameter group");
        expanded.push(OwnedParameterGroupSpec::new(owner, group));
    }
    let expected = expanded
        .iter()
        .map(|tagged| tagged.group().clone())
        .collect::<Vec<_>>();
    ArchitectureParameterDescription::new(&graph, &layout, expected, expanded)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

/// Derives the rank-local construction geometry of one tensor-parallel block
/// from the neutral placement layout.
pub fn static_parallel_parameter_groups<B: NeuralBackend>(
    embeddings: &B::Embedding,
    norm: &B::Normalization,
    head: Option<&B::Linear>,
    parameter_root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    parameter_metadata::static_groups::<B>(embeddings, norm, head, parameter_root, None)
        .map_err(parameter_metadata::ParameterGroupError::ordinary)
}

/// Pinned modules shared by resident and bounded-residency execution.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: NeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Final RMSNorm.
    pub norm: B::Normalization,
    /// Optional untied vocabulary projection.
    pub lm_head: Option<B::Linear>,
}

impl<B: NeuralBackend> Clone for StaticModules<B> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
        }
    }
}

/// Architecture-supplied identities and geometry for the shared pinned text
/// modules.
#[derive(Debug, Clone)]
pub struct StaticModuleSpec {
    /// Independent final normalization groups using float32 scale multiplication.
    pub normalization_groups: Option<i32>,
    /// Token embedding parameter identity.
    pub embedding_weight: String,
    /// Final normalization parameter identity.
    pub normalization_weight: String,
    /// Untied output-head parameter identity.
    pub head_weight: String,
    /// Vocabulary row count.
    pub vocabulary: i32,
    /// Hidden width.
    pub hidden_size: i32,
    /// Final RMS normalization epsilon.
    pub normalization_epsilon: f32,
    /// Fixed scalar added to the learned final-normalization scale.
    pub normalization_offset: f32,
    /// Packed embedding format, when supported by the general embedding operator.
    pub embedding_quantization: Option<WeightQuantization>,
    /// Complete output-head physical encoding.
    pub head_format: LinearFormat,
    /// Whether output logits reuse the embedding table.
    pub tied_head: bool,
}

impl<B: eredu_nn::DistributedNeuralBackend> StaticModules<B> {
    /// Projects an already normalized value through its actual tied or untied
    /// head, retaining the vocabulary collective and projection-input boundary.
    pub(crate) fn project_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let logits = match (&mut self.lm_head, parallel) {
            (Some(head), Some(parallel)) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                hidden,
                parallel,
                context,
            ),
            (None, Some(parallel)) => instrumentation.project_vocabulary_embedding::<B>(
                "projection_input",
                &mut self.embeddings,
                hidden,
                parallel,
                context,
            ),
            (Some(head), None) => {
                instrumentation.project::<B>("projection_input", head, hidden, None, context)
            }
            (None, None) => instrumentation.project_embedding(
                "projection_input",
                &mut self.embeddings,
                hidden,
                context,
            ),
        }?;
        instrumentation.apply("linear", logits)
    }

    pub(crate) fn finish_parallel_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = instrumentation.normalize_readout(hidden, &mut self.norm, context)?;
        self.project_instrumented(&hidden, Some(parallel), context, instrumentation)
    }
}

impl<B: NeuralBackend> StaticModules<B> {
    /// Applies the actual residual, normalization and affine readout boundaries.
    pub(crate) fn finish_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = instrumentation.normalize_readout(hidden, &mut self.norm, context)?;
        let logits = match &mut self.lm_head {
            Some(head) => {
                instrumentation.project::<B>("projection_input", head, &hidden, None, context)
            }
            None => instrumentation.project_embedding(
                "projection_input",
                &mut self.embeddings,
                &hidden,
                context,
            ),
        }?;
        let logits = instrumentation.apply("linear", logits)?;
        Ok(logits)
    }

    /// Builds unloaded pinned modules from architecture-owned parameter
    /// identities and physical formats.
    pub fn from_spec(
        spec: StaticModuleSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        static_construction::ordinary::<B>(&spec, context)
    }

    /// Builds the same pinned modules with planner-derived vocabulary ownership.
    pub fn from_parallel_spec(
        spec: StaticModuleSpec,
        embedding_range: VocabularyParallelRange,
        output_range: Option<VocabularyParallelRange>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        static_construction::parallel::<B>(&spec, &embedding_range, output_range.as_ref(), context)
    }

    /// Builds unloaded pinned modules for a decoder family.
    pub fn new<C: Config>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(
            static_construction::from_config::<B, C>(config, context)?,
            context,
        )
    }

    /// Builds rank-local pinned modules for a decoder family.
    pub fn new_parallel<C: Config>(
        config: &C,
        geometry: &LocalGeometry<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        Self::from_parallel_spec(
            static_construction::from_config::<B, C>(config, context)?,
            geometry.embedding_range().clone(),
            geometry.output_range().cloned(),
            context,
        )
    }
}

/// Borrowed token input for the shared layered lifecycle.
pub struct LayeredInput<'a, T> {
    /// Token ids shaped `[batch, sequence]`.
    pub tokens: &'a T,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a T>,
}

/// Shared declaration and validation for one ordered decoder execution group.
#[derive(Debug, Clone)]
pub struct SequentialGroup {
    name: &'static str,
    parameter_root: &'static str,
    units: usize,
}

/// Shared declaration for a target group followed by zero or more ordered
/// prediction groups.
#[derive(Debug, Clone)]
pub struct SequentialPredictionGroups {
    target: SequentialGroup,
    prediction_paths: Vec<Vec<String>>,
    execution_graph: eredu_runtime::ExecutionGraph,
}

impl SequentialPredictionGroups {
    /// Creates the target group and `mtp.{depth}` prediction groups.
    pub fn new(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_roots: impl IntoIterator<Item = String>,
    ) -> Result<Self, Error> {
        let prediction_paths: Vec<Vec<String>> = prediction_roots
            .into_iter()
            .map(|root| vec![root])
            .collect();
        let execution_graph = Self::build_execution_graph(prediction_paths.len(), None)?;
        Ok(Self {
            target: SequentialGroup::new(
                TARGET_EXECUTION_GROUP,
                target_parameter_root,
                target_units,
            )?,
            prediction_paths,
            execution_graph,
        })
    }

    /// Creates equally sized appended prediction groups over one physical namespace.
    pub fn new_pattern(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_parameter_root: &'static str,
        prediction_groups: usize,
        units_per_group: usize,
    ) -> Result<Self, Error> {
        Self::new_pattern_with_metadata(
            target_parameter_root,
            target_units,
            prediction_parameter_root,
            prediction_groups,
            units_per_group,
            None,
        )
    }

    pub(crate) fn new_pattern_with_metadata(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_parameter_root: &'static str,
        prediction_groups: usize,
        units_per_group: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self, Error> {
        Self::new_indexed_with_metadata(
            target_parameter_root,
            target_units,
            prediction_parameter_root,
            0,
            prediction_groups,
            units_per_group,
            context,
        )
    }

    /// Uses the same physical namespace worker with an explicit initial index.
    pub(crate) fn new_indexed_with_metadata(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_parameter_root: &'static str,
        first_physical: usize,
        prediction_groups: usize,
        units_per_group: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self, Error> {
        let metadata = identity::Metadata::new(context);
        metadata.controls::<(Self, Vec<Vec<String>>, Vec<String>)>()?;
        if (prediction_groups != 0 && units_per_group == 0) || prediction_parameter_root.is_empty()
        {
            return Err(metadata.error(format_args!(
                "prediction execution groups require non-empty names and units",
            )));
        }
        let mut prediction_paths = metadata.vector(prediction_groups)?;
        for group in 0..prediction_groups {
            let start = group
                .checked_mul(units_per_group)
                .and_then(|offset| first_physical.checked_add(offset))
                .ok_or_else(|| {
                    metadata.error(format_args!("prediction physical index overflowed"))
                })?;
            let mut paths = metadata.vector(units_per_group)?;
            for unit in 0..units_per_group {
                let physical = start.checked_add(unit).ok_or_else(|| {
                    metadata.error(format_args!("prediction physical index overflowed"))
                })?;
                paths
                    .push(metadata.format(format_args!("{prediction_parameter_root}.{physical}"))?);
            }
            prediction_paths.push(paths);
        }
        let target = match metadata.context() {
            Some(context) => SequentialGroup::new_with_metadata(
                TARGET_EXECUTION_GROUP,
                target_parameter_root,
                target_units,
                context,
            )?,
            None => {
                SequentialGroup::new(TARGET_EXECUTION_GROUP, target_parameter_root, target_units)?
            }
        };
        Ok(Self {
            target,
            execution_graph: Self::build_execution_graph(prediction_paths.len(), context)?,
            prediction_paths,
        })
    }

    /// Borrows the validated target-to-prediction chain retained at construction.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.execution_graph,
        ))
    }

    fn build_execution_graph(
        predictions: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ExecutionGraph, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<eredu_runtime::ExecutionGraph>()?;
        let count = predictions
            .checked_add(1)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let mut groups = destination.vector(count)?;
        groups.push(destination.group(TARGET_EXECUTION_GROUP, &[])?);
        let mut output = destination.text(format_args!("{TARGET_EXECUTION_GROUP}"))?;
        for depth in 0..predictions {
            let id = destination.text(format_args!("mtp.{depth}"))?;
            groups.push(destination.group(&id, &[&output])?);
            output = id;
        }
        destination.finish(groups, &output)
    }

    /// Returns stable prediction-group identities in prediction-depth order.
    pub fn prediction_execution_groups(&self) -> Vec<String> {
        (0..self.prediction_paths.len())
            .map(|depth| format!("mtp.{depth}"))
            .collect()
    }

    /// Returns the number of units in one group.
    pub fn unit_count(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if group == 0 {
            self.target.unit_count(0, metadata_context)
        } else if group <= self.prediction_paths.len() {
            Ok(self.prediction_paths[group - 1].len())
        } else {
            Err(metadata.error(format_args!(
                "execution group {group} is outside target plus {} prediction groups",
                self.prediction_paths.len()
            )))
        }
    }

    /// Returns one stable target or prediction unit path.
    pub fn unit_path(
        &self,
        group: usize,
        index: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            usize,
            usize,
            Option<&eredu_nn::workspace::WorkspaceContext>,
        )>()?;

        if group == 0 {
            return self.target.unit_path(0, index, metadata_context);
        }
        self.unit_count(group, metadata_context)?;
        let path = &self.prediction_paths[group - 1].get(index).ok_or_else(|| {
            metadata.error(format_args!(
                "unit {index} is outside {} units in prediction group {group}",
                self.prediction_paths[group - 1].len()
            ))
        })?;
        metadata.text(format_args!("{path}"))
    }

    /// Selects the activation carried into a ready chain group.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group, None)?;
        if group == 0 {
            return self.target.begin(0, initial, dependencies);
        }
        match dependencies {
            [dependency] => Ok((*dependency).clone()),
            _ => Err(Error::backend(format!(
                "prediction group {group} expected one dependency, received {}",
                dependencies.len()
            ))),
        }
    }

    /// Returns the number of appended prediction groups.
    pub fn prediction_count(&self) -> usize {
        self.prediction_paths.len()
    }
}

impl SequentialGroup {
    /// Creates one non-empty ordered group.
    pub fn new(
        name: &'static str,
        parameter_root: &'static str,
        units: usize,
    ) -> Result<Self, Error> {
        Self::new_with(name, parameter_root, units, |args| Error::backend(args))
    }

    pub(crate) fn new_with_metadata(
        name: &'static str,
        parameter_root: &'static str,
        units: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(size_of::<Self>() + size_of::<Result<Self, Error>>())?;
        Self::new_with(name, parameter_root, units, |args| {
            context.metadata_error(args)
        })
    }

    fn new_with(
        name: &'static str,
        parameter_root: &'static str,
        units: usize,
        invalid: impl FnOnce(std::fmt::Arguments<'_>) -> Error,
    ) -> Result<Self, Error> {
        if name.trim().is_empty() || parameter_root.is_empty() || units == 0 {
            return Err(invalid(format_args!(
                "sequential decoder group requires non-empty names and units",
            )));
        }
        Ok(Self {
            name,
            parameter_root,
            units,
        })
    }

    /// Loans the validated single-group declaration without allocating a graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        eredu_runtime::ArchitectureExecutionGraph::single(self.name).map_err(Error::backend)
    }

    /// Validates the group ordinal and returns its unit count.
    pub fn unit_count(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if group != 0 {
            return Err(metadata.error(format_args!(
                "execution group {group} is outside {}",
                self.name
            )));
        }
        Ok(self.units)
    }

    /// Returns one validated stable unit path.
    pub fn unit_path(
        &self,
        group: usize,
        index: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            usize,
            usize,
            Option<&eredu_nn::workspace::WorkspaceContext>,
        )>()?;

        let count = self.unit_count(group, metadata_context)?;
        if index >= count {
            return Err(metadata.error(format_args!(
                "unit {index} is outside {count} {} units",
                self.name
            )));
        }
        metadata.text(format_args!("{}.{index}", self.parameter_root))
    }

    /// Starts the sole group from the initial activation.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group, None)?;
        if !dependencies.is_empty() {
            return Err(Error::backend(format!(
                "{} received {} unexpected dependencies",
                self.name,
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }
}

/// Architecture-owned values retained across one layered forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    allow_sliding_prefill: bool,
    rotary_embeddings: Option<(T, T)>,
    // Same original Context, retained after the actual transient tensor roots.
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}

/// Statically dispatched construction policy for one decoder block family.
pub trait BlockFactory<B: NeuralBackend, C: Config>: 'static {
    /// This factory's complete shared block equations preserve causal ordinary
    /// request rows. Unknown custom factories must declare their own semantics;
    /// implementing the outer decoder interface is not a causal proof.
    const CAUSAL_PREFILL_ROWS: bool = false;

    /// Architecture-selected feed-forward policy inside the shared block.
    type FeedForward: DecoderProjectionOperator<B>;

    /// Exact residual hook of the configured projection worker. Dynamic
    /// factories use the same branch as construction and execution.
    fn feed_forward_residual_observation(_config: &C, _layer: usize) -> &'static str {
        "feed_forward.output"
    }

    /// Delegate exact internal row declarations to the actual selected worker.
    /// Dynamic factories override this using the same configuration branch as
    /// their construction; a tensor axis or matching path is not a proof.
    fn append_component_prefill_observations(
        _config: &C,
        unit_path: &str,
        _layer: usize,
        declarations: &mut Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(), Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &str,
            usize,
            &mut Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            String,
            Result<(), Error>,
        )>()?;

        <Self::FeedForward as DecoderProjectionOperator<B>>::append_component_prefill_observations(
            unit_path,
            declarations,
            metadata_context,
        )?;

        Ok(())
    }

    /// Validates configuration requirements specific to this block policy.
    fn validate(config: &C) -> Result<(), Error> {
        let _ = config;
        Ok(())
    }

    /// Checked counterpart of this factory's own validation policy.
    /// The ordinary default does not qualify a custom metadata constructor.
    fn validate_with_metadata(
        config: &C,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), Error> {
        if context.uses_checked_metadata() {
            Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
        } else {
            Self::validate(config)
        }
    }

    /// Builds one unloaded decoder block.
    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error>;

    /// Builds one partition-local block while retaining global family policy.
    ///
    /// Dense families use only the localized configuration. Routed families
    /// override this hook so the router keeps global expert cardinality while
    /// the grouped bank uses owner-local EP and TP geometry.
    fn build_partitioned(
        global: &C,
        local: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error> {
        let _ = global;
        Self::build(local, layer, context)
    }

    /// Declares the complete neutral parameter placement for one built block.
    fn parameter_groups(
        block: &TransformerBlock<B, Self::FeedForward>,
        config: &C,
        layer: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>;
    /// Optional checked producer for this factory's exact parameter groups.
    /// Absence is not evidence that its ordinary constructor is metadata bounded.
    fn parameter_groups_with_metadata(
        _block: &TransformerBlock<B, Self::FeedForward>,
        _config: &C,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Option<Result<Vec<ParameterGroupSpec>, Error>> {
        None
    }
}

/// Feed-forward policy that can delegate routed experts to runtime residency.
pub trait RoutedProjectionOperator<B: GroupedNeuralBackend>: DecoderProjectionOperator<B> {
    /// All declared scalar components are emitted by the provider-aware methods.
    const COMPONENT_OBSERVATIONS: bool = false;

    /// Executes the optional value projection through the same provider as the
    /// feed-forward bank. Independent banks retain distinct request identities.
    fn project_values_with_provider<P>(
        &mut self,
        _layer: usize,
        _input: &B::Tensor,
        _pass: eredu_runtime::ExpertPass,
        _provider: &mut P,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        Ok(None)
    }

    /// Executes replicated dense or provider-backed routed work.
    fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display;
    /// Preserves actual provider observations while allowing scalar component instrumentation.
    #[allow(clippy::too_many_arguments)]
    fn project_values_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<Option<B::Tensor>, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (points, instrumentation.observer()) {
            (Some(points), Some(observer)) => {
                eredu_runtime::ObservedExpertProvider::new(provider, observer, points)
                    .execute_neural(|provider| {
                        self.project_values_with_provider(layer, input, pass, provider, context)
                    })
            }
            _ => self.project_values_with_provider(layer, input, pass, provider, context),
        }
    }

    /// Preserves actual provider observations while allowing scalar component instrumentation.
    #[allow(clippy::too_many_arguments)]
    fn forward_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (points, instrumentation.observer()) {
            (Some(points), Some(observer)) => {
                eredu_runtime::ObservedExpertProvider::new(provider, observer, points)
                    .execute_neural(|provider| {
                        self.forward_with_provider(layer, input, pass, provider, context)
                    })
            }
            _ => self.forward_with_provider(layer, input, pass, provider, context),
        }
    }
}

/// Additive routed feed-forward execution for tensor-parallel realizations.
pub trait TensorParallelRoutedProjectionOperator<B: GroupedNeuralBackend>:
    RoutedProjectionOperator<B> + TensorParallelProjectionOperator<B>
{
    /// Executes tensor-parallel dense or provider-backed routed work.
    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display;
    /// Preserves actual provider observations while allowing scalar component instrumentation.
    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (points, instrumentation.observer()) {
            (Some(points), Some(observer)) => {
                eredu_runtime::ObservedExpertProvider::new(provider, observer, points)
                    .execute_neural(|provider| {
                        self.forward_parallel_with_provider(
                            layer, input, pass, provider, parallel, context,
                        )
                    })
            }
            _ => {
                self.forward_parallel_with_provider(layer, input, pass, provider, parallel, context)
            }
        }
    }
}

/// Dense SwiGLU block factory used by Llama and other all-dense decoders.
pub struct DenseBlockFactory;

impl<B: NeuralBackend, C: Config> BlockFactory<B, C> for DenseBlockFactory {
    // Causal attention and row-local normalization/SwiGLU/residual equations.
    const CAUSAL_PREFILL_ROWS: bool = true;
    type FeedForward = Mlp<B>;

    fn validate_with_metadata(
        _config: &C,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), Error> {
        // This factory has no additional configuration predicate.
        Ok(())
    }

    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error> {
        TransformerBlock::new(config, layer, context)
    }

    fn parameter_groups(
        block: &TransformerBlock<B, Self::FeedForward>,
        config: &C,
        layer: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        layer_parallel_parameter_groups(block, config, layer)
    }
    fn parameter_groups_with_metadata(
        block: &TransformerBlock<B, Self::FeedForward>,
        config: &C,
        layer: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Option<Result<Vec<ParameterGroupSpec>, Error>> {
        Some(layer_parallel_parameter_groups_with_metadata(
            block, config, layer, context,
        ))
    }
}

/// Shared layered decoder lifecycle over architecture configuration and block policy.
pub struct LayeredModel<B: NeuralBackend, C: Config, P = DenseBlockFactory> {
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry<C>>>,
    block_factory: std::marker::PhantomData<fn() -> P>,
    // Retire module shells before their retained immutable source configuration.
    args: crate::replicated_text::ConfigOwner<C>,
}

/// Pinned decoder modules physically present on one pipeline partition.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PartitionStaticModules<B: NeuralBackend> {
    /// Input embedding, present on the input owner and for tied output.
    pub embeddings: Option<B::Embedding>,
    /// Final normalization, present only on the output owner.
    pub norm: Option<B::Normalization>,
    /// Untied vocabulary head, present only on the output owner.
    pub lm_head: Option<B::Linear>,
}

impl<B: NeuralBackend> Clone for PartitionStaticModules<B> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
        }
    }
}

impl<B: eredu_nn::DistributedNeuralBackend> PartitionStaticModules<B> {
    /// Shared readout for an output-owning pipeline partition.
    pub(crate) fn finish_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let norm = self
            .norm
            .as_mut()
            .ok_or_else(|| Error::backend("decoder partition does not own final normalization"))?;
        let hidden = instrumentation.normalize_readout(hidden, norm, context)?;
        let logits = match (parallel, self.lm_head.as_mut()) {
            (Some(parallel), Some(head)) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                &hidden,
                parallel,
                context,
            ),
            (None, Some(head)) => {
                instrumentation.project::<B>("projection_input", head, &hidden, None, context)
            }
            (Some(parallel), None) => instrumentation.project_vocabulary_embedding::<B>(
                "projection_input",
                self.embeddings.as_mut().ok_or_else(|| {
                    Error::backend("tied decoder output partition has no embedding")
                })?,
                &hidden,
                parallel,
                context,
            ),
            (None, None) => instrumentation.project_embedding(
                "projection_input",
                self.embeddings.as_mut().ok_or_else(|| {
                    Error::backend("tied decoder output partition has no embedding")
                })?,
                &hidden,
                context,
            ),
        }?;
        let logits = instrumentation.apply("linear", logits)?;
        Ok(logits)
    }
}

/// Backend-neutral geometry retained by one genuinely pipeline-local decoder.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry<C> {
    owned_units: Range<usize>,
    blocks: Vec<C>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
}

impl<C: Config> PartitionLocalGeometry<C> {
    /// Returns the architecture-global execution-unit range allocated locally.
    pub fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }

    /// Returns the local configuration for one architecture-global unit.
    pub fn block(&self, global_unit: usize) -> Option<&C> {
        self.owned_units
            .contains(&global_unit)
            .then(|| &self.blocks[global_unit - self.owned_units.start])
    }

    /// Returns the number of locally allocated unit configurations.
    pub fn local_unit_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns complete TP-local state geometry before pipeline slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }
}

/// A dense decoder whose modules are limited to one admitted pipeline partition.
pub struct PartitionedLayeredModel<B: NeuralBackend, C: Config, P = DenseBlockFactory> {
    source: std::sync::Arc<PartitionModelSource<C>>,
    static_modules: PartitionStaticModules<B>,
    block_factory: std::marker::PhantomData<fn() -> P>,
}

// The immutable semantic payload from this exact completed constructor. It
// owns no tensors, source graph, request, quote Context or execution authority.
pub(crate) struct PartitionModelSource<C> {
    args: C,
    geometry: PartitionLocalGeometry<C>,
    parameters: ArchitectureParameterDescription,
    ownership: eredu_runtime::PartitionOwnership,
}

fn partition_static_modules<B, C>(
    config: &C,
    geometry: &PartitionLocalGeometry<C>,
    ownership: &eredu_runtime::PartitionOwnership,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PartitionStaticModules<B>, Error>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: Config,
{
    let metadata = module_metadata::ModuleMetadata::new::<B>(context);
    metadata.controls::<(
        PartitionStaticModules<B>,
        Result<PartitionStaticModules<B>, Error>,
    )>()?;
    let embedding_name = metadata.text(format_args!(
        "{}.embed_tokens.weight",
        config.parameter_root()
    ))?;
    let embeddings = (ownership.owns_input()
        || (ownership.owns_output() && config.tie_word_embeddings()))
    .then(|| {
        B::vocabulary_parallel_embedding(
            EmbeddingSpec {
                vocabulary: config.vocabulary_size(),
                dimensions: config.hidden_size(),
                weight: metadata.plain_parameter(&embedding_name)?,
                format: metadata.format(
                    &embedding_name,
                    metadata.linear_format(config, &embedding_name)?,
                )?,
            },
            geometry.embedding_range.clone(),
            context,
        )
    })
    .transpose()?;
    let norm = ownership
        .owns_output()
        .then(|| {
            B::normalization(
                NormalizationConstructionSpec {
                    groups: config.normalization_groups(),
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    scale: normalization_scale(
                        metadata.named_parameter(format_args!(
                            "{}.norm.weight",
                            config.parameter_root()
                        ))?,
                        config.normalization_offset(),
                    ),
                },
                context,
            )
        })
        .transpose()?;
    let lm_head = (ownership.owns_output() && !config.tie_word_embeddings())
        .then(|| {
            let name = "lm_head.weight";
            let range = geometry.output_range.clone().ok_or_else(|| {
                metadata.error(format_args!(
                    "untied decoder output owner has no vocabulary range"
                ))
            })?;
            B::vocabulary_parallel_linear(
                LinearSpec {
                    input: config.hidden_size(),
                    output: config.vocabulary_size(),
                    weight: metadata.plain_parameter(name)?,
                    bias: None,
                    format: metadata.format(name, metadata.linear_format(config, name)?)?,
                },
                range,
                context,
            )
        })
        .transpose()?;
    Ok(PartitionStaticModules {
        embeddings,
        norm,
        lm_head,
    })
}

impl<B, C, P> PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
{
    /// Constructs only modules already selected by an exact neutral partition.
    pub fn from_partition<A>(
        args: C,
        parameters: &ArchitectureParameterDescription,
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry<C>, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate_config()?;
        P::validate(&args)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        args.validate_partition_parameters(parameters)?;
        if parameters.graph() != partition.graph()
            || parameters.unit_layout() != partition.unit_layout()
        {
            return Err(Error::backend(
                "decoder partition belongs to a different normalized parameter topology",
            ));
        }
        let [group] = partition.groups() else {
            return Err(Error::backend(
                "dense decoder partition must own exactly one execution group",
            ));
        };
        if group.group().as_str() != TEXT_DECODER_EXECUTION_GROUP
            || group.global_units() != partition.local_geometry().owned_units
        {
            return Err(Error::backend(
                "decoder module construction range differs from selected partition",
            ));
        }
        let owned = partition.local_geometry().owned_units();
        let state = partition
            .state()
            .ok_or_else(|| Error::backend("partitioned decoder has no selected state"))?;
        let expected_state = partition
            .local_geometry()
            .complete_state_layout()
            .slice(owned.clone())
            .map_err(Error::backend)?;
        if state.global_layer_offset() != owned.start || state.layout() != &expected_state {
            return Err(Error::backend(
                "decoder partition state does not match its global unit range",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let static_modules =
            partition_static_modules(&args, &geometry, partition.ownership(), context)?;
        Ok(Self {
            source: std::sync::Arc::new(PartitionModelSource {
                args,
                geometry,
                parameters: parameters.clone(),
                ownership: partition.ownership().clone(),
            }),
            static_modules,
            block_factory: std::marker::PhantomData,
        })
    }

    pub(crate) fn retained_partition_source(&self) -> std::sync::Arc<PartitionModelSource<C>> {
        self.source.clone()
    }

    pub(crate) fn from_retained_partition_source(
        source: std::sync::Arc<PartitionModelSource<C>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        // Only from_partition can create the source. Reuse its validated
        // ownership, local geometry and parameters without a deep clone.
        let metadata = module_metadata::ModuleMetadata::new::<B>(context);
        metadata.controls::<(
            Self,
            std::sync::Arc<PartitionModelSource<C>>,
            Result<Self, Error>,
        )>()?;
        let static_modules =
            partition_static_modules(&source.args, &source.geometry, &source.ownership, context)?;
        Ok(Self {
            source,
            static_modules,
            block_factory: std::marker::PhantomData,
        })
    }

    /// Returns normalized architecture configuration.
    pub fn args(&self) -> &C {
        &self.source.args
    }

    /// Returns exact local pipeline geometry.
    pub fn local_geometry(&self) -> &PartitionLocalGeometry<C> {
        &self.source.geometry
    }

    /// Returns the physically allocated static modules.
    pub const fn static_modules(&self) -> &PartitionStaticModules<B> {
        &self.static_modules
    }

    /// Constructs an admitted unit and rejects every unowned global index.
    pub fn construct_unit(
        &self,
        global_unit: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, P::FeedForward>, Error> {
        let config = self.source.geometry.block(global_unit).ok_or_else(|| {
            Error::backend(format!(
                "decoder unit {global_unit} is not owned by local range {:?}",
                self.source.geometry.owned_units
            ))
        })?;
        P::build_partitioned(&self.source.args, config, global_unit, context)
    }
}

/// Derives backend-free local decoder geometry for one selected PP range.
pub fn partition_local_geometry<C: PartitionedConfig>(
    config: &C,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry<C>, ParallelPlanError> {
    partition_local_geometry_with(config, layout, owned_units, |config, unit, layout| {
        config.local_block_config(unit, layout)
    })
}

/// Derives partition-local decoder geometry with a family-owned block localizer.
///
/// This remains family-blind: the callback is responsible only for translating
/// an already selected physical layout and routed realization into one local
/// configuration. Static ownership and state geometry stay centralized here.
pub(crate) fn partition_local_geometry_with<C, F>(
    config: &C,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    mut localize: F,
) -> Result<PartitionLocalGeometry<C>, ParallelPlanError>
where
    C: PartitionedConfig,
    F: FnMut(&C, usize, &LocalModelLayout) -> Result<C, Error>,
{
    let count = usize::try_from(config.num_hidden_layers())
        .map_err(|_| ParallelPlanError::InvalidGroup("decoder layer count exceeds usize".into()))?;
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "decoder local unit range {owned_units:?} is outside {count} layers"
        )));
    }
    let mut blocks = Vec::with_capacity(owned_units.len());
    let mut local_key_value_heads = Vec::with_capacity(count);
    for unit in 0..count {
        let local = localize(config, unit, layout)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        local_key_value_heads.push(local.num_key_value_heads());
        if owned_units.contains(&unit) {
            blocks.push(local);
        }
    }
    let complete_state_layout = StateLayout::new(
        cache_layout_with_key_value_heads(config, local_key_value_heads)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let vocabulary = usize::try_from(config.vocabulary_size())
        .map_err(|_| ParallelPlanError::InvalidGroup("decoder vocabulary exceeds usize".into()))?;
    let embedding_range = vocabulary_range(
        layout,
        &format!("{}.embed_tokens", config.parameter_root()),
        vocabulary,
    )?;
    let output_range = if config.tie_word_embeddings() {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    Ok(PartitionLocalGeometry {
        owned_units,
        blocks,
        embedding_range,
        output_range,
        complete_state_layout,
    })
}

/// Validates the common graph and unit ownership of a routed decoder description.
pub(crate) fn validate_partitioned_decoder_description(
    config: &impl Config,
    parameters: &ArchitectureParameterDescription,
) -> Result<(), Error> {
    let [group] = parameters.graph().groups() else {
        return Err(Error::backend(
            "partitioned decoder parameter graph must contain one execution group",
        ));
    };
    if group.id() != TEXT_DECODER_EXECUTION_GROUP {
        return Err(Error::backend(
            "partitioned decoder parameter graph names a different execution group",
        ));
    }
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    if parameters
        .unit_layout()
        .group_range(0)
        .map(|range| range.len())
        != Some(layers)
    {
        return Err(Error::backend(
            "partitioned decoder parameter unit layout differs from configured depth",
        ));
    }
    let group_id = eredu_runtime::ExecutionGroupId::new(TEXT_DECODER_EXECUTION_GROUP)
        .map_err(Error::backend)?;
    for layer in 0..layers {
        if !parameters.groups().iter().any(|owned| {
            owned.owner()
                == &eredu_runtime::ParameterGroupOwner::execution_unit(group_id.clone(), layer)
        }) {
            return Err(Error::backend(format!(
                "partitioned decoder parameter description omits unit {layer}"
            )));
        }
    }
    Ok(())
}

fn dense_local_block_config<C: PartitionedConfig>(
    config: &C,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<C, Error> {
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    let fields = config.block_parameter_fields().validate()?;
    let tensor = |path: String| {
        layout
            .tensor(&path)
            .ok_or_else(|| Error::backend(format!("missing local decoder layout for {path}")))
    };
    let query = tensor(format!(
        "{prefix}.{}.{}.weight",
        fields.attention, fields.attention_query
    ))?;
    let key = tensor(format!(
        "{prefix}.{}.{}.weight",
        fields.attention, fields.attention_key
    ))?;
    let gate = tensor(format!(
        "{prefix}.{}.{}.weight",
        fields.feed_forward, fields.feed_forward_gate
    ))?;
    let query_width = i32::try_from(query.local_shape()[0]).map_err(Error::backend)?;
    let key_width = i32::try_from(key.local_shape()[0]).map_err(Error::backend)?;
    if query_width <= 0
        || key_width <= 0
        || query_width % config.head_dim() != 0
        || key_width % config.head_dim() != 0
    {
        return Err(Error::backend("invalid local decoder attention geometry"));
    }
    let mut local = config.clone();
    local.set_local_geometry(
        query_width / config.head_dim(),
        key_width / config.head_dim(),
        i32::try_from(gate.local_shape()[0]).map_err(Error::backend)?,
    )?;
    Ok(local)
}

impl<B, C, P> PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
{
    fn begin_hidden<S>(
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let metadata = module_metadata::ModuleMetadata::new::<B>(context);
        metadata.controls::<(
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>,
            eredu_runtime::layered::LayeredMetadata<Error>,
        )>()?;
        if state.layout() != expected {
            return Err(metadata.error(format_args!(
                "decoder runtime state does not match partition state layout"
            )));
        }
        let sequence = hidden.dim(1);
        let allow_sliding_prefill = mask.is_none();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => {
                let cache = state
                    .layer(0)
                    .map_err(|cause| metadata.error(format_args!("{cause}")))?;
                Some(B::causal_mask(sequence, cache.offset(), None, context)?)
            }
            None => None,
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                allow_sliding_prefill,
                rotary_embeddings: None,
                metadata: B::construction_metadata(context)
                    .filter(|c| c.uses_checked_metadata())
                    .cloned(),
            },
        })
    }

    fn finish_local(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_local_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn finish_local_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let norm =
            self.static_modules.norm.as_mut().ok_or_else(|| {
                Error::backend("decoder partition does not own final normalization")
            })?;
        let hidden = instrumentation.normalize_readout(hidden, norm, context)?;
        let logits = match (parallel, self.static_modules.lm_head.as_mut()) {
            (Some(parallel), Some(head)) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                &hidden,
                parallel,
                context,
            ),
            (None, Some(head)) => {
                instrumentation.project::<B>("projection_input", head, &hidden, None, context)
            }
            (Some(parallel), None) => instrumentation.project_vocabulary_embedding::<B>(
                "projection_input",
                self.static_modules.embeddings.as_mut().ok_or_else(|| {
                    Error::backend("tied decoder output partition has no embedding")
                })?,
                &hidden,
                parallel,
                context,
            ),
            (None, None) => instrumentation.project_embedding(
                "projection_input",
                self.static_modules.embeddings.as_mut().ok_or_else(|| {
                    Error::backend("tied decoder output partition has no embedding")
                })?,
                &hidden,
                context,
            ),
        }?;
        let logits = instrumentation.apply("linear", logits)?;
        softcap_logits(logits, self.source.args.output_softcap(), context)
    }
}

impl<B, C, P> eredu_runtime::ArchitectureParameters<B> for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
{
    type DefinitionError = Error;

    fn state_layout(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError> {
        match context {
            Some(context) => self
                .source
                .geometry
                .complete_state_layout
                .clone_workspace(context),
            None => Ok(self.source.geometry.complete_state_layout.clone()),
        }
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        match context {
            Some(context) => state_identity_with(
                &self.source.args,
                state.layout(),
                state.global_layer_offset(),
                topology,
                identity::Metadata::new(Some(context)),
            ),
            None => state_identity(
                &self.source.args,
                state.layout(),
                state.global_layer_offset(),
                topology,
            ),
        }
    }

    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        Ok(std::borrow::Cow::Borrowed(&self.source.parameters))
    }

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_nn::Parameterized::retained_value_slot_bound(&self.static_modules)
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_nn::Parameterized::visit_retained_values(&self.static_modules, visitor)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(embedding) = &self.static_modules.embeddings {
            visitor.visit("embedding", embedding)?;
        }
        if let Some(norm) = &self.static_modules.norm {
            visitor.visit("norm", norm)?;
        }
        if let Some(head) = &self.static_modules.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(embedding) = &mut self.static_modules.embeddings {
            visitor.visit_mut("embedding", embedding)?;
        }
        if let Some(norm) = &mut self.static_modules.norm {
            visitor.visit_mut("norm", norm)?;
        }
        if let Some(head) = &mut self.static_modules.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B, C, P, S> LayeredArchitecture<B, S> for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn prefill_observation_declarations(
        &self,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &Self,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            usize,
            usize,
            String,
            Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            std::ops::Range<usize>,
            Option<eredu_runtime::RoutedObservationPoints>,
            Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Error>,
        )>()?;

        // Partitioning changes parameter/state ownership, not the ordinary
        // decoder's causal row semantics. Keep the complete global declaration
        // so inactive ranks can reserve the same remote observation receipt;
        // the separate retained placement still determines actual hook owners.
        prefill_observations::declarations(
            &self.source.args,
            P::CAUSAL_PREFILL_ROWS,
            |layer| {
                <P as BlockFactory<B, C>>::feed_forward_residual_observation(
                    &self.source.args,
                    layer,
                )
            },
            |path, layer, declarations| {
                <P as BlockFactory<B, C>>::append_component_prefill_observations(
                    &self.source.args,
                    path,
                    layer,
                    declarations,
                    metadata_context,
                )
            },
            metadata_context,
        )
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    type Input<'a> = LayeredInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = PartitionStaticModules<B>;
    type Unit = TransformerBlock<B, P::FeedForward>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        crate::transport::decoder()
    }

    fn group_transport_matches(
        &self,
        _group: usize,
        expected: &eredu_runtime::ArchitectureGroupTransport,
    ) -> bool {
        crate::transport::decoder_declaration().matches(expected)
    }

    fn forward_metadata(
        &self,
        forward: &Self::ForwardContext,
    ) -> Option<eredu_runtime::layered::LayeredMetadata<Error>> {
        forward
            .metadata
            .as_ref()
            .map(|context| eredu_runtime::layered::LayeredMetadata::new(context, |error| error))
    }

    fn primary_execution_group(&self) -> &str {
        TEXT_DECODER_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(0, layout)
    }

    fn execution_graph(
        &self,
    ) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            self.source.parameters.graph(),
        ))
    }

    fn group_unit_count(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if group != 0 {
            return Err(metadata.error(format_args!(
                "{}",
                "decoder group is outside the text decoder"
            )));
        }
        usize::try_from(self.source.args.num_hidden_layers())
            .map_err(|cause| metadata.source(cause))
    }

    fn unit_path(
        &self,
        group: usize,
        index: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            usize,
            usize,
            Option<&eredu_nn::workspace::WorkspaceContext>,
        )>()?;

        if group != 0
            || index
                >= usize::try_from(self.source.args.num_hidden_layers())
                    .map_err(|cause| metadata.source(cause))?
        {
            return Err(metadata.error(format_args!(
                "{}",
                "decoder unit is outside the text decoder"
            )));
        }
        metadata.text(format_args!(
            "{}.layers.{index}",
            self.source.args.parameter_root()
        ))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        if group != 0 {
            return Err(Error::backend("decoder group is outside the text decoder"));
        }
        self.construct_unit(index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = self
            .static_modules
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("decoder partition does not own input embedding"))?
            .forward(input.tokens, context)?;
        let hidden = scale_token_embeddings(hidden, self.source.args.embedding_scale(), context)?;
        Self::begin_hidden(
            hidden,
            input.mask,
            state,
            self.source.geometry.complete_state_layout(),
            0,
            context,
        )
    }

    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_forward(input, state, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend(
                "text decoder received invalid group dependencies",
            ));
        }
        Ok(initial.clone())
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("decoder attempted an unowned unit"));
        }
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        unit.forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            context,
        )
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("decoder attempted an unowned unit"));
        }
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_observed(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }

    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        crate::readout::select_readout_positions(hidden, demand, 1, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, None, context)
    }

    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_local_instrumented(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, C, P, S> ParallelLayeredArchitecture<B, S> for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = B::vocabulary_parallel_lookup(
            self.static_modules
                .embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("decoder partition does not own input embedding"))?,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let hidden = scale_token_embeddings(hidden, self.source.args.embedding_scale(), context)?;
        Self::begin_hidden(
            hidden,
            input.mask,
            state,
            self.source.geometry.complete_state_layout(),
            0,
            context,
        )
    }

    fn begin_forward_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_forward_parallel(input, state, parallel, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("decoder attempted an unowned unit"));
        }
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        unit.forward_tensor_parallel(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            parallel,
            context,
        )
    }

    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("decoder attempted an unowned unit"));
        }
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_tensor_parallel_observed(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, Some(parallel), context)
    }

    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_local_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, C, P, S> eredu_runtime::RoutedLayeredArchitecture<B, S> for PartitionedLayeredModel<B, C, P>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    P::FeedForward: RoutedProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        <P::FeedForward as RoutedProjectionOperator<B>>::COMPONENT_OBSERVATIONS
    }

    fn routed_sparse_observations(&self) -> bool {
        true
    }

    fn routed_observation_points(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Self::Error> {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend(
                "routed decoder observation requested for an unowned unit",
            ));
        }
        let unit_path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        Ok(self
            .source
            .args
            .routed_observation_points(&unit_path, index, None)?)
    }

    fn forward_unit_observed_with_provider<R, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::RoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let points = self
            .source
            .args
            .routed_observation_points(&path, index, None)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("routed decoder attempted an unowned unit"));
        }
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        unit.forward_routed_observed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            points,
        )
    }

    fn forward_unit_with_provider<R>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::RoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
    {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("routed decoder attempted an unowned unit"));
        }
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        unit.forward_routed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            pass,
            provider,
            context,
        )
    }
}

impl<B, C, P, S> eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
    for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelRoutedProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        <P::FeedForward as RoutedProjectionOperator<B>>::COMPONENT_OBSERVATIONS
    }

    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }

    fn forward_unit_parallel_observed_with_provider<R, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let points = self
            .source
            .args
            .routed_observation_points(&path, index, None)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("routed decoder attempted an unowned unit"));
        }
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        unit.forward_routed_parallel_observed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            points,
        )
    }

    fn forward_unit_parallel_with_provider<R>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
    {
        if group != 0 || !self.source.geometry.owned_units.contains(&index) {
            return Err(Error::backend("routed decoder attempted an unowned unit"));
        }
        let cache = state
            .layer(index - self.source.geometry.owned_units.start)
            .map_err(Error::backend)?;
        unit.forward_routed_parallel(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: None,
            },
            pass,
            provider,
            parallel,
            context,
        )
    }
}

impl<B, C, P, S> PartitionedLayeredArchitecture<B, S> for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self,
                Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary,
                Result<Self::Boundary, Self::Error>,
            )>())?;
        }

        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.source.args.hidden_size(),
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let hidden = self
                    .static_modules
                    .embeddings
                    .as_mut()
                    .ok_or_else(|| {
                        Error::backend("decoder partition does not own input embedding")
                    })?
                    .forward(tokens, context)?;
                scale_token_embeddings(hidden, self.source.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        Self::begin_hidden(hidden, mask, state, expected, first_state_ordinal, context)
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let hidden = B::vocabulary_parallel_lookup(
                    self.static_modules.embeddings.as_mut().ok_or_else(|| {
                        Error::backend("decoder partition does not own input embedding")
                    })?,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                scale_token_embeddings(hidden, self.source.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        Self::begin_hidden(hidden, mask, state, expected, first_state_ordinal, context)
    }

    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = match parallel {
            Some(parallel) => self.begin_partition_parallel(
                input,
                mask,
                state,
                expected,
                first_state_ordinal,
                parallel,
                context,
            ),
            None => {
                self.begin_partition(input, mask, state, expected, first_state_ordinal, context)
            }
        }?;
        if embedded {
            let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
                .apply("embedding", forward.hidden)?;
        }
        Ok(forward)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output: self.finish_local(hidden, parallel, context)?,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }

    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut observer);
        let output =
            self.finish_local_instrumented(hidden, parallel, context, &mut instrumentation)?;
        Ok(eredu_runtime::LayeredPartitionOutput::Final {
            output,
            retained: None,
        })
    }
}

impl<B, C, P, S> eredu_runtime::ReplicatedTextArchitecture<B, S>
    for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, C, P, S> crate::partitioned_execution::TextPartitionArchitecture<B, S>
    for PartitionedLayeredModel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn partition_text_input<'a>(input: Self::Input<'a>) -> (&'a B::Tensor, Option<&'a B::Tensor>) {
        (input.tokens, input.mask)
    }

    fn partition_output_width(&self) -> i32 {
        self.source.args.vocabulary_size()
    }

    fn partition_routed_bank_order(&self, unit: usize) -> Vec<eredu_runtime::RoutedBankId> {
        self.source.args.routed_bank_order(unit)
    }

    fn partition_routed_bank_tensor_reductions(
        &self,
        unit: usize,
        bank: eredu_runtime::RoutedBankId,
    ) -> Result<(usize, usize), Error> {
        self.source.args.routed_bank_tensor_reductions(unit, bank)
    }
}

impl<B, C, P> LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    /// Builds unloaded pinned modules from normalized architecture arguments.
    pub fn new(args: C, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Self::new_with_config(args.into(), context)
    }

    pub(crate) fn new_with_config(
        args: crate::replicated_text::ConfigOwner<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(&C, std::marker::PhantomData<P>)>()?;
        if let Some(context) = metadata.context() {
            args.validate_config_with_metadata(context)?;
            P::validate_with_metadata(&*args, context)?;
        } else {
            args.validate_config()?;
            P::validate(&*args)?;
        }
        crate::operator_requirements::require_with_metadata::<B>(
            "shared decoder equations",
            operator_requirements(&*args),
            metadata,
        )?;
        let static_modules = StaticModules::new(&*args, context)?;
        Ok(Self {
            args,
            static_modules,
            parallel_geometry: None,
            block_factory: std::marker::PhantomData,
        })
    }

    /// Builds the same model lifecycle with planner-derived rank-local modules.
    pub fn new_parallel(
        args: C,
        geometry: LocalGeometry<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        args.validate_config()?;
        P::validate(&args)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let static_modules = StaticModules::new_parallel(&args, &geometry, context)?;
        Ok(Self {
            args: args.into(),
            static_modules,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
            block_factory: std::marker::PhantomData,
        })
    }

    /// Returns the normalized architecture arguments.
    pub fn args(&self) -> &C {
        &self.args
    }

    /// Borrows pinned modules for neutral checkpoint loading.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for neutral checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Returns the replicated or rank-local mutable-state layout for this model.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| state_layout(self.args()), Ok)
    }

    /// Returns planner-derived geometry when this is a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry<C>> {
        match self.parallel_geometry.as_ref() {
            Some(geometry) => Some(geometry.as_ref()),
            None => None,
        }
    }

    /// Shares the authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry<C>>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical replicated or rank-local decoder unit.
    ///
    /// Residency and pipeline runtimes use this entry point so streamed units
    /// cannot drift from the model's planner-derived geometry.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, P::FeedForward>, Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(usize, &C, TransformerBlock<B, P::FeedForward>)>()?;
        let count = usize::try_from(self.args.num_hidden_layers())
            .map_err(|cause| metadata.error(format_args!("{cause}")))?;
        if index >= count {
            return Err(metadata.error(format_args!(
                "decoder unit {index} is outside {count} decoder layers"
            )));
        }
        let args = match &self.parallel_geometry {
            Some(geometry) => geometry.block(index).ok_or_else(|| {
                metadata.error(format_args!(
                    "decoder local geometry is missing block {index}"
                ))
            })?,
            None => self.args(),
        };
        P::build_partitioned(self.args(), args, index, context)
    }

    /// Prepares architecture-owned mask state after an execution policy has
    /// produced embeddings, including vocabulary-parallel embeddings.
    pub fn begin_embedded<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let expected = state_layout(self.args())?;
        self.begin_embedded_with_layout(hidden, supplied_mask, state, &expected, context)
    }

    /// Prepares architecture-owned mask state against an explicitly realized
    /// state layout, such as the rank-local KV geometry produced by tensor
    /// parallel planning.
    pub fn begin_embedded_with_layout<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.begin_embedded_with_layout_and_rotary(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            context,
        )
    }

    /// Starts one replicated pipeline partition from tokens or upstream hidden
    /// state using the partition's authoritative local state layout.
    fn prepare_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let hidden = self.static_modules.embeddings.forward(tokens, context)?;
                scale_token_embeddings(hidden, self.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_embedded_with_layout_at(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Starts one tensor-parallel pipeline partition through the same neutral
    /// entry point, including vocabulary-parallel embedding on the input rank.
    fn prepare_partition_parallel<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let hidden = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                scale_token_embeddings(hidden, self.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_embedded_with_layout_at(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Prepares a layered pass with caller-provided explicit rotary embeddings.
    pub fn begin_embedded_with_layout_and_rotary<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.begin_embedded_with_layout_at(
            hidden,
            supplied_mask,
            rotary_embeddings,
            state,
            expected,
            0,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_embedded_with_layout_at<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "decoder runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let sequence = hidden.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if sequence > 1 {
            let cache = state.layer(first_state_ordinal).map_err(Error::backend)?;
            // The shared mask is consumed by full-attention layers. Sliding
            // layers use their typed window-aware attention path, so deriving
            // this mask from layer zero's retention policy would incorrectly
            // impose that window on later full-attention layers.
            Some(B::causal_mask(sequence, cache.offset(), None, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                allow_sliding_prefill: supplied_mask.is_none(),
                rotary_embeddings: rotary_embeddings
                    .map(|(cosine, sine)| (cosine.clone(), sine.clone())),
                metadata: None,
            },
        })
    }

    /// Executes one replicated block using architecture-owned forward state.
    pub fn forward_block<S>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
        )
    }

    /// Executes one replicated block while delegating its feed-forward policy
    /// to a composition-supplied executor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_with_feed_forward<S, H>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        H: FnOnce(
            &mut P::FeedForward,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_with_feed_forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
            feed_forward,
        )
    }

    /// Executes one tensor-parallel block using the same architecture-owned
    /// mask and state semantics as replicated execution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_parallel<S>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P::FeedForward: TensorParallelProjectionOperator<B>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_tensor_parallel(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
        )
    }

    /// Executes one tensor-parallel block while delegating its feed-forward
    /// policy to a composition-supplied executor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_parallel_with_feed_forward<S, H>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        H: FnOnce(
            &mut P::FeedForward,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_tensor_parallel_with_feed_forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
            feed_forward,
        )
    }

    /// Applies final normalization and the tied or untied output projection.
    pub fn finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_hidden_instrumented(hidden, context, &mut ComponentInstrumentation::disabled())
    }

    fn finish_hidden_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let logits = self
            .static_modules
            .finish_instrumented(hidden, context, instrumentation)?;
        softcap_logits(logits, self.args.output_softcap(), context)
    }

    /// Applies rank-local normalization and vocabulary-parallel projection for
    /// an output-owning pipeline partition.
    pub fn finish_hidden_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        self.finish_hidden_parallel_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn finish_hidden_parallel_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let hidden =
            instrumentation.normalize_readout(hidden, &mut self.static_modules.norm, context)?;
        let logits = match &mut self.static_modules.lm_head {
            Some(head) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                &hidden,
                parallel,
                context,
            ),
            None => instrumentation.project_vocabulary_embedding::<B>(
                "projection_input",
                &mut self.static_modules.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }?;
        let logits = instrumentation.apply("linear", logits)?;
        softcap_logits(logits, self.args.output_softcap(), context)
    }
}

impl<B, C, P> LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    /// Describes every shared-decoder parameter group with explicit static or
    /// architecture-global unit ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let metadata =
            B::construction_metadata(context).filter(|context| context.uses_checked_metadata());
        if let Some(metadata) = metadata {
            let controls = [
                size_of::<ArchitectureParameterDescription>(),
                size_of::<ExecutionGraph>(),
                size_of::<ExecutionUnitLayout>(),
                size_of::<[usize; 1]>(),
                size_of::<Vec<OwnedParameterGroupSpec>>(),
                size_of::<Option<Vec<ParameterGroupSpec>>>(),
                size_of::<ParameterGroupOwner>(),
                size_of::<Result<ArchitectureParameterDescription, Error>>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            metadata.charge_metadata(controls)?;
        }
        let graph = match metadata {
            Some(context) => {
                ExecutionGraph::single_with_metadata(TEXT_DECODER_EXECUTION_GROUP, context)?
            }
            None => {
                ExecutionGraph::chain([TEXT_DECODER_EXECUTION_GROUP]).map_err(Error::backend)?
            }
        };
        let count =
            usize::try_from(self.args.num_hidden_layers()).map_err(|cause| match metadata {
                Some(context) => context.metadata_source(cause),
                None => Error::backend(cause),
            })?;
        let layout = match metadata {
            Some(context) => ExecutionUnitLayout::new_with_metadata(&graph, &[count], context)?,
            None => ExecutionUnitLayout::new(&graph, [count]).map_err(Error::backend)?,
        };
        let static_groups = parameter_metadata::static_groups::<B>(
            &self.static_modules.embeddings,
            &self.static_modules.norm,
            self.static_modules.lm_head.as_ref(),
            self.args.parameter_root(),
            metadata,
        )
        .map_err(parameter_metadata::ParameterGroupError::into_neural)?;
        let mut expected = metadata.is_none().then(|| static_groups.clone());
        let mut owned = match metadata {
            Some(context) => context.metadata_vec(static_groups.len())?,
            None => Vec::new(),
        };
        for (index, group) in static_groups.into_iter().enumerate() {
            let role = match index {
                0 => "embedding",
                1 => "norm",
                _ => "output",
            };
            let owner = if index == 0 && self.args.tie_word_embeddings() {
                match metadata {
                    Some(context) => {
                        let mut roles = context.metadata_vec(2)?;
                        roles.push(context.metadata_string(format_args!("embedding"))?);
                        roles.push(context.metadata_string(format_args!("output"))?);
                        ParameterGroupOwner::StaticAnyOf(roles)
                    }
                    None => ParameterGroupOwner::static_any_of(["embedding", "output"]),
                }
            } else {
                match metadata {
                    Some(context) => ParameterGroupOwner::static_role(
                        context.metadata_string(format_args!("{role}"))?,
                    ),
                    None => ParameterGroupOwner::static_role(role),
                }
            };
            owned.push(OwnedParameterGroupSpec::new(owner, group));
        }
        let group_id = layout.group_id(0).expect("decoder layout group");
        for index in 0..count {
            let unit = self.construct_unit(index, context)?;
            let groups = match metadata {
                Some(context) => {
                    P::parameter_groups_with_metadata(&unit, self.args(), index, context)
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)??
                }
                None => P::parameter_groups(&unit, self.args(), index).map_err(Error::backend)?,
            };
            if let Some(expected) = &mut expected {
                expected.extend(groups.iter().cloned());
            }
            if let Some(context) = metadata {
                context.reserve_metadata_vec(&mut owned, groups.len())?;
            }
            for group in groups {
                let group_id = match metadata {
                    Some(context) => eredu_runtime::ExecutionGroupId::new(
                        context.metadata_string(format_args!("{}", group_id.as_str()))?,
                    )
                    .map_err(|cause| context.metadata_source(cause))?,
                    None => group_id.clone(),
                };
                owned.push(OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(group_id, index),
                    group,
                ));
            }
        }
        match metadata {
            Some(context) => ArchitectureParameterDescription::from_owned_with_metadata(
                graph, layout, owned, context,
            ),
            None => ArchitectureParameterDescription::new(
                &graph,
                &layout,
                expected.expect("ordinary description retains expected groups"),
                owned,
            )
            .map_err(Error::backend),
        }
    }
}

impl<B, C, P> eredu_runtime::ArchitectureParameters<B> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    type DefinitionError = Error;

    fn state_layout(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError> {
        match context {
            Some(context) => match &self.parallel_geometry {
                Some(geometry) => geometry.state_layout.clone_workspace(context),
                None => state_layout_with_metadata(self.args(), context),
            },
            None => self.state_layout_impl(),
        }
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        match context {
            Some(context) => state_identity_with(
                self.args(),
                state.layout(),
                state.global_layer_offset(),
                topology,
                identity::Metadata::new(Some(context)),
            ),
            None => state_identity(
                self.args(),
                state.layout(),
                state.global_layer_offset(),
                topology,
            ),
        }
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
            &Self,
            &<B::Tensor as Tensor>::Context,
            std::borrow::Cow<'_, ArchitectureParameterDescription>,
        )>()?;
        self.parameter_description_impl(context)
            .map(std::borrow::Cow::Owned)
    }

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_nn::Parameterized::retained_value_slot_bound(&self.static_modules)
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_nn::Parameterized::visit_retained_values(&self.static_modules, visitor)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        visitor.visit("embedding", &self.static_modules.embeddings)?;
        visitor.visit("norm", &self.static_modules.norm)?;
        if let Some(head) = &self.static_modules.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.norm)?;
        if let Some(head) = &mut self.static_modules.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B, C, P, S> LayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = TransformerBlock<B, P::FeedForward>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn prefill_observation_declarations(
        &self,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &Self,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            usize,
            usize,
            String,
            Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            std::ops::Range<usize>,
            Option<eredu_runtime::RoutedObservationPoints>,
            Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Error>,
        )>()?;

        prefill_observations::declarations(
            self.args(),
            P::CAUSAL_PREFILL_ROWS,
            |layer| {
                <P as BlockFactory<B, C>>::feed_forward_residual_observation(self.args(), layer)
            },
            |path, layer, declarations| {
                <P as BlockFactory<B, C>>::append_component_prefill_observations(
                    self.args(),
                    path,
                    layer,
                    declarations,
                    metadata_context,
                )
            },
            metadata_context,
        )
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        crate::transport::decoder()
    }

    fn group_transport_matches(
        &self,
        _group: usize,
        expected: &eredu_runtime::ArchitectureGroupTransport,
    ) -> bool {
        crate::transport::decoder_declaration().matches(expected)
    }

    fn primary_execution_group(&self) -> &str {
        TEXT_DECODER_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(0, layout)
    }

    fn execution_graph(
        &self,
    ) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        eredu_runtime::ArchitectureExecutionGraph::single(TEXT_DECODER_EXECUTION_GROUP)
            .map_err(Error::backend)
    }

    fn group_unit_count(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if group != 0 {
            return Err(metadata.error(format_args!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        usize::try_from(self.args.num_hidden_layers()).map_err(|cause| metadata.source(cause))
    }

    fn unit_path(
        &self,
        group: usize,
        index: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            usize,
            usize,
            Option<&eredu_nn::workspace::WorkspaceContext>,
        )>()?;

        if group != 0 {
            return Err(metadata.error(format_args!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        let count = usize::try_from(self.args.num_hidden_layers())
            .map_err(|cause| metadata.source(cause))?;
        if index >= count {
            return Err(metadata.error(format_args!(
                "decoder unit {index} is outside {count} decoder layers"
            )));
        }
        metadata.text(format_args!(
            "{}.layers.{index}",
            self.args.parameter_root()
        ))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(usize, usize, Result<Self::Unit, Self::Error>)>()?;
        if group != 0 {
            return Err(metadata.error(format_args!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        self.construct_unit(index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = self
            .static_modules
            .embeddings
            .forward(input.tokens, context)?;
        let hidden = scale_token_embeddings(hidden, self.args.embedding_scale(), context)?;
        self.begin_embedded(hidden, input.mask, state, context)
    }

    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_forward(input, state, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend(format!(
                "text decoder group {group} received {} dependencies",
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }

    fn forward_unit(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block(index, unit, hidden, state, forward, context)
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let cache = state.layer(index).map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_observed(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }

    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        crate::readout::select_readout_positions(hidden, demand, 1, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_hidden(hidden, context)
    }

    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_hidden_instrumented(
            hidden,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, C, P, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, C, P, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("decoder model was not built with local geometry"))?
            .state_layout()
            .clone();
        let hidden = B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let hidden = scale_token_embeddings(hidden, self.args.embedding_scale(), context)?;
        self.begin_embedded_with_layout(hidden, input.mask, state, &expected, context)
    }

    fn begin_forward_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_forward_parallel(input, state, parallel, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }

    fn forward_unit_parallel(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block_parallel(index, unit, hidden, state, forward, parallel, context)
    }

    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let cache = state.layer(index).map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_tensor_parallel_observed(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_hidden_parallel(hidden, parallel, context)
    }

    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_hidden_parallel_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, C, P, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    fn boundary_schema(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self,
                Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary,
                Result<Self::Boundary, Self::Error>,
            )>())?;
        }

        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args().hidden_size(),
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        LayeredModel::prepare_partition(
            self,
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        LayeredModel::prepare_partition_parallel(
            self,
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
        )
    }

    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = match parallel {
            Some(parallel) => self.begin_partition_parallel(
                input,
                mask,
                state,
                expected,
                first_state_ordinal,
                parallel,
                context,
            ),
            None => {
                self.begin_partition(input, mask, state, expected, first_state_ordinal, context)
            }
        }?;
        if embedded {
            let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
                .apply("embedding", forward.hidden)?;
        }
        Ok(forward)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }

    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut observer);
        let output = match parallel {
            Some(parallel) => self.finish_hidden_parallel_instrumented(
                hidden,
                parallel,
                context,
                &mut instrumentation,
            ),
            None => self.finish_hidden_instrumented(hidden, context, &mut instrumentation),
        }?;
        Ok(eredu_runtime::LayeredPartitionOutput::Final {
            output,
            retained: None,
        })
    }
}

impl<B, C, P, S> eredu_runtime::RoutedLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: GroupedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: RoutedProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        <P::FeedForward as RoutedProjectionOperator<B>>::COMPONENT_OBSERVATIONS
    }

    fn routed_sparse_observations(&self) -> bool {
        true
    }

    fn routed_observation_points(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Self::Error> {
        let unit_path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        Ok(self
            .args
            .routed_observation_points(&unit_path, index, None)?)
    }

    fn forward_unit_observed_with_provider<R, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::RoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let points = self.args.routed_observation_points(&path, index, None)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let cache = state.layer(index).map_err(Error::backend)?;
        unit.forward_routed_observed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            points,
        )
    }

    fn forward_unit_with_provider<R>(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::RoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        unit.forward_routed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            pass,
            provider,
            context,
        )
    }
}

impl<B, C, P, S> eredu_runtime::ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelRoutedProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        <P::FeedForward as RoutedProjectionOperator<B>>::COMPONENT_OBSERVATIONS
    }

    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }

    fn forward_unit_parallel_observed_with_provider<R, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index, None)?;
        let points = self.args.routed_observation_points(&path, index, None)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let cache = state.layer(index).map_err(Error::backend)?;
        unit.forward_routed_parallel_observed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            points,
        )
    }

    fn forward_unit_parallel_with_provider<R>(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        unit.forward_routed_parallel(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            pass,
            provider,
            parallel,
            context,
        )
    }
}

fn scale_token_embeddings<T: Tensor>(
    embeddings: T,
    scale: f32,
    context: &T::Context,
) -> Result<T, Error> {
    if scale == 1.0 {
        Ok(embeddings)
    } else {
        embeddings.multiply_scalar(scale, context)
    }
}

fn softcap_logits<T: Tensor>(
    logits: T,
    cap: Option<f32>,
    context: &T::Context,
) -> Result<T, Error> {
    match cap {
        Some(cap) => logits
            .multiply_scalar(cap.recip(), context)?
            .tanh(context)?
            .multiply_scalar(cap, context),
        None => Ok(logits),
    }
}

fn normalization_scale(weight: ParameterSpec, offset: f32) -> eredu_nn::NormalizationScale {
    if offset == 0.0 {
        eredu_nn::NormalizationScale::Learned(weight)
    } else {
        eredu_nn::NormalizationScale::LearnedOffset { weight, offset }
    }
}
