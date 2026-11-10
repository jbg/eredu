//! Shared backend-neutral decoder mechanics.
//!
//! Architecture families retain configuration, checkpoint naming, identity, and
//! policy while reusing these statically dispatched decoder operations.

use std::ops::Range;

// Shared by cold architecture requirements and the concrete layered executor.
pub(crate) const CHUNKED_TEXT_PREFILL: bool = true;

mod attention_parameters;
pub(crate) mod attention_partition;
mod boundary_modules;
pub use boundary_modules::BoundaryModules;
mod declarations;
/// Shared-weight repeated stacks with independent state for each invocation.
pub(crate) mod repeated;
pub(crate) mod topology;
pub use declarations::{DecoderDeclaration, DecoderLocalProjection, ProjectionDeclarations};

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::cache::LayerCachePolicy;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, AttentionRequest, DeclareLogicalParameters, EmbeddingLookupPolicy,
    EmbeddingOperator, EmbeddingSpec, Error, FusedProjectionLayout, FusedProjectionSegment,
    GatedProductPolicy, GroupedNeuralBackend, Index, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, Parameter, ParameterSpec, RotaryOperator,
    RotaryPosition, RotarySpec, RotarySubspace, Tensor, VocabularyParallelRange,
};
use eredu_runtime::PartitionComponentKernel as _;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, LayerRuntimeState,
    LayeredForwardState, LayeredPartitionInput, LocalModelLayout, MemberSharding,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterRole, StateLayout, VocabularyModulePlacement,
};

/// Narrows a causal decoder's position-independent readout before projection.
pub fn final_hidden_position<T: Tensor>(hidden: &T, context: &T::Context) -> Result<T, Error> {
    if hidden.shape().len() < 3 || hidden.dim(1) <= 0 {
        return Err(Error::backend(
            "decoder readout requires nonempty batch/sequence/features",
        ));
    }
    let sequence = hidden.dim(1);
    let mut indices = vec![Index::Full; hidden.shape().len()];
    indices[1] = Index::Range(sequence - 1, sequence);
    hidden.index(&indices, context)
}

/// Optional component instrumentation at the values actually consumed downstream.
/// An absent observer neither clones nor retains tensors and constructs no paths.
pub struct ComponentInstrumentation<'a, T> {
    observer: Option<&'a mut dyn eredu_runtime::ActivationObserver<T, Error>>,
    unit_path: &'a str,
}

/// Typed residual ingress and readout transformations.
pub mod boundary;
mod schedule;
pub use schedule::{DecoderOperation, DecoderSchedule};
pub mod unary;
pub use boundary::{DecoderBoundary, NormalizedBoundary};

impl<T: Clone> ComponentInstrumentation<'_, T> {
    /// Ordinary execution, without additional tensor work.
    pub fn disabled() -> Self {
        Self {
            observer: None,
            unit_path: "",
        }
    }

    /// Applies the decoder embedding seam while preserving request-local context.
    pub(crate) fn apply_embedding<C, O>(
        mut forward: LayeredForwardState<T, C>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<T, C>, Error>
    where
        O: eredu_runtime::ActivationObserver<T, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut borrowed)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
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
            observer: Some(observer),
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
}
impl<T: Tensor> ComponentInstrumentation<'_, T> {
    /// Selects once through the shared control/evidence policy before dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select_groups<B, P>(
        &mut self,
        boundary: &str,
        selector: &mut B::Selector,
        input: &T,
        provider: &mut P,
        bank: eredu_runtime::RoutedBankId,
        context: &T::Context,
    ) -> Result<eredu_nn::GroupSelection<T>, Error>
    where
        B: eredu_nn::GroupedNeuralBackend<Tensor = T>,
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self.observer.as_deref_mut() {
            Some(observer) => eredu_runtime::select_routes_with_observer(
                selector,
                input,
                context,
                &format!("{}.{}", self.unit_path, boundary),
                observer,
            ),
            None => eredu_runtime::select_routes_with_provider::<B, P>(
                selector, input, context, provider, bank,
            ),
        }
    }

    /// Emits one completed routing record after architecture-owned transforms,
    /// then applies the shared intervention on that complete routed contribution.
    pub(crate) fn routing_output(
        &mut self,
        boundary: &str,
        routes: &eredu_nn::GroupSelection<T>,
        output: T,
        expert_count: i32,
    ) -> Result<T, Error> {
        let Some(observer) = self.observer.as_deref_mut() else {
            return Ok(output);
        };
        let path = format!("{}.{}", self.unit_path, boundary);
        observer.observe_routing(eredu_runtime::RoutingObservation {
            path: &path,
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &output,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: None,
            combined_output: None,
            expert_count,
        })?;
        eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &output)
    }

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
            None => execute(request).map_err(Error::backend_source),
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
                return Err(Error::backend(format!(
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
    /// Canonical parameter fields used within each shared decoder block.
    fn block_parameter_fields(&self) -> BlockParameterFields<'_> {
        BlockParameterFields::default()
    }
    /// Returns the canonical routed-observation point for one decoder layer.
    fn routed_observation_points(
        &self,
        _unit_path: &str,
        _layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoints> {
        None
    }
    /// Validates architecture-owned configuration policy.
    fn validate_config(&self) -> Result<(), Error>;

    /// Resolves family storage references after complete common scopes are declared.
    /// Provider families validate their additional owners at the enclosing boundary.
    fn validate_parameter_formats(&self, _declaration: &DecoderDeclaration) -> Result<(), Error> {
        Ok(())
    }

    /// Adds independently owned family parameters to the common decoder topology.
    /// The supplied common groups already project retained neural scopes.
    fn extend_common_parameters(
        &self,
        common: ArchitectureParameterDescription,
    ) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
        Ok(common)
    }

    /// Transformer hidden size.
    fn hidden_size(&self) -> i32;
    /// Number of decoder layers.
    fn num_hidden_layers(&self) -> i32;
    /// Canonical body parameter owner for one logical layer invocation.
    /// Owners are stable logical layer indices with identical body construction
    /// policy. Mutable attention state remains local
    /// to each invocation. Families declare sharing explicitly, independently of
    /// checkpoint aliases.
    fn block_parameter_owner(&self, layer: usize) -> usize {
        layer
    }
    /// Optional learned RMS normalization after a block's final residual.
    /// The returned parameter belongs to an independently leased stateless operation.
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
    /// Position-dependent query gain, applied after rotary encoding.
    fn query_position_scaling(&self) -> Option<PositionQueryScaling> {
        None
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
        let fields = self.block_parameter_fields();
        self.linear_format(&format!(
            "{}.layers.{layer}.{}.{}.weight",
            self.parameter_root(),
            fields.attention,
            fields.attention_value
        ))
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
    /// Offset added to learned per-head Q/K RMS scales.
    fn query_key_normalization_offset(&self) -> f32 {
        0.0
    }
    /// Float32 reduction and gain policy for shared per-head Q/K scales.
    /// Per-head weight banks retain their head-count grouping.
    fn query_key_normalization_groups(&self) -> Option<i32> {
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
    /// Optional floating projection execution precision; residuals keep their input dtype.
    fn linear_compute_element(&self) -> Option<eredu_nn::TensorElementType> {
        None
    }

    /// Complete matrix encoding, including block-FP8 companions.
    fn linear_format(&self, name: &str) -> eredu_checkpoint::LinearFormat {
        self.weight_quantization(name).into()
    }
    /// Complete rotary-position construction specification.
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec;
    /// Layer-specific rotary policy; families may distinguish local and global heads.
    fn layer_rotary_spec(&self, _layer: usize, dimensions: i32) -> RotarySpec {
        self.rotary_spec(dimensions)
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
    if config.query_position_scaling().is_some() {
        required = required.union(Caps::CAST_FLOAT);
    }
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
    /// Restores all partition-owned dimensions before comparing source execution policy.
    fn restore_global_geometry(&mut self, global: &Self) -> Result<(), Error> {
        self.set_local_geometry(
            global.num_attention_heads(),
            global.num_key_value_heads(),
            global.intermediate_size(),
        )
    }

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
    /// Dense families inherit the ordinary split or fused projection derivation. Routed
    /// families override this hook because their expert bank has an additional
    /// owner-local group axis that is not the dense MLP intermediate axis.
    fn local_block_config(&self, layer: usize, layout: &LocalModelLayout) -> Result<Self, Error> {
        dense_local_block_config(self, layer, layout, None)
    }

    /// Declares the canonical topology admitted by partition construction.
    /// Specialized families supply their existing router, bank or adapter declaration.
    fn partition_parameter_description(
        &self,
    ) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
        dense_parameter_description(self)
    }
}

/// Declares cache identity shared by ordinary layered decoder families.
pub fn state_identity<C: Config>(
    args: &C,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, Error> {
    args.validate_config()?;
    topology.validate().map_err(Error::backend)?;
    let layer_count = usize::try_from(args.num_hidden_layers()).map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("decoder owned state range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "{} owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers",
            args.model_family()
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        args.model_family(),
        args.model_identity(),
        args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
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
    #[parameter(skip)]
    name: String,
    #[parameter(skip)]
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
        Self::from_specs(
            specs.into_iter().map(|spec| (spec, ())).collect(),
            |spec, ()| B::embedding(spec, context),
        )
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
        for (spec, range) in specs.iter().zip(&ranges) {
            range.validate_global_rows(spec.embedding.vocabulary)?;
        }
        Self::from_specs(specs.into_iter().zip(ranges).collect(), |spec, range| {
            VocabularyModulePlacement::VocabularyParallel(range)
                .materialize_embedding::<B>(spec, context)
                .map(|embedding| embedding.expect("selected embedding table is owned"))
        })
    }

    fn from_specs<P>(
        specs: Vec<(NamedEmbeddingSpec, P)>,
        mut materialize: impl FnMut(EmbeddingSpec, P) -> Result<B::Embedding, Error>,
    ) -> Result<Self, Error> {
        if specs.is_empty() {
            return Err(Error::backend(
                "multi-table embedding sum requires at least one table",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let dimensions = specs[0].0.embedding.dimensions;
        for (spec, _) in &specs {
            if spec.name.trim().is_empty() || !names.insert(&spec.name) {
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
            if spec.embedding.dimensions != dimensions {
                return Err(Error::backend(format!(
                    "embedding table {:?} width {} differs from preceding width {}",
                    spec.name, spec.embedding.dimensions, dimensions
                )));
            }
        }
        let tables = specs
            .into_iter()
            .map(|(spec, placement)| {
                Ok(NamedEmbedding {
                    embedding: materialize(spec.embedding, placement)?,
                    name: spec.name,
                    lookup: spec.lookup,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
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
    let geometry = (0..config.attention_schedule().len())
        .map(|layer| {
            topology::AttentionParameters::new(config, config.block_parameter_owner(layer))
                .map(|parameters| parameters.geometry)
        })
        .collect::<Result<Vec<_>, _>>()?;
    cache_layout_with_attention_geometry(config, geometry)
}

/// Declares the complete mutable-state geometry consumed by either resident
/// or bounded-residency execution.
pub fn state_layout<C: Config>(config: &C) -> Result<StateLayout, Error> {
    StateLayout::new(cache_layout(config)?).map_err(Error::backend)
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
        self.validate_header(
            config,
            layers,
            config.vocabulary_size(),
            config.tie_word_embeddings(),
        )?;
        let expected = StateLayout::new(
            cache_layout_with_attention_geometry(
                config,
                self.blocks
                    .iter()
                    .enumerate()
                    .map(|(layer, block)| {
                        topology::AttentionParameters::new(
                            block,
                            config.block_parameter_owner(layer),
                        )
                        .map(|parameters| parameters.geometry)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        )
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        self.validate_state(&expected)
    }

    /// Validates complete local state and ranges using the retained dense slots.
    pub fn validate_with_declaration(
        &self,
        config: &C,
        declaration: &DecoderDeclaration,
    ) -> Result<(), ParallelPlanError>
    where
        C: PartitionedConfig,
    {
        let invalid = |e: Error| ParallelPlanError::InvalidGroup(e.to_string());
        declaration.validate_for(config).map_err(invalid)?;
        self.validate_header(
            config,
            declaration.blocks.len(),
            declaration.statics.embedding_spec().vocabulary,
            declaration.statics.head_spec().is_none(),
        )?;
        let geometry = self
            .blocks
            .iter()
            .enumerate()
            .map(|(layer, local)| {
                declaration.validate_local_policy(config, local)?;
                let old = declaration.blocks[layer].attention.geometry;
                Ok((
                    eredu_nn::AttentionHeadGeometry::new(
                        local.num_attention_heads(),
                        local.num_key_value_heads(),
                        old.key_dimensions(),
                        old.value_dimensions(),
                    )?,
                    local.intermediate_size(),
                ))
            })
            .collect::<Result<Vec<_>, Error>>()
            .map_err(invalid)?;
        let projected = declaration
            .local_projection(config, &geometry)
            .map_err(invalid)?;
        self.validate_state(projected.state_layout())
    }

    fn validate_header(
        &self,
        config: &C,
        layers: usize,
        vocabulary: i32,
        tied: bool,
    ) -> Result<(), ParallelPlanError> {
        if self.architecture_fingerprint != config.architecture_fingerprint()
            || self.blocks.len() != layers
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local decoder geometry belongs to a different normalized configuration"
                    .into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(vocabulary)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (tied, &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(vocabulary)
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
        Ok(())
    }

    fn validate_state(&self, expected: &StateLayout) -> Result<(), ParallelPlanError> {
        if expected != &self.state_layout {
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
    decoder_scopes: &DecoderDeclaration,
    layout: &LocalModelLayout,
    mut local_block: F,
) -> Result<LocalGeometry<C>, ParallelPlanError>
where
    C: PartitionedConfig,
    F: FnMut(&C, usize, &LocalModelLayout) -> Result<C, ParallelPlanError>,
{
    decoder_scopes
        .validate_for(config)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let layers = usize::try_from(config.num_hidden_layers())
        .map_err(|_| ParallelPlanError::InvalidGroup("decoder layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|index| match decoder_scopes {
            scopes
                if scopes
                    .block_parameters(config.block_parameter_owner(index))
                    .feed_forward
                    .is_some() =>
            {
                dense_local_block_config(
                    config,
                    config.block_parameter_owner(index),
                    layout,
                    Some(scopes.block_parameters(config.block_parameter_owner(index))),
                )
                .map_err(|e| ParallelPlanError::InvalidTensor(e.to_string()))
            }
            _ => local_block(config, config.block_parameter_owner(index), layout),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = blocks
        .iter()
        .enumerate()
        .map(|(layer, block)| {
            let old = decoder_scopes
                .block_parameters(config.block_parameter_owner(layer))
                .attention
                .geometry;
            eredu_nn::AttentionHeadGeometry::new(
                block.num_attention_heads(),
                block.num_key_value_heads(),
                old.key_dimensions(),
                old.value_dimensions(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let state_layout = StateLayout::new(
        cache_layout_with_attention_geometry(config, geometry)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let root = config.parameter_root();
    let statics = &decoder_scopes.statics;
    let (embedding_range, output_range) = statics.vocabulary_ranges(
        layout,
        &format!("{root}.embed_tokens"),
        &format!("{root}.norm"),
        "lm_head",
    )?;
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: config.architecture_fingerprint(),
    };
    geometry.validate_with_declaration(config, decoder_scopes)?;
    Ok(geometry)
}

/// Projects invocation cache policies from exact construction-declared head geometry.
/// Logical invocation order and attention windows come from the retained family
/// schedule; parameter sharing does not merge mutable cache owners.
pub fn cache_layout_with_attention_geometry<C: Config>(
    config: &C,
    geometry: impl IntoIterator<Item = eredu_nn::AttentionHeadGeometry>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    let geometry = geometry.into_iter().collect::<Vec<_>>();
    if geometry.len() != layers {
        return Err(Error::backend(format!(
            "decoder cache geometry has {} layers, expected {layers}",
            geometry.len()
        )));
    }
    let policies = config
        .attention_schedule()
        .iter()
        .zip(geometry)
        .map(|(attention, geometry)| {
            if geometry.key_dimensions() != geometry.value_dimensions() {
                return Err(Error::backend(
                    "ordinary decoder cache requires equal key/value head widths",
                ));
            }
            LayerCachePolicy::key_value(
                *attention,
                geometry.key_value_heads(),
                geometry.key_dimensions(),
            )
            .map_err(Error::backend)
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies).map_err(Error::backend)
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

// Type witness for the missing cache in un-cached attention.
enum AbsentAttentionCache {}

impl<T: Tensor> AttentionCache<T> for AbsentAttentionCache {
    fn offset(&self) -> i32 {
        match *self {}
    }
    fn max_size(&self) -> Option<i32> {
        match *self {}
    }
    fn update_for_attention(
        &mut self,
        _keys: T,
        _values: T,
        _context: &T::Context,
    ) -> Result<(T, T), Error> {
        match *self {}
    }
    fn attention(
        &mut self,
        _request: AttentionRequest<'_, T>,
        _context: &T::Context,
    ) -> Result<T, Error> {
        match *self {}
    }
}

/// Hidden-state input for one decoder block.
pub struct AttentionInput<'a, T, C> {
    /// Hidden states shaped `[batch, sequence, hidden]`.
    pub hidden: &'a T,
    /// Optional absolute selected K/V positions, shaped `[batch,sequence,slots]`.
    /// `-1` marks unused slots; architectures own causal eligibility.
    pub selected_positions: Option<&'a T>,
    /// Optional additive or boolean attention mask (selected-slot shape when indexed).
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
    #[parameter(skip)]
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
    /// Immutable query gain equations; positions belong to each invocation.
    #[parameter(skip)]
    pub query_position_scaling: Option<PositionQueryScaling>,
    /// Independent projection of the attention output gate.
    pub output_gate: Option<B::Linear>,
    /// Activation applied to either separately projected or fused-query gates.
    #[parameter(skip)]
    pub output_gate_activation: OutputGateActivation,
    /// Broadcast one output-gate scalar over each query head.
    #[parameter(skip)]
    pub output_gate_per_head: bool,
    /// Normalize complete Q/K projections using per-head scale vectors.
    #[parameter(skip)]
    pub query_key_norm_per_head_weights: bool,
    /// Rotated paired width; zero means the complete head.
    #[parameter(skip)]
    pub rotary_pair_dimensions: i32,
    /// Number of query heads.
    #[parameter(skip)]
    pub query_heads: i32,
    /// Number of key/value heads.
    #[parameter(skip)]
    pub key_value_heads: i32,
    /// Inverse square-root head scaling.
    #[parameter(skip)]
    pub scale: f32,
    /// Optional score cap applied before masking and softmax.
    #[parameter(skip)]
    pub softcap: Option<f32>,
    /// Selected score/probability arithmetic.
    #[parameter(skip)]
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
    #[parameter(skip)]
    pub sliding_window: Option<i32>,
    /// Whether the query projection's second half gates attended values.
    #[parameter(skip)]
    pub query_output_gate: bool,
}

/// Query gain `1 + beta * ln(1 + floor(position / original_max_positions))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionQueryScaling {
    /// Finite nonnegative logarithmic gain.
    pub beta: f32,
    /// Positive original positional extent.
    pub original_max_positions: i32,
}

impl PositionQueryScaling {
    /// Validates the gain before module construction.
    pub fn validate(self) -> Result<(), Error> {
        if !self.beta.is_finite() || self.beta < 0.0 || self.original_max_positions <= 0 {
            return Err(Error::backend("invalid position-dependent query scaling"));
        }
        Ok(())
    }

    /// Gain at a canonical causal position, including transient block positions.
    pub fn gain(self, position: i32) -> Result<f32, Error> {
        self.validate()?;
        if position < 0 {
            return Err(Error::backend("negative query position"));
        }
        Ok(1.0 + self.beta * (1.0 + (position / self.original_max_positions) as f32).ln())
    }

    fn apply<T: Tensor>(
        self,
        queries: T,
        offset: i32,
        sequence: i32,
        context: &T::Context,
    ) -> Result<T, Error> {
        let positions = (0..sequence)
            .map(|position| {
                self.gain(
                    offset
                        .checked_add(position)
                        .ok_or_else(|| Error::backend("query position overflow"))?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let element = queries
            .element_type()
            .ok_or_else(|| Error::backend("query scaling requires known floating arithmetic"))?;
        let gains = T::from_f32_slice(&positions, &[1, 1, sequence, 1], context)?
            .cast_float(element, context)?;
        queries.multiply(&gains, context)
    }
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
    fn gate_attended(
        &self,
        attended: B::Tensor,
        gate: Option<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let Some(gate) = gate else {
            return Ok(attended);
        };
        let gate = self.output_gate_activation.apply::<B>(gate, context)?;
        if self.output_gate_per_head {
            let shape = attended.shape();
            attended
                .reshape(&[shape[0], shape[1], self.query_heads, -1], context)?
                .multiply(&gate.expand_dims(-1, context)?, context)?
                .reshape(&shape, context)
        } else {
            attended.multiply(&gate, context)
        }
    }

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
            query_position_scaling: None,
            output_gate: None,
            output_gate_activation: OutputGateActivation::Sigmoid,
            output_gate_per_head: false,
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
        let parameters = topology::AttentionParameters::new(config, layer)?;
        Self::from_parameters(
            config,
            layer,
            &parameters,
            topology::attention_normalizations(config, layer, parameters.geometry)?,
            topology::attention_sink(config, layer, parameters.geometry)?,
            context,
        )
    }
    fn from_parameters<C: Config>(
        config: &C,
        layer: usize,
        parameters: &topology::AttentionParameters,
        normalizations: [Option<NormalizationConstructionSpec>; 2],
        sinks: Option<topology::AttentionSinkSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config
            .query_position_scaling()
            .map(PositionQueryScaling::validate)
            .transpose()?;
        if config.learned_attention_sinks() {
            crate::operator_requirements::require::<B>(
                "shared decoder attention sinks",
                eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
            )?;
        }
        eredu_nn::validate_logical_parameter_scope(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let geometry = parameters.geometry;
        let head = geometry.key_dimensions();
        let query_heads = geometry.query_heads();
        let key_value_heads = geometry.key_value_heads();
        let [query_normalization, key_normalization] = normalizations;
        let linear = |spec: &LinearSpec| B::linear(spec.clone(), context);
        let policy = config.attention_schedule().get(layer).ok_or_else(|| {
            Error::backend(format!(
                "decoder attention schedule has no policy for layer {layer}"
            ))
        })?;
        let input_projection = match &parameters.input {
            topology::AttentionInputParameters::Split { query, key, value } => match value {
                Some(value) => AttentionInputProjection::Split {
                    query: linear(query)?,
                    key: linear(key)?,
                    value: linear(value)?,
                },
                None => AttentionInputProjection::ExternalValue {
                    query: linear(query)?,
                    key: linear(key)?,
                },
            },
            topology::AttentionInputParameters::Fused { projection, layout } => {
                AttentionInputProjection::Fused(FusedAttentionProjection {
                    projection: linear(projection)?,
                    layout: layout.clone(),
                })
            }
        };
        Ok(Self {
            query_position_scaling: config.query_position_scaling(),
            output_gate: parameters
                .gate
                .as_ref()
                .map(|(spec, _)| linear(spec))
                .transpose()?,
            output_gate_activation: parameters
                .gate
                .as_ref()
                .map_or(OutputGateActivation::Sigmoid, |(_, activation)| *activation),
            output_gate_per_head: false,
            query_key_norm_per_head_weights: config.query_key_norm_per_head_weights(),
            rotary_pair_dimensions: config.rotary_pair_dimensions(),
            query_heads,
            key_value_heads,
            scale: config.attention_scale(),
            softcap: config.attention_softcap(),
            arithmetic: config.attention_arithmetic(),
            input_projection,
            output: linear(&parameters.output)?,
            sinks: sinks
                .map(|spec| Parameter::unloaded(spec.weight, &[spec.heads], context))
                .transpose()?,
            query_norm: query_normalization
                .map(|spec| B::normalization(spec, context))
                .transpose()?,
            key_norm: key_normalization
                .map(|spec| B::normalization(spec, context))
                .transpose()?,
            rotary: config
                .rotary_enabled()
                .then(|| B::rotary(config.layer_rotary_spec(layer, head), context))
                .transpose()?,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(Error::backend)?,
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
        selected_positions: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if selected_positions.is_some() && (self.softcap.is_some() || self.sliding_window.is_some())
        {
            return Err(Error::backend(
                "indexed selection requires uncapped full-history attention",
            ));
        }
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
        let queries = match self.query_position_scaling {
            Some(scaling) => scaling.apply(
                queries,
                match rotary_position {
                    Some(RotaryPosition::Offset(position)) => position,
                    _ => offset,
                },
                sequence,
                context,
            )?,
            None => queries,
        };
        let (keys, values) = match cache.as_mut() {
            Some(cache) => cache.update_for_attention(keys, values, context)?,
            None => (keys, values),
        };
        let sinks = self.sinks.as_ref().map(Parameter::as_ref);
        if let Some(positions) = selected_positions {
            let request = eredu_nn::IndexedAttentionInput {
                queries: &queries,
                keys: &keys,
                values: &values,
                key_position_offset: offset + sequence - keys.dim(2),
                selected_positions: positions,
                validity: None,
                mask,
                local: None,
                scale: self.scale,
                arithmetic: self.arithmetic,
                sinks,
            };
            let attended = if let Some(cache) = cache {
                cache.indexed_attention::<B>(request, context)?
            } else {
                B::indexed_attention(request, context)?
            };
            let attended = attended
                .transpose_axes(&[0, 2, 1, 3], context)?
                .reshape(&[batch, sequence, -1], context)?;
            return self.gate_attended(attended, output_gate, context);
        }
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
            return self.gate_attended(attended, output_gate, context);
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
        self.gate_attended(attended, output_gate, context)
    }

    /// Executes grouped-query attention and its output projection.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_value(input, None, context)
    }

    /// Runs attention over the complete input without allocating or updating a cache.
    pub(crate) fn forward_uncached(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward::<AbsentAttentionCache>(
            AttentionInput {
                hidden,
                mask,
                selected_positions: None,
                cache: None,
                allow_sliding_prefill: false,
                rotary_position: None,
            },
            context,
        )
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
            input.selected_positions,
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
            input.selected_positions,
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
            input.selected_positions,
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
    #[parameter(skip)]
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

/// Shared dense gated feed-forward network.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: NeuralBackend> {
    /// Split or fused gate/up input projection.
    pub input_projection: GatedInputProjection<B>,
    /// Down projection.
    pub down: B::Linear,
    /// Resolved activation, clipping and scaling before the down projection.
    #[parameter(skip)]
    pub policy: GatedProductPolicy,
}

impl<B: NeuralBackend> Mlp<B> {
    /// Assembles a dense gated network from independent gate/up projections.
    pub fn from_parts(
        gate: B::Linear,
        up: B::Linear,
        down: B::Linear,
        policy: GatedProductPolicy,
    ) -> Self {
        Self {
            input_projection: GatedInputProjection::Split { gate, up },
            down,
            policy,
        }
    }

    /// Builds an unloaded dense gated network for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_parameters(&topology::gated_parameters(config, layer)?, context)
    }
    pub(crate) fn from_parameters(
        parameters: &eredu_nn::GatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        parameters.logical_parameters()?;
        let input_projection = match parameters {
            eredu_nn::GatedProductSpec::Split(spec) => GatedInputProjection::Split {
                gate: B::linear(spec.gate.clone(), context)?,
                up: B::linear(spec.up.clone(), context)?,
            },
            eredu_nn::GatedProductSpec::Fused(spec) => {
                let layout = spec.layout()?;
                GatedInputProjection::Fused(FusedGatedProjection {
                    projection: B::linear(spec.gate_up.clone(), context)?,
                    layout,
                })
            }
        };
        Ok(Self {
            input_projection,
            down: B::linear(parameters.down().clone(), context)?,
            policy: parameters.policy(),
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
        B::gated_product(gate, up, self.policy, context)
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

/// Attention and feed-forward parameter body of an RMS-pre-norm decoder block.
/// Terminal invocation normalization is borrowed by the equation and is not owned here.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlockBody<B: NeuralBackend, F = Mlp<B>> {
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
}

/// One decoder invocation with its body and optional terminal normalization.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: NeuralBackend, F = Mlp<B>> {
    /// Attention, feed-forward and residual-normalization parameters.
    pub body: TransformerBlockBody<B, F>,
    /// Invocation-specific normalization of the completed block output.
    pub output_norm: Option<B::Normalization>,
}

/// Independently materialized decoder equation resource.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum DecoderUnit<B: NeuralBackend, F = Mlp<B>> {
    /// Canonical attention/feed-forward parameters, without a terminal.
    Body(TransformerBlockBody<B, F>),
    /// Canonical stateless terminal normalization.
    Terminal(B::Normalization),
}

impl<B: NeuralBackend> TransformerBlockBody<B, Mlp<B>> {
    /// Constructs an unloaded dense body without invocation terminal parameters.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_factory::<C, DenseBlockFactory>(
            config,
            config,
            layer,
            &topology::BlockParameters::new(config, layer)?,
            context,
        )
    }
}

impl<B: NeuralBackend, F> TransformerBlockBody<B, F> {
    /// Constructs a body from shared attention/residual policy and a typed
    /// feed-forward factory. Global routing and rank-local bank geometry remain distinct.
    pub(crate) fn from_factory<C: Config, P: BlockFactory<B, C, FeedForward = F>>(
        global: &C,
        local: &C,
        layer: usize,
        spec: &topology::BlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameters =
            prepare_projection::<B, C, P>(global, local, layer, spec.feed_forward.as_ref(), None)?;
        validate_prepared_projection(spec, &parameters)?;
        Self::from_prepared::<C, P>(global, local, layer, spec, &parameters, context)
    }

    pub(crate) fn from_prepared<C: Config, P: BlockFactory<B, C, FeedForward = F>>(
        global: &C,
        local: &C,
        layer: usize,
        spec: &topology::BlockParameters,
        parameters: &P::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        P::validate_partitioned(global, local)?;
        Self::construct_from_specs(
            spec.normalizations.clone(),
            context,
            || {
                Attention::from_parameters(
                    local,
                    layer,
                    &spec.attention,
                    spec.query_key.clone(),
                    spec.sinks.clone(),
                    context,
                )
            },
            || Ok(()),
            || P::build_feed_forward(global, local, layer, parameters, context),
        )
        .map(|(body, ())| body)
    }

    fn construct_from_specs<T>(
        normalizations: topology::BlockNormalizations,
        context: &<B::Tensor as Tensor>::Context,
        attention: impl FnOnce() -> Result<Attention<B>, Error>,
        terminal: impl FnOnce() -> Result<T, Error>,
        feed_forward: impl FnOnce() -> Result<F, Error>,
    ) -> Result<(Self, T), Error> {
        let attention_output_norm = normalizations
            .attention_output
            .map(|spec| B::normalization(spec, context))
            .transpose()?;
        let feed_forward_output_norm = normalizations
            .feed_forward_output
            .map(|spec| B::normalization(spec, context))
            .transpose()?;
        let terminal = terminal()?;
        Ok((
            Self {
                attention_output_norm,
                feed_forward_output_norm,
                self_attention: attention()?,
                mlp: feed_forward()?,
                input_norm: B::normalization(normalizations.input, context)?,
                post_attention_norm: B::normalization(normalizations.post_attention, context)?,
            },
            terminal,
        ))
    }
}

impl<B: NeuralBackend> TransformerBlock<B, Mlp<B>> {
    /// Materializes exact retained ordinary block slots; configuration supplies execution policy.
    pub(crate) fn from_parameters<C: Config>(
        config: &C,
        layer: usize,
        spec: &topology::BlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let feed_forward = spec
            .feed_forward
            .as_ref()
            .ok_or_else(|| Error::backend("ordinary block has no gated declaration"))?;
        spec.logical_parameters()?;
        let (body, output_norm) = TransformerBlockBody::construct_from_specs(
            spec.normalizations.clone(),
            context,
            || {
                Attention::from_parameters(
                    config,
                    layer,
                    &spec.attention,
                    spec.query_key.clone(),
                    spec.sinks.clone(),
                    context,
                )
            },
            || {
                spec.terminal
                    .clone()
                    .map(|spec| B::normalization(spec, context))
                    .transpose()
            },
            || Mlp::from_parameters(feed_forward, context),
        )?;
        Ok(Self { body, output_norm })
    }
    /// Constructs one unloaded dense body and its optional terminal normalization.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_factory::<C, DenseBlockFactory>(
            config,
            config,
            layer,
            &topology::BlockParameters::new(config, layer)?,
            context,
        )
    }
}

impl<B: NeuralBackend, F> TransformerBlock<B, F> {
    /// Assembles the shared body and declared terminal without changing their
    /// independent construction interfaces or canonical parameter names.
    pub(crate) fn from_factory<C: Config, P: BlockFactory<B, C, FeedForward = F>>(
        global: &C,
        local: &C,
        layer: usize,
        spec: &topology::BlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameters =
            prepare_projection::<B, C, P>(global, local, layer, spec.feed_forward.as_ref(), None)?;
        validate_prepared_projection(spec, &parameters)?;
        Self::from_prepared::<C, P>(global, local, layer, spec, &parameters, context)
    }

    pub(crate) fn from_prepared<C: Config, P: BlockFactory<B, C, FeedForward = F>>(
        global: &C,
        local: &C,
        layer: usize,
        spec: &topology::BlockParameters,
        parameters: &P::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        P::validate_partitioned(global, local)?;
        let (body, output_norm) = TransformerBlockBody::construct_from_specs(
            spec.normalizations.clone(),
            context,
            || {
                Attention::from_parameters(
                    local,
                    layer,
                    &spec.attention,
                    spec.query_key.clone(),
                    spec.sinks.clone(),
                    context,
                )
            },
            || {
                spec.terminal
                    .clone()
                    .map(|spec| B::normalization(spec, context))
                    .transpose()
            },
            || P::build_feed_forward(global, local, layer, parameters, context),
        )?;
        Ok(Self { body, output_norm })
    }
}

impl<B, F> TransformerBlockBody<B, F>
where
    B: NeuralBackend,
    F: DecoderProjectionOperator<B>,
{
    /// Executes the shared normalization, attention, cache, and residual lifecycle
    /// with family-owned projection callbacks. The same provider is borrowed
    /// sequentially by value and feed-forward execution.
    fn forward_projections<C, D, V, H>(
        &mut self,
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        output_adapter: Option<&mut eredu_nn::low_rank::LowRankProjection<B>>,
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
            selected_positions: input.selected_positions,
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
            attention_input.selected_positions,
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
        let attention = match output_adapter {
            Some(adapter) => {
                let element = attention.element_type().ok_or_else(|| {
                    Error::backend("unknown base projection precision for low-rank delta")
                })?;
                let combined =
                    attention.add(&adapter.forward(&attended, parallel, context)?, context)?;
                // PEFT adds in the adapter's promoted precision and restores the
                // base output type once, after the complete additive update.
                if combined.element_type() == Some(element) {
                    combined
                } else {
                    combined.cast_float(element, context)?
                }
            }
            None => attention,
        };
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
        match output_norm {
            Some(norm) => norm.forward(&hidden, context),
            None => Ok(hidden),
        }
    }
}

impl<B, F> TransformerBlockBody<B, F>
where
    B: NeuralBackend,
    F: DecoderProjectionOperator<B>,
{
    /// Executes this body with replicated projections and a separately borrowed terminal.
    /// The caller owns both resource lifetimes through ordinary completion.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_projections(
            output_norm,
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            None,
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
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_projections(
            output_norm,
            input,
            None,
            context,
            instrumentation,
            None,
            &mut (),
            |policy, input, _, context, instrumentation| {
                policy.project_values_observed(input, context, instrumentation)
            },
            |policy, input, _, context, instrumentation| {
                policy.forward_feed_forward_observed(input, context, instrumentation)
            },
        )
    }

    /// Borrows a separate low-rank attention-output delta for this invocation.
    /// Base projection storage and all cache equations remain shared. Parallel
    /// rank activations are reduced before the replicated adapter up projection.
    pub fn forward_with_output_adapter_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        adapter: Option<&mut eredu_nn::low_rank::LowRankProjection<B>>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        F: TensorParallelProjectionOperator<B>,
    {
        self.forward_projections(
            output_norm,
            input,
            parallel,
            context,
            instrumentation,
            adapter,
            &mut (),
            |policy, input, _, context, instrumentation| {
                policy.project_values_observed(input, context, instrumentation)
            },
            |policy, input, _, context, instrumentation| match parallel {
                Some(parallel) => policy.forward_feed_forward_parallel_observed(
                    input,
                    parallel,
                    context,
                    instrumentation,
                ),
                None => policy.forward_feed_forward_observed(input, context, instrumentation),
            },
        )
    }

    /// Delegates feed-forward computation while retaining the shared block lifecycle.
    pub fn forward_with_feed_forward<C, H>(
        &mut self,
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_projections(
            output_norm,
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            None,
            &mut (),
            |policy, input, _, context, _| policy.project_values(input, context),
            |policy, input, _, context, _| feed_forward(policy, input, context),
        )
    }

    /// Executes column partitions and reduced row projections.
    pub fn forward_tensor_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        F: TensorParallelProjectionOperator<B>,
    {
        self.forward_tensor_parallel_observed(
            output_norm,
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
        output_norm: Option<&mut B::Normalization>,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        F: TensorParallelProjectionOperator<B>,
    {
        self.forward_projections(
            output_norm,
            input,
            Some(parallel),
            context,
            instrumentation,
            None,
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
        output_norm: Option<&mut B::Normalization>,
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
            output_norm,
            input,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
            None,
            &mut (),
            |policy, input, _, context, _| policy.project_values(input, context),
            |policy, input, _, context, _| feed_forward(policy, input, context),
        )
    }

    /// Routes both replaceable projection stages through one runtime provider.
    pub fn forward_routed<C, P>(
        &mut self,
        output_norm: Option<&mut B::Normalization>,
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
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            output_norm,
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            None,
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
        output_norm: Option<&mut B::Normalization>,
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
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            output_norm,
            input,
            None,
            context,
            instrumentation,
            None,
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
        output_norm: Option<&mut B::Normalization>,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            output_norm,
            input,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
            None,
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
        output_norm: Option<&mut B::Normalization>,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_projections(
            output_norm,
            input,
            Some(parallel),
            context,
            instrumentation,
            None,
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

/// Derives the complete dense-decoder parameter topology from normalized
/// configuration without constructing backend modules.
pub fn dense_parameter_description(
    config: &impl Config,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    DecoderDeclaration::new(config)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
        .parameter_description(config, ParameterEncoding::Stored)
}

/// Logical dense parameter declaration before physical encoding expansion.
/// Construction specifications supply geometry; ownership, invocation sharing
/// and semantic partition rules remain architecture-authored.
pub fn dense_logical_parameter_description(
    config: &impl Config,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    DecoderDeclaration::new(config)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
        .parameter_description(config, ParameterEncoding::Logical)
}

/// Neutral router and packed-bank declaration for one ordinary routed block.
pub(crate) struct RoutedBlockParameters {
    pub selector_name: String,
    pub bank_name: String,
    pub parameters: eredu_nn::RoutedGatedProductSpec,
}

/// Geometry projection applied to a source-free parameter declaration.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ParameterEncoding {
    /// Logical tensor geometry with retained encoding declarations, before physical
    /// expansion, alignment and format companions.
    Logical,
    /// Stored tensor geometry, including exact encoding companions and alignment.
    Stored,
}

impl ParameterEncoding {
    pub(crate) fn format(self, format: LinearFormat) -> LinearFormat {
        match self {
            Self::Logical => LinearFormat::Dense,
            Self::Stored => format,
        }
    }

    pub(crate) fn parameter_group(
        self,
        name: impl Into<String>,
        role: ParameterRole,
        units: Option<usize>,
        spec: &(impl DeclareLogicalParameters + ?Sized),
        sharding: impl FnMut(&eredu_nn::LogicalParameterDeclaration<'_>) -> MemberSharding,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        match self {
            Self::Stored => {
                eredu_runtime::logical_parameter_group(name, role, units, spec, sharding)
            }
            Self::Logical => {
                let declarations = spec
                    .logical_parameters()
                    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
                eredu_nn::validate_logical_parameter_scope(&declarations)
                    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
                let members = eredu_runtime::logical_parameter_members(&declarations, sharding)?;
                match units {
                    Some(units) => ParameterGroupSpec::partitioned(name, role, units, members),
                    None => ParameterGroupSpec::new(name, role, members),
                }
            }
        }
    }

    fn projection_group(
        self,
        name: impl Into<String>,
        role: ParameterRole,
        units: usize,
        projections: &[(&LinearSpec, MemberSharding)],
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        let mut declarations = Vec::new();
        for (spec, _) in projections {
            declarations.extend(
                spec.logical_parameters()
                    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
            );
        }
        self.parameter_group(name, role, Some(units), &declarations, |declaration| {
            let (_, sharding) = projections
                .iter()
                .find(|(spec, _)| {
                    spec.weight.id == declaration.parameter.id
                        || spec
                            .bias
                            .as_ref()
                            .is_some_and(|bias| bias.id == declaration.parameter.id)
                })
                .expect("projection owns its declared parameter");
            if declaration.role == eredu_nn::LogicalParameterRole::LinearWeight
                || matches!(
                    sharding,
                    MemberSharding::Partitioned { axis: 0 }
                        | MemberSharding::PartitionedSegments { axis: 0, .. }
                )
            {
                sharding.clone()
            } else {
                MemberSharding::Replicated
            }
        })
    }
}

/// Projects ordinary routed blocks into the common decoder topology.
pub(crate) fn routed_parameter_description(
    config: &impl Config,
    encoding: ParameterEncoding,
    declaration: &DecoderDeclaration,
    block: impl Fn(usize) -> Result<RoutedBlockParameters, Error>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let invalid = |error: Error| ParallelPlanError::InvalidGroup(error.to_string());
    config.validate_config().map_err(invalid)?;
    declaration.validate_for(config).map_err(invalid)?;
    let layers = usize::try_from(config.num_hidden_layers())
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let mut groups = Vec::with_capacity(layers);
    for layer in 0..layers {
        let block = block(layer).map_err(invalid)?;
        block.parameters.validate().map_err(invalid)?;
        let selector = &block.parameters.selector;
        let bank = &block.parameters.experts;
        if selector.selection().group_count() != bank.group_count()
            || bank.input_dimensions() != config.hidden_size()
            || bank.output_dimensions() != config.hidden_size()
        {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "routed decoder unit {layer} has incompatible selector and bank geometry"
            )));
        }
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = bank.layout() else {
            return Err(ParallelPlanError::InvalidGroup(
                "ordinary routed decoder topology requires packed expert projections".into(),
            ));
        };
        let units = eredu_runtime::aligned_partition_units(
            &block.bank_name,
            bank.intermediate_dimensions() as usize,
            1,
            crate::linear_format::input_partition_alignment(
                encoding.format(down.format().encoding()),
            ) as usize,
        )?;
        let intermediate = bank.intermediate_dimensions() as usize;
        groups.push(vec![
            encoding.parameter_group(
                block.selector_name,
                ParameterRole::Replicated,
                None,
                selector,
                |_| MemberSharding::Replicated,
            )?,
            encoding.parameter_group(
                block.bank_name,
                ParameterRole::ExpertIntermediate,
                Some(units),
                bank,
                |declaration| {
                    let parameter = &declaration.parameter.id;
                    if parameter == &gate_up.weight().id
                        || gate_up.bias().is_some_and(|bias| parameter == &bias.id)
                    {
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..intermediate, intermediate..2 * intermediate],
                        }
                    } else if declaration.role == eredu_nn::LogicalParameterRole::LinearWeight {
                        MemberSharding::Partitioned { axis: 2 }
                    } else {
                        MemberSharding::Replicated
                    }
                },
            )?,
        ]);
    }
    parameter_description_with_encoding(
        config,
        |layer| Ok(Some(groups[layer].clone())),
        true,
        ParameterConstruction::Decoder(declaration),
        encoding,
    )
}

/// Derives block and final-normalization topology for supplied-embedding execution,
/// without declaring a vocabulary table or a language-model output head.
pub(crate) fn embedded_parameter_description(
    config: &impl Config,
    declaration: &topology::EmbeddedParameters,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    declaration
        .validate_for(config)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    parameter_description_with_encoding(
        config,
        |_| Ok(None),
        false,
        ParameterConstruction::Embedded(declaration),
        ParameterEncoding::Stored,
    )
}

enum ParameterConstruction<'a> {
    Embedded(&'a topology::EmbeddedParameters),
    Decoder(&'a DecoderDeclaration),
}
impl ParameterConstruction<'_> {
    fn blocks(&self) -> &[topology::BlockParameters] {
        match self {
            Self::Embedded(spec) => &spec.blocks,
            Self::Decoder(spec) => &spec.blocks,
        }
    }
    fn normalization(&self) -> &NormalizationConstructionSpec {
        match self {
            Self::Embedded(spec) => &spec.normalization,
            Self::Decoder(spec) => spec.statics.normalization_spec(),
        }
    }
}

fn parameter_description_with_encoding(
    config: &impl Config,
    replacement_groups: impl Fn(usize) -> Result<Option<Vec<ParameterGroupSpec>>, ParallelPlanError>,
    include_vocabulary: bool,
    declaration: ParameterConstruction<'_>,
    encoding: ParameterEncoding,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    config
        .validate_config()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let schedule = DecoderSchedule::new(config)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let graph = ExecutionGraph::chain([TEXT_DECODER_EXECUTION_GROUP])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [schedule.len()])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let group_id = layout
        .group_id(0)
        .expect("dense decoder layout contains its text group")
        .clone();
    let root = config.parameter_root();
    let mut owned = if include_vocabulary {
        let ParameterConstruction::Decoder(spec) = &declaration else {
            return Err(ParallelPlanError::InvalidGroup(
                "vocabulary requires a decoder declaration".into(),
            ));
        };
        let statics = &spec.statics;
        statics
            .pinned_parameter_groups(
                encoding,
                &format!("{root}.embed_tokens"),
                &format!("{root}.norm"),
                "lm_head",
            )?
            .into_iter()
            .map(|(role, group)| {
                let owner = if role == "embedding" && statics.head_spec().is_none() {
                    ParameterGroupOwner::static_any_of(["embedding", "output"])
                } else {
                    ParameterGroupOwner::static_role(role)
                };
                OwnedParameterGroupSpec::new(owner, group)
            })
            .collect::<Vec<_>>()
    } else {
        vec![OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("norm"),
            encoding.parameter_group(
                format!("{root}.norm"),
                ParameterRole::Replicated,
                None,
                declaration.normalization(),
                |_| MemberSharding::Replicated,
            )?,
        )]
    };

    for (owner, consumers) in schedule.resources() {
        let operation = schedule
            .operation(owner)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let layer = operation.layer();
        let block = &declaration.blocks()[layer];
        if let DecoderOperation::Terminal { .. } = operation {
            let name = config
                .block_output_normalization(layer)
                .expect("declared terminal");
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit_consumers(group_id.clone(), owner, consumers),
                encoding.parameter_group(
                    name.trim_end_matches(".weight"),
                    ParameterRole::Replicated,
                    None,
                    &block.terminal.clone().ok_or_else(|| {
                        ParallelPlanError::InvalidGroup(
                            "terminal operation has no declared normalization".into(),
                        )
                    })?,
                    |_| MemberSharding::Replicated,
                )?,
            ));
            continue;
        }
        let prefix = format!("{root}.layers.{layer}");
        let attention_prefix = format!("{prefix}.{}", fields.attention);
        let attention_specs = &block.attention;
        let head_partition = attention_partition::AttentionPartition::from_parameters(
            &attention_specs,
            config.attention_value_format(layer),
        )?;
        let attention_units = head_partition.preferred_units();
        let attention_projections = attention_specs.partitioned_projections();
        let attention = encoding.projection_group(
            format!("{attention_prefix}.projections"),
            ParameterRole::AttentionHeads,
            attention_units,
            &attention_projections,
        )?;
        let attention_format = |name: &str| {
            attention_specs
                .projections()
                .into_iter()
                .find(|spec| spec.weight.id.as_str() == name)
                .map_or(LinearFormat::Dense, |spec| {
                    encoding.format(spec.format.encoding())
                })
        };

        let feed_forward_groups = match replacement_groups(layer)? {
            Some(groups) => groups,
            None => {
                let feed_forward_prefix = format!("{prefix}.{}", fields.feed_forward);
                let parameters = block.feed_forward.as_ref().ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(
                        "ordinary block has no gated declaration".into(),
                    )
                })?;
                let output = parameters.down();
                let intermediate = output.input as usize;
                let output_format = encoding.format(output.format.encoding());
                let feed_forward_units = crate::linear_format::input_partition_units(
                    &feed_forward_prefix,
                    intermediate,
                    1,
                    output_format,
                )?;
                let mut feed_forward_projections = Vec::new();
                for spec in parameters.projections() {
                    let sharding = if spec.weight.id == output.weight.id {
                        MemberSharding::Partitioned { axis: 1 }
                    } else if matches!(&parameters, eredu_nn::GatedProductSpec::Fused(_)) {
                        MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: vec![0..intermediate, intermediate..spec.output as usize],
                        }
                    } else {
                        MemberSharding::Partitioned { axis: 0 }
                    };
                    feed_forward_projections.push((spec, sharding));
                }
                let feed_forward = encoding.projection_group(
                    format!("{feed_forward_prefix}.projections"),
                    ParameterRole::FeedForwardIntermediate,
                    feed_forward_units,
                    &feed_forward_projections,
                )?;
                vec![crate::linear_format::dense_ffn_partition_tail(
                    feed_forward,
                    intermediate,
                    output_format,
                    |name| {
                        parameters
                            .projections()
                            .find(|spec| spec.weight.id.as_str() == name)
                            .map_or(LinearFormat::Dense, |spec| {
                                encoding.format(spec.format.encoding())
                            })
                    },
                )?]
            }
        };
        let normalization_group = |spec: &NormalizationConstructionSpec| {
            let declarations = spec
                .logical_parameters()
                .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
            let name = declarations
                .first()
                .ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(
                        "residual normalization must declare its gain".into(),
                    )
                })?
                .parameter
                .id
                .as_str();
            encoding.parameter_group(
                name.trim_end_matches(".weight"),
                ParameterRole::Replicated,
                None,
                &declarations,
                |_| MemberSharding::Replicated,
            )
        };
        let residual = &block.normalizations;
        let input_norm = normalization_group(&residual.input)?;
        let post_attention_norm = normalization_group(&residual.post_attention)?;
        let mut unit_groups = vec![attention];
        if let Some(spec) = block.sinks.clone() {
            unit_groups.push(encoding.parameter_group(
                spec.weight.id.as_str(),
                ParameterRole::AttentionHeads,
                Some(spec.heads as usize),
                &spec,
                |_| MemberSharding::Partitioned { axis: 0 },
            )?);
        }
        let normalizations = block.query_key.clone();
        for (spec, (field, heads)) in normalizations.into_iter().zip([
            (
                fields.attention_query_norm,
                attention_specs.geometry.query_heads() as usize,
            ),
            (
                fields.attention_key_norm,
                attention_specs.geometry.key_value_heads() as usize,
            ),
        ]) {
            let Some(spec) = spec else {
                continue;
            };
            let name = format!("{attention_prefix}.{field}");
            let owned = config.query_key_norm_per_head_weights();
            unit_groups.push(encoding.parameter_group(
                name,
                if owned {
                    ParameterRole::AttentionHeads
                } else {
                    ParameterRole::Replicated
                },
                owned.then_some(heads),
                &spec,
                |_| {
                    if owned {
                        MemberSharding::Partitioned { axis: 0 }
                    } else {
                        MemberSharding::Replicated
                    }
                },
            )?);
        }
        for spec in residual
            .attention_output
            .iter()
            .chain(residual.feed_forward_output.iter())
        {
            unit_groups.push(normalization_group(spec)?);
        }
        unit_groups.extend([input_norm, post_attention_norm]);
        for group in unit_groups {
            let group = head_partition.apply(group, attention_format)?;
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit_consumers(
                    group_id.clone(),
                    owner,
                    consumers.iter().copied(),
                ),
                group,
            ));
        }
        for group in feed_forward_groups {
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit_consumers(
                    group_id.clone(),
                    owner,
                    consumers.iter().copied(),
                ),
                head_partition.apply(group, |name| encoding.format(config.linear_format(name)))?,
            ));
        }
    }

    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

/// Pinned modules shared by resident and bounded-residency execution.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: NeuralBackend, D: DecoderBoundary<B> = NormalizedBoundary<B>> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Architecture-owned input expansion and residual readout.
    pub boundary: D,
    /// Optional untied vocabulary projection.
    pub lm_head: Option<B::Linear>,
}

impl<B: NeuralBackend, D: DecoderBoundary<B> + Clone> Clone for StaticModules<B, D> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            boundary: self.boundary.clone(),
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
    /// Complete embedding physical encoding, including its declared companions.
    pub embedding_format: LinearFormat,
    /// Complete output-head physical encoding.
    pub head_format: LinearFormat,
    /// Whether output logits reuse the embedding table.
    pub tied_head: bool,
}

/// Validated ordinary pinned construction, retaining the full normalization policy.
#[derive(Debug, Clone)]
pub(crate) struct StaticModuleConstruction {
    embedding: EmbeddingSpec,
    head: Option<LinearSpec>,
    normalization: NormalizationConstructionSpec,
}

impl DeclareLogicalParameters for StaticModuleConstruction {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut declarations = self.embedding.logical_parameters()?;
        declarations.extend(self.normalization.logical_parameters()?);
        if let Some(head) = &self.head {
            declarations.extend(head.logical_parameters()?);
        }
        Ok(declarations)
    }
}

impl StaticModuleConstruction {
    pub(crate) fn embedding_spec(&self) -> &EmbeddingSpec {
        &self.embedding
    }
    pub(crate) fn head_spec(&self) -> Option<&LinearSpec> {
        self.head.as_ref()
    }
    /// A tied projection retains the embedding identity and encoding without adding a slot.
    pub(crate) fn output_projection_spec(&self) -> std::borrow::Cow<'_, LinearSpec> {
        match &self.head {
            Some(head) => std::borrow::Cow::Borrowed(head),
            None => std::borrow::Cow::Owned(self.embedding.output_projection()),
        }
    }

    pub(crate) fn normalization_spec(&self) -> &NormalizationConstructionSpec {
        &self.normalization
    }

    pub(crate) fn new(
        embedding: EmbeddingSpec,
        head: Option<LinearSpec>,
        normalization: NormalizationConstructionSpec,
    ) -> Result<Self, Error> {
        validate_vocabulary_specs(&embedding, head.as_ref())?;
        normalization.validate()?;
        if normalization.dimensions != embedding.dimensions {
            return Err(Error::backend(
                "final normalization and embedding widths differ",
            ));
        }
        let construction = Self {
            embedding,
            head,
            normalization,
        };
        eredu_nn::validate_logical_parameter_topology(&construction.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(construction)
    }

    /// Projects pinned roles from construction declarations and their encoding extensions.
    pub(crate) fn pinned_parameter_groups(
        &self,
        encoding: ParameterEncoding,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
        match &self.normalization.scale {
            eredu_nn::NormalizationScale::Learned(_)
            | eredu_nn::NormalizationScale::LearnedOffset { .. } => (),
            eredu_nn::NormalizationScale::Unit => {
                return Err(ParallelPlanError::InvalidGroup(
                    "pinned normalization catalog requires a learned scale".into(),
                ));
            }
        };
        let mut groups = vec![
            (
                "embedding",
                encoding.parameter_group(
                    embedding_group,
                    ParameterRole::Vocabulary,
                    None,
                    &self.embedding,
                    |_| MemberSharding::Balanced { axis: 0 },
                )?,
            ),
            (
                "norm",
                encoding.parameter_group(
                    normalization_group,
                    ParameterRole::Replicated,
                    None,
                    &self.normalization,
                    |_| MemberSharding::Replicated,
                )?,
            ),
        ];
        if let Some(head) = &self.head {
            groups.push((
                "output",
                encoding.parameter_group(
                    output_group,
                    ParameterRole::Vocabulary,
                    None,
                    head,
                    |_| MemberSharding::Balanced { axis: 0 },
                )?,
            ));
        }
        Ok(groups)
    }

    /// Projects only vocabulary resources physically retained by a selected layout.
    pub(crate) fn selected_vocabulary_ranges(
        &self,
        layout: &LocalModelLayout,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<
        (
            Option<VocabularyParallelRange>,
            Option<VocabularyParallelRange>,
        ),
        ParallelPlanError,
    > {
        let groups = self.pinned_parameter_groups(
            ParameterEncoding::Stored,
            embedding_group,
            normalization_group,
            output_group,
        )?;
        let global_vocabulary = self.embedding.vocabulary as usize;
        let range = |role| -> Result<_, ParallelPlanError> {
            let Some((_, group)) = groups.iter().find(|(resource, _)| *resource == role) else {
                return Ok(None);
            };
            Ok(eredu_runtime::balanced_parameter_group_range_if_present(
                group,
                layout,
                0,
                global_vocabulary,
            )?
            .map(|local| VocabularyParallelRange {
                global_vocabulary,
                local,
            }))
        };
        Ok((range("embedding")?, range("output")?))
    }

    /// Projects exact vocabulary intervals from declared physical storage.
    /// Runtime validates row companions and replicas; tied readout retains one owner.
    pub(crate) fn vocabulary_ranges(
        &self,
        layout: &LocalModelLayout,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<(VocabularyParallelRange, Option<VocabularyParallelRange>), ParallelPlanError> {
        let (embedding, output) = self.selected_vocabulary_ranges(
            layout,
            embedding_group,
            normalization_group,
            output_group,
        )?;
        let required = |range: Option<VocabularyParallelRange>, role| {
            range.ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!(
                    "static declaration has no local {role} resource"
                ))
            })
        };
        Ok((
            required(embedding, "embedding")?,
            if self.head.is_none() {
                None
            } else {
                Some(required(output, "output")?)
            },
        ))
    }
}

impl StaticModuleSpec {
    pub(crate) fn construction_specs(&self) -> Result<StaticModuleConstruction, Error> {
        let embedding = EmbeddingSpec {
            vocabulary: self.vocabulary,
            dimensions: self.hidden_size,
            weight: ParameterSpec::trainable(&self.embedding_weight).map_err(Error::backend)?,
            format: crate::linear_format::standard_linear_format(
                &self.embedding_weight,
                self.embedding_format,
            )?,
        };
        let head = (!self.tied_head)
            .then(|| -> Result<_, Error> {
                Ok(LinearSpec {
                    input: self.hidden_size,
                    output: self.vocabulary,
                    weight: ParameterSpec::trainable(&self.head_weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &self.head_weight,
                        self.head_format,
                    )?,
                })
            })
            .transpose()?;
        let weight =
            ParameterSpec::trainable(&self.normalization_weight).map_err(Error::backend)?;
        let normalization = NormalizationConstructionSpec {
            reduction: eredu_nn::NormalizationReduction::Native,
            groups: self.normalization_groups,
            dimensions: self.hidden_size,
            epsilon: self.normalization_epsilon,
            scale: if self.normalization_offset == 0.0 {
                eredu_nn::NormalizationScale::Learned(weight)
            } else {
                eredu_nn::NormalizationScale::LearnedOffset {
                    weight,
                    offset: self.normalization_offset,
                }
            },
        };
        StaticModuleConstruction::new(embedding, head, normalization)
    }

    /// Shared ordinary static-module specification before backend binding.
    pub(crate) fn from_config<C: Config>(config: &C) -> Self {
        let embedding_name = format!("{}.embed_tokens.weight", config.parameter_root());
        Self {
            normalization_groups: config.normalization_groups(),
            embedding_weight: embedding_name.clone(),
            normalization_weight: format!("{}.norm.weight", config.parameter_root()),
            head_weight: "lm_head.weight".into(),
            vocabulary: config.vocabulary_size(),
            hidden_size: config.hidden_size(),
            normalization_epsilon: config.rms_norm_epsilon(),
            normalization_offset: config.normalization_offset(),
            embedding_format: config.linear_format(&embedding_name),
            head_format: config.linear_format("lm_head.weight"),
            tied_head: config.tie_word_embeddings(),
        }
    }
}

// Borrow the actual tied or untied owner for every decoder readout. Instrumentation
// and vocabulary collectives share this projection without changing normalization.
struct DecoderOutput<'a, B: NeuralBackend> {
    embedding: Option<&'a mut B::Embedding>,
    head: Option<&'a mut B::Linear>,
}
impl<B: NeuralBackend> DecoderOutput<'_, B> {
    fn project(
        self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let logits = match self.head {
            Some(head) => {
                instrumentation.project::<B>("projection_input", head, hidden, None, context)
            }
            None => instrumentation.project_embedding(
                "projection_input",
                self.embedding
                    .ok_or_else(|| Error::backend("tied decoder output has no embedding"))?,
                hidden,
                context,
            ),
        }?;
        instrumentation.apply("linear", logits)
    }
}
impl<B: eredu_nn::DistributedNeuralBackend> DecoderOutput<'_, B> {
    fn project_parallel(
        self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let logits = match self.head {
            Some(head) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                hidden,
                parallel,
                context,
            ),
            None => instrumentation.project_vocabulary_embedding::<B>(
                "projection_input",
                self.embedding
                    .ok_or_else(|| Error::backend("tied decoder output has no embedding"))?,
                hidden,
                parallel,
                context,
            ),
        }?;
        instrumentation.apply("linear", logits)
    }
}

impl<B: eredu_nn::DistributedNeuralBackend, D: DecoderBoundary<B>> StaticModules<B, D> {
    /// Projects normalized hidden values through the existing tensor-parallel head.
    pub fn project_normalized_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Looks up rank-local vocabulary rows, reduces them, then expands the residual.
    pub fn embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        policy: EmbeddingLookupPolicy,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings =
            B::vocabulary_parallel_lookup(&mut self.embeddings, tokens, policy, parallel, context)?;
        self.expand_embeddings(&embeddings, context)
    }

    /// Projects an already normalized value through its actual tied or untied
    /// head, retaining the vocabulary collective and projection-input boundary.
    pub(crate) fn project_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let output = DecoderOutput::<B> {
            embedding: Some(&mut self.embeddings),
            head: self.lm_head.as_mut(),
        };
        match parallel {
            Some(parallel) => output.project_parallel(hidden, parallel, context, instrumentation),
            None => output.project(hidden, context, instrumentation),
        }
    }

    /// Applies the declared readout boundary and vocabulary collective with instrumentation.
    pub fn finish_parallel_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.boundary.collapse(hidden, context, instrumentation)?;
        self.project_instrumented(&hidden, Some(parallel), context, instrumentation)
    }
}

impl<B: NeuralBackend, D: DecoderBoundary<B>> StaticModules<B, D> {
    /// Projects normalized hidden values through the existing vocabulary head.
    /// Final normalization is not repeated; tied embeddings retain their owner.
    pub fn project_normalized(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_normalized_instrumented(
            hidden,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Applies the actual residual, normalization and affine readout boundaries.
    pub fn finish_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.boundary.collapse(hidden, context, instrumentation)?;
        self.project_normalized_instrumented(&hidden, context, instrumentation)
    }

    fn project_normalized_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        DecoderOutput::<B> {
            embedding: Some(&mut self.embeddings),
            head: self.lm_head.as_mut(),
        }
        .project(hidden, context, instrumentation)
    }

    /// Embeds token IDs and applies the architecture's residual expansion.
    pub fn embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings = self.embeddings.forward(tokens, context)?;
        self.expand_embeddings(&embeddings, context)
    }

    /// Expands already prepared embeddings after any media replacement.
    pub fn expand_embeddings(
        &self,
        embeddings: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.boundary.expand(embeddings, context)
    }

    /// Builds vocabulary parameters around an architecture-owned residual boundary.
    /// No ordinary normalization is allocated for a different readout algorithm.
    pub fn from_boundary(
        embedding: EmbeddingSpec,
        head: Option<LinearSpec>,
        boundary: D,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        validate_vocabulary_specs(&embedding, head.as_ref())?;
        if embedding.dimensions != boundary.embedding_width() {
            return Err(Error::backend(
                "decoder boundary width differs from its vocabulary parameters",
            ));
        }
        Ok(Self {
            embeddings: B::embedding(embedding, context)?,
            boundary,
            lm_head: head.map(|spec| B::linear(spec, context)).transpose()?,
        })
    }

    /// Builds the same boundary with explicitly selected vocabulary ownership.
    pub fn from_parallel_boundary(
        embedding: EmbeddingSpec,
        head: Option<LinearSpec>,
        boundary: D,
        embedding_range: VocabularyParallelRange,
        output_range: Option<VocabularyParallelRange>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        validate_vocabulary_specs(&embedding, head.as_ref())?;
        if embedding.dimensions != boundary.embedding_width() {
            return Err(Error::backend(
                "decoder boundary width differs from its vocabulary parameters",
            ));
        }
        embedding_range.validate_global_rows(embedding.vocabulary)?;
        if head.is_some() != output_range.is_some() {
            return Err(Error::backend(
                "vocabulary ownership must match the tied or separate head",
            ));
        }
        if let (Some(head), Some(range)) = (&head, &output_range) {
            range.validate_global_rows(head.output)?;
        }
        Ok(Self {
            embeddings: B::vocabulary_parallel_embedding(embedding, embedding_range, context)?,
            boundary,
            lm_head: head
                .zip(output_range)
                .map(|(spec, range)| B::vocabulary_parallel_linear(spec, range, context))
                .transpose()?,
        })
    }
}

pub(crate) fn validate_vocabulary_specs(
    embedding: &EmbeddingSpec,
    head: Option<&LinearSpec>,
) -> Result<(), Error> {
    if embedding.vocabulary <= 0
        || embedding.dimensions <= 0
        || head.is_some_and(|head| {
            head.input != embedding.dimensions || head.output != embedding.vocabulary
        })
    {
        return Err(Error::backend(
            "decoder embedding and vocabulary projection geometry disagree",
        ));
    }
    embedding.format.validate_for_weight(&embedding.weight)?;
    if let Some(head) = head {
        head.format.validate_for_weight(&head.weight)?;
    }
    Ok(())
}

impl<B: NeuralBackend> StaticModules<B> {
    /// Builds unloaded pinned modules from architecture-owned parameter
    /// identities and physical formats.
    pub fn from_spec(
        spec: StaticModuleSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_construction(spec.construction_specs()?, context)
    }

    pub(crate) fn from_construction(
        construction: StaticModuleConstruction,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let StaticModuleConstruction {
            embedding,
            head,
            normalization,
        } = construction;
        let boundary = NormalizedBoundary::new(normalization, context)?;
        Self::from_boundary(embedding, head, boundary, context)
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
        Self::from_parallel_construction(
            spec.construction_specs()?,
            embedding_range,
            output_range,
            context,
        )
    }

    pub(crate) fn from_parallel_construction(
        construction: StaticModuleConstruction,
        embedding_range: VocabularyParallelRange,
        output_range: Option<VocabularyParallelRange>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let StaticModuleConstruction {
            embedding,
            head,
            normalization,
        } = construction;
        let boundary = NormalizedBoundary::new(normalization, context)?;
        Self::from_parallel_boundary(
            embedding,
            head,
            boundary,
            embedding_range,
            output_range,
            context,
        )
    }

    /// Builds unloaded pinned modules for a decoder family.
    pub fn new<C: Config>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(StaticModuleSpec::from_config(config), context)
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
            StaticModuleSpec::from_config(config),
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
}

impl SequentialPredictionGroups {
    /// Creates the target group and `mtp.{depth}` prediction groups.
    pub fn new(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_roots: impl IntoIterator<Item = String>,
    ) -> Result<Self, Error> {
        Ok(Self {
            target: SequentialGroup::new(
                TARGET_EXECUTION_GROUP,
                target_parameter_root,
                target_units,
            )?,
            prediction_paths: prediction_roots
                .into_iter()
                .map(|root| vec![root])
                .collect(),
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
        if (prediction_groups != 0 && units_per_group == 0) || prediction_parameter_root.is_empty()
        {
            return Err(Error::backend(
                "prediction execution groups require non-empty names and units",
            ));
        }
        let prediction_paths = (0..prediction_groups)
            .map(|group| {
                let start = group
                    .checked_mul(units_per_group)
                    .ok_or_else(|| Error::backend("prediction physical index overflowed"))?;
                (0..units_per_group)
                    .map(|unit| {
                        start
                            .checked_add(unit)
                            .map(|physical| format!("{prediction_parameter_root}.{physical}"))
                            .ok_or_else(|| Error::backend("prediction physical index overflowed"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            target: SequentialGroup::new(
                TARGET_EXECUTION_GROUP,
                target_parameter_root,
                target_units,
            )?,
            prediction_paths,
        })
    }

    /// Builds one dependency chain from target through every prediction depth.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(
            std::iter::once(TARGET_EXECUTION_GROUP.to_owned())
                .chain(self.prediction_execution_groups()),
        )
        .map_err(Error::backend)
    }

    /// Returns stable prediction-group identities in prediction-depth order.
    pub fn prediction_execution_groups(&self) -> Vec<String> {
        (0..self.prediction_paths.len())
            .map(|depth| format!("mtp.{depth}"))
            .collect()
    }

    /// Returns the number of units in one group.
    pub fn unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            self.target.unit_count(0)
        } else if group <= self.prediction_paths.len() {
            Ok(self.prediction_paths[group - 1].len())
        } else {
            Err(Error::backend(format!(
                "execution group {group} is outside target plus {} prediction groups",
                self.prediction_paths.len()
            )))
        }
    }

    /// Returns one stable target or prediction unit path.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if group == 0 {
            return self.target.unit_path(0, index);
        }
        self.unit_count(group)?;
        self.prediction_paths[group - 1]
            .get(index)
            .cloned()
            .ok_or_else(|| {
                Error::backend(format!(
                    "unit {index} is outside {} units in prediction group {group}",
                    self.prediction_paths[group - 1].len()
                ))
            })
    }

    /// Selects the activation carried into a ready chain group.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group)?;
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
        if name.is_empty() || parameter_root.is_empty() || units == 0 {
            return Err(Error::backend(
                "sequential decoder group requires non-empty names and units",
            ));
        }
        Ok(Self {
            name,
            parameter_root,
            units,
        })
    }

    /// Builds the corresponding one-group dependency graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain([self.name]).map_err(Error::backend)
    }

    /// Validates the group ordinal and returns its unit count.
    pub fn unit_count(&self, group: usize) -> Result<usize, Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "execution group {group} is outside {}",
                self.name
            )));
        }
        Ok(self.units)
    }

    /// Returns one validated stable unit path.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        let count = self.unit_count(group)?;
        if index >= count {
            return Err(Error::backend(format!(
                "unit {index} is outside {count} {} units",
                self.name
            )));
        }
        Ok(format!("{}.{index}", self.parameter_root))
    }

    /// Starts the sole group from the initial activation.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group)?;
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
}

impl<T> ForwardContext<T> {
    /// Complete position visibility resolved for the current invocation.
    pub fn visibility_mask(&self) -> Option<&T> {
        self.mask.as_ref()
    }

    /// Borrows the same invocation geometry for every ordinary block equation.
    fn attention_input<'a, C>(
        &'a self,
        hidden: &'a T,
        cache: &'a mut C,
    ) -> AttentionInput<'a, T, C> {
        AttentionInput {
            hidden,
            selected_positions: None,
            mask: self.mask.as_ref(),
            cache: Some(cache),
            allow_sliding_prefill: self.allow_sliding_prefill,
            rotary_position: self
                .rotary_embeddings
                .as_ref()
                .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
        }
    }
}

// Full and selected decoder models resolve visibility through the same ingress.
// The caller supplies the state ordinal in its own selected state view.
#[allow(clippy::too_many_arguments)]
fn begin_decoder_hidden<B, S>(
    hidden: B::Tensor,
    supplied_mask: Option<&B::Tensor>,
    rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
    state: &mut S,
    expected: Option<&StateLayout>,
    first_state_ordinal: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    if state.optional_layout() != expected {
        return Err(Error::backend(format!(
            "decoder runtime state layout {:?} does not match architecture layout {expected:?}",
            state.optional_layout()
        )));
    }
    let sequence = hidden.dim(1);
    let mask = if let Some(mask) = supplied_mask {
        Some(mask.clone())
    } else if sequence > 1 && expected.is_some() {
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
        },
    })
}

/// Typed projection construction and parameter policy for one decoder block family.
/// The shared constructor owns attention, residual and terminal assembly.
pub trait BlockFactory<B: NeuralBackend, C: Config>: 'static {
    /// Architecture-selected feed-forward policy inside the shared block.
    type FeedForward: DecoderProjectionOperator<B>;

    /// Validates configuration requirements specific to this block policy.
    fn validate(config: &C) -> Result<(), Error> {
        let _ = config;
        Ok(())
    }

    /// Admits factory-specific global/local policy before allocating body resources.
    /// Ordinary factories validate local construction; routed factories retain
    /// global cardinality checks independently of owner-local bank dimensions.
    fn validate_partitioned(global: &C, local: &C) -> Result<(), Error> {
        let _ = global;
        Self::validate(local)
    }

    /// Backend-neutral typed parameters for the factory's dense or provider-owned stages.
    type Parameters: Clone + eredu_nn::DeclareLogicalParameters;

    /// Prepares complete local stage geometry before any backend allocation.
    /// Supplied dense or typed projection slots are mutually exclusive construction authorities.
    fn prepare_parameters(
        global: &C,
        local: &C,
        layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self::Parameters>,
    ) -> Result<Self::Parameters, Error>;

    /// Projects the complete typed source topology before native construction.
    fn parameter_description(
        config: &C,
        common: &DecoderDeclaration,
        _projections: Option<&ProjectionDeclarations<Self::Parameters>>,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let common = common
            .parameter_description(config, ParameterEncoding::Stored)
            .map_err(Error::backend_source)?;
        config
            .extend_common_parameters(common)
            .map_err(Error::backend_source)
    }

    /// Materializes the exact prepared stages without redeclaring parameter geometry.
    fn build_feed_forward(
        global: &C,
        local: &C,
        layer: usize,
        parameters: &Self::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error>;
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
        P: eredu_runtime::ParameterProvider<B>,
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
        P: eredu_runtime::ParameterProvider<B>,
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
        P: eredu_runtime::ParameterProvider<B>,
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
        P: eredu_runtime::ParameterProvider<B>,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
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
    type FeedForward = Mlp<B>;

    type Parameters = eredu_nn::GatedProductSpec;

    fn prepare_parameters(
        _global: &C,
        local: &C,
        _layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self::Parameters>,
    ) -> Result<Self::Parameters, Error> {
        if let Some(parameters) = retained {
            return parameters.with_intermediate_dimensions(local.intermediate_size());
        }
        dense.cloned().ok_or_else(|| {
            Error::backend("dense construction requires retained feed-forward slots")
        })
    }

    fn build_feed_forward(
        _global: &C,
        _local: &C,
        _layer: usize,
        parameters: &Self::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        Mlp::from_parameters(parameters, context)
    }
}

/// Shared runtime adapter for a complete decoder and its typed resource aggregate.
pub type LayeredModel<B, C, P = DenseBlockFactory> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompleteDecoderKernel<B, C, P>,
    eredu_runtime::KernelParameterModules,
>;

/// Complete decoder equations, with ordinary or tensor-local geometry.
pub type CompleteDecoderKernel<B, C, P = DenseBlockFactory> =
    DecoderKernel<B, C, P, StaticModules<B>, CompleteDecoderGeometry<C>>;

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

impl<B: eredu_nn::DistributedNeuralBackend> PartitionStaticModules<B> {
    /// Materializes only canonical pinned roles selected before native allocation.
    pub(crate) fn from_selected_construction(
        construction: StaticModuleConstruction,
        vocabulary: Option<(VocabularyParallelRange, Option<VocabularyParallelRange>)>,
        owns: impl Fn(&str) -> bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if let Some((embedding_range, output_range)) = &vocabulary {
            embedding_range.validate_global_rows(construction.embedding.vocabulary)?;
            if construction.head.is_some() != output_range.is_some() {
                return Err(Error::backend(
                    "vocabulary ownership differs from tied or independent readout",
                ));
            }
            if let (Some(head), Some(range)) = (&construction.head, output_range) {
                range.validate_global_rows(head.output)?;
            }
        }
        let embedding = if owns("embedding") {
            vocabulary
                .as_ref()
                .map_or(VocabularyModulePlacement::Local, |(range, _)| {
                    VocabularyModulePlacement::VocabularyParallel(range.clone())
                })
        } else {
            VocabularyModulePlacement::Omitted
        };
        let norm = owns("norm");
        let head = if construction.head.is_some() && owns("output") {
            vocabulary
                .as_ref()
                .map_or(VocabularyModulePlacement::Local, |(_, range)| {
                    VocabularyModulePlacement::VocabularyParallel(
                        range.clone().expect("validated output range"),
                    )
                })
        } else {
            VocabularyModulePlacement::Omitted
        };
        Self::from_construction(construction, embedding, head, norm, context)
    }

    /// Materializes validated resources with independently selected vocabulary owners.
    pub(crate) fn from_construction(
        construction: StaticModuleConstruction,
        embedding: VocabularyModulePlacement,
        head: VocabularyModulePlacement,
        owns_norm: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let StaticModuleConstruction {
            embedding: embedding_spec,
            head: head_spec,
            normalization,
        } = construction;
        embedding.validate_rows(embedding_spec.vocabulary)?;
        if let Some(spec) = &head_spec {
            head.validate_rows(spec.output)?;
        } else if !matches!(head, VocabularyModulePlacement::Omitted) {
            return Err(Error::backend(
                "selected output has no vocabulary projection",
            ));
        }
        let embeddings = embedding.materialize_embedding::<B>(embedding_spec, context)?;
        let norm = owns_norm
            .then(|| B::normalization(normalization, context))
            .transpose()?;
        let lm_head = match head_spec {
            Some(spec) => head.materialize_linear::<B>(spec, context)?,
            None => None,
        };
        Ok(Self {
            embeddings,
            norm,
            lm_head,
        })
    }
}

impl<B: NeuralBackend> From<StaticModules<B>> for PartitionStaticModules<B> {
    fn from(modules: StaticModules<B>) -> Self {
        Self {
            embeddings: Some(modules.embeddings),
            norm: Some(modules.boundary.norm),
            lm_head: modules.lm_head,
        }
    }
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

/// Canonical pinned decoder roles borrowed from their physical module owners.
pub struct DecoderResources<'a, B: NeuralBackend> {
    /// Input or tied-output embedding, when physically retained.
    pub embedding: Option<&'a B::Embedding>,
    /// Final normalization, when physically retained.
    pub norm: Option<&'a B::Normalization>,
    /// Untied output head, when physically retained.
    pub head: Option<&'a B::Linear>,
}

/// Mutable access to the same physical pinned decoder roles.
pub struct DecoderResourcesMut<'a, B: NeuralBackend> {
    /// Input or tied-output embedding, when physically retained.
    pub embedding: Option<&'a mut B::Embedding>,
    /// Final normalization, when physically retained.
    pub norm: Option<&'a mut B::Normalization>,
    /// Untied output head, when physically retained.
    pub head: Option<&'a mut B::Linear>,
}

/// Typed access to complete or selected pinned resources without a second owner.
pub trait DecoderStaticResources<B: NeuralBackend>:
    eredu_nn::Parameterized<B::Tensor> + 'static
{
    /// Borrows the canonical roles actually allocated by this aggregate.
    fn decoder_resources(&self) -> DecoderResources<'_, B>;
    /// Mutably borrows those same roles for equations and parameter editing.
    fn decoder_resources_mut(&mut self) -> DecoderResourcesMut<'_, B>;
}
impl<B: NeuralBackend> DecoderStaticResources<B> for StaticModules<B> {
    fn decoder_resources(&self) -> DecoderResources<'_, B> {
        DecoderResources {
            embedding: Some(&self.embeddings),
            norm: Some(&self.boundary.norm),
            head: self.lm_head.as_ref(),
        }
    }
    fn decoder_resources_mut(&mut self) -> DecoderResourcesMut<'_, B> {
        DecoderResourcesMut {
            embedding: Some(&mut self.embeddings),
            norm: Some(&mut self.boundary.norm),
            head: self.lm_head.as_mut(),
        }
    }
}
impl<B: NeuralBackend> DecoderStaticResources<B> for PartitionStaticModules<B> {
    fn decoder_resources(&self) -> DecoderResources<'_, B> {
        DecoderResources {
            embedding: self.embeddings.as_ref(),
            norm: self.norm.as_ref(),
            head: self.lm_head.as_ref(),
        }
    }
    fn decoder_resources_mut(&mut self) -> DecoderResourcesMut<'_, B> {
        DecoderResourcesMut {
            embedding: self.embeddings.as_mut(),
            norm: self.norm.as_mut(),
            head: self.lm_head.as_mut(),
        }
    }
}
impl<B: NeuralBackend> DecoderResources<'_, B> {
    /// Visits each physically retained canonical decoder role once.
    pub fn visit<V: eredu_runtime::StaticParameterVisitor<B>>(
        self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(embedding) = self.embedding {
            visitor.visit("embedding", embedding)?;
        }
        if let Some(norm) = self.norm {
            visitor.visit("norm", norm)?;
        }
        if let Some(head) = self.head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }
}
impl<'a, B: NeuralBackend> DecoderResourcesMut<'a, B> {
    /// Mutably visits the same canonical roles and physical resources.
    pub fn visit<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(embedding) = self.embedding {
            visitor.visit_mut("embedding", embedding)?;
        }
        if let Some(norm) = self.norm {
            visitor.visit_mut("norm", norm)?;
        }
        if let Some(head) = self.head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
    fn input_embedding(self) -> Result<&'a mut B::Embedding, Error> {
        self.embedding
            .ok_or_else(|| Error::backend("decoder does not own input embedding"))
    }
    pub(crate) fn normalize(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let norm = self
            .norm
            .as_deref_mut()
            .ok_or_else(|| Error::backend("decoder does not own final normalization"))?;
        instrumentation.normalize_readout(hidden, norm, context)
    }
    fn finish(
        mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.normalize(hidden, context, instrumentation)?;
        DecoderOutput::<B> {
            embedding: self.embedding,
            head: self.head,
        }
        .project(&hidden, context, instrumentation)
    }
    /// Projects without normalization through the actually retained vocabulary owner.
    pub(crate) fn project(
        self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let output = DecoderOutput::<B> {
            embedding: self.embedding,
            head: self.head,
        };
        match parallel {
            Some(parallel) => output.project_parallel(hidden, parallel, context, instrumentation),
            None => output.project(hidden, context, instrumentation),
        }
    }
    pub(crate) fn finish_parallel(
        mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let Some(parallel) = parallel else {
            return self.finish(hidden, context, instrumentation);
        };
        let hidden = self.normalize(hidden, context, instrumentation)?;
        DecoderOutput::<B> {
            embedding: self.embedding,
            head: self.head,
        }
        .project_parallel(&hidden, parallel, context, instrumentation)
    }
}

/// Backend-neutral geometry retained by one genuinely pipeline-local decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionLocalGeometry<C> {
    owned_units: Range<usize>,
    owned_state: Option<Range<usize>>,
    blocks: std::collections::BTreeMap<usize, C>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
}

impl<C: Config> PartitionLocalGeometry<C> {
    /// Returns the selected equation range, including stateless operations.
    pub fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }

    /// Returns one selected canonical parameter owner's construction geometry.
    /// Its address may lie outside this partition's logical invocation range.
    pub fn block(&self, global_unit: usize) -> Option<&C> {
        self.blocks.get(&global_unit)
    }

    /// Returns the number of selected physical owner configurations.
    pub fn local_unit_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns complete TP-local state geometry before pipeline slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }
}

/// Geometry operations required by the common decoder equations.
pub trait DecoderGeometry<C: Config>: 'static {
    /// Whether ordinary text requests admit chunked prefill.
    const CHUNKED_PREFILL: bool = false;
    /// Whether text-only readout narrows hidden state before projection.
    const FINAL_TEXT_POSITION: bool = false;
    /// Complete local-head state geometry before pipeline slicing.
    fn complete_state_layout(&self) -> &StateLayout;
    /// Selected operation range, independent of cache ownership.
    fn owned_units(&self) -> Range<usize>;
    /// Logical cache slots owned independently of operation addresses.
    fn owned_state(&self) -> Option<Range<usize>>;
    /// Physical construction authority, independent of invocation placement.
    fn owns_construction_unit(&self, owner: usize) -> bool {
        self.owned_units().contains(&owner)
    }
    /// Construction geometry of one selected canonical physical owner.
    /// Mutable-state ownership is checked independently by `state_index`.
    fn block<'a>(&'a self, global: &'a C, index: usize, layer: usize) -> Result<&'a C, Error>;
    /// Ordinary ingress can require replicated state even on a TP-constructed kernel.
    fn ordinary_state_layout<'a>(
        &'a self,
        _global: &C,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        Ok(std::borrow::Cow::Borrowed(self.complete_state_layout()))
    }
    /// Tensor-parallel ingress requires an admitted rank-local realization.
    fn parallel_state_layout(&self) -> Result<&StateLayout, Error> {
        Ok(self.complete_state_layout())
    }
    /// State ordinal in this kernel's local cache view.
    fn state_index(&self, index: usize) -> Result<usize, Error> {
        let owned = self
            .owned_state()
            .ok_or_else(|| Error::backend("stateless decoder partition"))?;
        if !owned.contains(&index) {
            return Err(Error::backend(format!(
                "decoder cache slot {index} is outside owned range {owned:?}"
            )));
        }
        Ok(index - owned.start)
    }
    /// Complete low-level calls select their own cache origin; selected state is already sliced.
    fn partition_state_ordinal(&self, _requested: usize) -> usize {
        0
    }
}

/// Complete decoder geometry projected from its retained declaration and TP plan.
#[derive(Clone)]
pub struct CompleteDecoderGeometry<C> {
    parallel: Option<std::sync::Arc<LocalGeometry<C>>>,
    ordinary_state: StateLayout,
    state: StateLayout,
    units: usize,
}
impl<C: Config> DecoderGeometry<C> for CompleteDecoderGeometry<C> {
    const CHUNKED_PREFILL: bool = CHUNKED_TEXT_PREFILL;
    const FINAL_TEXT_POSITION: bool = true;
    fn complete_state_layout(&self) -> &StateLayout {
        &self.state
    }
    fn owned_units(&self) -> Range<usize> {
        0..self.units
    }
    fn owned_state(&self) -> Option<Range<usize>> {
        Some(0..self.state.len())
    }
    fn block<'a>(&'a self, global: &'a C, _index: usize, layer: usize) -> Result<&'a C, Error> {
        self.state_index(layer)?;
        match &self.parallel {
            Some(geometry) => geometry.block(layer).ok_or_else(|| {
                Error::backend(format!("decoder local geometry is missing block {layer}"))
            }),
            None => Ok(global),
        }
    }
    fn ordinary_state_layout<'a>(
        &'a self,
        _global: &C,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        Ok(std::borrow::Cow::Borrowed(&self.ordinary_state))
    }
    fn parallel_state_layout(&self) -> Result<&StateLayout, Error> {
        self.parallel
            .as_ref()
            .map(|geometry| geometry.state_layout())
            .ok_or_else(|| Error::backend("decoder model was not built with local geometry"))
    }
    fn partition_state_ordinal(&self, requested: usize) -> usize {
        requested
    }
}
impl<C: Config> DecoderGeometry<C> for PartitionLocalGeometry<C> {
    fn owns_construction_unit(&self, owner: usize) -> bool {
        self.blocks.contains_key(&owner)
    }
    fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }
    fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }
    fn owned_state(&self) -> Option<Range<usize>> {
        self.owned_state.clone()
    }
    fn block<'a>(&'a self, _global: &'a C, index: usize, _layer: usize) -> Result<&'a C, Error> {
        self.block(index).ok_or_else(|| {
            Error::backend(format!(
                "decoder physical owner {index} is not selected for construction"
            ))
        })
    }
}

/// Shared runtime adapter over an admitted decoder partition and its typed equations.
pub type PartitionedLayeredModel<B, C, P = DenseBlockFactory> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    DecoderKernel<B, C, P>,
    eredu_runtime::KernelParameterModules,
>;

/// Declares decoder operations and resource ownership without native allocation.
pub fn component_definition<C: Config>(
    args: &C,
    parameters: ArchitectureParameterDescription,
    state: StateLayout,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let schedule = DecoderSchedule::new(args)?;
    if parameters.graph().groups().len() != 1
        || parameters.graph().groups()[0].id() != TEXT_DECODER_EXECUTION_GROUP
        || parameters.unit_layout().len() != schedule.len()
        || (0..schedule.len()).any(|index| {
            parameters
                .unit_layout()
                .address(index)
                .is_none_or(|address| address.index() != index)
        })
        || state.len() != args.num_hidden_layers() as usize
    {
        return Err(eredu_runtime::ComponentDefinitionError::Invalid {
            component: "text_decoder".into(),
            resource: "configured operation or cache identities".into(),
        }
        .into());
    }

    let units = (0..parameters.unit_layout().len())
        .map(|ordinal| {
            parameters
                .unit_layout()
                .address(ordinal)
                .expect("declared unit")
        })
        .map(|address| eredu_runtime::ComponentUnit {
            path: schedule
                .path(args, address.index())
                .expect("declared operation"),
            state: match schedule
                .operation(address.index())
                .expect("declared operation")
            {
                DecoderOperation::Body { layer } => layer..layer + 1,
                DecoderOperation::Terminal { .. } => 0..0,
            },
        })
        .collect::<Vec<_>>();
    let partition = schedule.state_partition_plan(parameters.unit_layout());
    eredu_runtime::ComponentDefinition::new(
        "text_decoder",
        parameters,
        units,
        vec![crate::transport::decoder()],
        TEXT_DECODER_EXECUTION_GROUP,
        vec![],
        state,
        partition,
    )
    .and_then(|definition| definition.with_unit_parameter_paths())
    .and_then(|definition| {
        definition.map_unit_observation_boundaries(|address, _| {
            schedule.boundaries(args, address.index())
        })
    })
    .map_err(Error::from)
}

fn prepare_projection<B: NeuralBackend, C: Config, P: BlockFactory<B, C>>(
    global: &C,
    local: &C,
    layer: usize,
    dense: Option<&eredu_nn::GatedProductSpec>,
    retained: Option<&P::Parameters>,
) -> Result<P::Parameters, Error> {
    if dense.is_some() && retained.is_some() {
        return Err(Error::backend(
            "conflicting projection declaration authorities",
        ));
    }
    P::validate_partitioned(global, local)?;
    P::prepare_parameters(global, local, layer, dense, retained)
}

fn validate_prepared_projection(
    common: &topology::BlockParameters,
    parameters: &impl eredu_nn::DeclareLogicalParameters,
) -> Result<(), Error> {
    let mut common = common.clone();
    common.feed_forward = None;
    let mut declarations = common.logical_parameters()?;
    declarations.extend(parameters.logical_parameters()?);
    eredu_nn::validate_logical_parameter_scope(&declarations).map_err(Error::backend_source)
}

#[derive(Clone)]
pub(crate) struct PreparedDecoderBody<S> {
    pub(crate) common: topology::BlockParameters,
    pub(crate) parameters: S,
}
fn prepare_decoder_body<C: Config, S: Clone + eredu_nn::DeclareLogicalParameters>(
    args: &C,
    config: &C,
    layer: usize,
    decoder_scopes: &DecoderDeclaration,
    projection_declarations: Option<&ProjectionDeclarations<S>>,
    prepare: &impl Fn(
        &C,
        &C,
        usize,
        Option<&eredu_nn::GatedProductSpec>,
        Option<&S>,
    ) -> Result<S, Error>,
) -> Result<PreparedDecoderBody<S>, Error> {
    let mut body = {
        let source = decoder_scopes
            .blocks
            .get(layer)
            .ok_or_else(|| Error::backend("missing retained dense block"))?;
        let old = source.attention.geometry;
        source.with_geometry(
            eredu_nn::AttentionHeadGeometry::new(
                config.num_attention_heads(),
                config.num_key_value_heads(),
                old.key_dimensions(),
                old.value_dimensions(),
            )?,
            source
                .feed_forward
                .as_ref()
                .map(|_| config.intermediate_size()),
            args.query_key_norm_per_head_weights(),
        )?
    };
    body.terminal = None;
    let retained = projection_declarations
        .map(|declarations| declarations.parameters(layer))
        .transpose()?;
    if body.feed_forward.is_some() && retained.is_some() {
        return Err(Error::backend(
            "conflicting projection declaration authorities",
        ));
    }
    let parameters = prepare(args, config, layer, body.feed_forward.as_ref(), retained)?;
    body.feed_forward = None;
    validate_prepared_projection(&body, &parameters)?;
    Ok(PreparedDecoderBody {
        common: body,
        parameters,
    })
}
/// Projects retained common/provider slots before any module allocation.
pub(crate) fn prepare_retained_body<
    B: NeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
>(
    global: &C,
    local: &C,
    layer: usize,
    common: &DecoderDeclaration,
    projections: Option<&ProjectionDeclarations<P::Parameters>>,
) -> Result<PreparedDecoderBody<P::Parameters>, Error> {
    common.validate_local_policy(global, local)?;
    if let Some(scopes) = projections {
        scopes.validate_for(global)?;
    }
    prepare_decoder_body(
        global,
        local,
        layer,
        common,
        projections,
        &prepare_projection::<B, C, P>,
    )
}

#[derive(Clone)]
enum DecoderUnitConstruction<S> {
    Body {
        common: topology::BlockParameters,
        parameters: S,
    },
    Terminal(NormalizationConstructionSpec),
}

fn decoder_unit_construction<C: Config, S: Clone + eredu_nn::DeclareLogicalParameters>(
    args: &C,
    geometry: &impl DecoderGeometry<C>,
    decoder_scopes: &DecoderDeclaration,
    projection_declarations: Option<&ProjectionDeclarations<S>>,
    prepare: impl Fn(&C, &C, usize, Option<&eredu_nn::GatedProductSpec>, Option<&S>) -> Result<S, Error>,
) -> Result<std::collections::BTreeMap<usize, DecoderUnitConstruction<S>>, Error> {
    decoder_scopes.validate_for(args)?;
    if let Some(declarations) = projection_declarations {
        declarations.validate_for(args)?;
    }
    let schedule = DecoderSchedule::new(args)?;
    schedule
        .resources()
        .into_iter()
        .filter(|(owner, _)| geometry.owns_construction_unit(*owner))
        .map(|(owner, _)| {
            let operation = schedule.operation(owner)?;
            let layer = operation.layer();
            let config = geometry.block(args, owner, layer)?;
            let scope = match operation {
                DecoderOperation::Body { .. } => {
                    let prepared = prepare_decoder_body(
                        args,
                        config,
                        layer,
                        decoder_scopes,
                        projection_declarations,
                        &prepare,
                    )?;
                    DecoderUnitConstruction::Body {
                        common: prepared.common,
                        parameters: prepared.parameters,
                    }
                }
                DecoderOperation::Terminal { .. } => DecoderUnitConstruction::Terminal(
                    decoder_scopes
                        .blocks
                        .get(layer)
                        .and_then(|block| block.terminal.clone())
                        .ok_or_else(|| {
                            Error::backend("decoder terminal has no normalization policy")
                        })?,
                ),
            };
            Ok((owner, scope))
        })
        .collect()
}

/// Typed decoder equations over exact admitted pipeline geometry.
pub struct DecoderKernel<
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C> = DenseBlockFactory,
    M = PartitionStaticModules<B>,
    G = PartitionLocalGeometry<C>,
> {
    args: C,
    schedule: DecoderSchedule,
    static_modules: M,
    geometry: G,
    construction: std::collections::BTreeMap<usize, DecoderUnitConstruction<P::Parameters>>,
    block_factory: std::marker::PhantomData<fn() -> (B, P)>,
}

impl<B: NeuralBackend, C: Config + Clone, P: BlockFactory<B, C>, M: Clone, G: Clone> Clone
    for DecoderKernel<B, C, P, M, G>
{
    fn clone(&self) -> Self {
        Self {
            args: self.args.clone(),
            schedule: self.schedule.clone(),
            static_modules: self.static_modules.clone(),
            geometry: self.geometry.clone(),
            construction: self.construction.clone(),
            block_factory: std::marker::PhantomData,
        }
    }
}

impl<B, C, P> DecoderKernel<B, C, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
{
    /// Constructs only modules already selected by an exact neutral partition.
    pub fn from_partition<A>(
        args: C,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        decoder_scopes: &DecoderDeclaration,
        projection_declarations: Option<&ProjectionDeclarations<P::Parameters>>,
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry<C>, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedLayeredModel<B, C, P>, Error> {
        args.validate_config()?;
        if let Some(declarations) = projection_declarations {
            declarations.validate_against(&args, definition.parameters())?;
        }
        P::validate(&args)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        for local in partition.local_geometry().blocks.values() {
            decoder_scopes.validate_local_policy(&args, local)?;
        }
        let expected = P::parameter_description(&args, decoder_scopes, projection_declarations)?;
        definition.validate_parameter_topology(&expected)?;
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
        let expected_storage = partition.invocations().storage_layout();
        if !partition
            .local_geometry()
            .blocks
            .keys()
            .copied()
            .eq((0..expected_storage.len()).map(|ordinal| {
                expected_storage
                    .address(ordinal)
                    .expect("selected physical owner")
                    .index()
            }))
        {
            return Err(Error::backend(
                "decoder factory geometry differs from selected parameter owners",
            ));
        }
        match (&partition.local_geometry().owned_state, partition.state()) {
            (Some(owned), Some(state)) => {
                let expected = partition
                    .local_geometry()
                    .complete_state_layout()
                    .slice(owned.clone())
                    .map_err(Error::backend)?;
                if state.global_layer_offset() != owned.start || state.layout() != &expected {
                    return Err(Error::backend(
                        "decoder partition cache geometry differs from selected body operations",
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(Error::backend(
                    "decoder partition state ownership differs from its operations",
                ));
            }
        }
        let geometry = partition.local_geometry().clone();
        let definition = definition.select_state_layout(geometry.complete_state_layout())?;
        validate_complete_definition(&args, &definition, geometry.complete_state_layout())?;
        let selection = definition.select_partition(partition)?;
        let construction = decoder_unit_construction(
            &args,
            &geometry,
            decoder_scopes,
            projection_declarations,
            prepare_projection::<B, C, P>,
        )?;
        let statics = decoder_scopes.statics.clone();
        let static_modules = PartitionStaticModules::from_selected_construction(
            statics,
            Some((
                geometry.embedding_range.clone(),
                geometry.output_range.clone(),
            )),
            |role| {
                selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(role.into()))
            },
            context,
        )?;
        let kernel = Self {
            schedule: DecoderSchedule::new(&args)?,
            args,
            static_modules,
            geometry,
            construction,
            block_factory: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }
}

impl<B, C, P, M, G> DecoderKernel<B, C, P, M, G>
where
    B: NeuralBackend,
    C: Config,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
    P: BlockFactory<B, C>,
{
    pub(crate) fn finish_hidden_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let logits =
            self.static_modules
                .decoder_resources_mut()
                .finish(hidden, context, instrumentation)?;
        softcap_logits(logits, self.args.output_softcap(), context)
    }

    /// Returns normalized architecture configuration.
    pub const fn args(&self) -> &C {
        &self.args
    }

    /// Returns exact local pipeline geometry.
    pub const fn local_geometry(&self) -> &G {
        &self.geometry
    }

    /// Returns the physically allocated static modules.
    pub const fn static_modules(&self) -> &M {
        &self.static_modules
    }

    /// Mutably borrows the physically retained static aggregate for binding.
    pub fn static_modules_mut(&mut self) -> &mut M {
        &mut self.static_modules
    }

    /// Constructs an admitted physical owner, independently of local cache ownership.
    pub fn construct_block(
        &self,
        global_unit: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, P::FeedForward>, Error> {
        let owner = self
            .schedule
            .owner(self.schedule.body_operation(global_unit));
        let config = self.geometry.block(&self.args, owner, global_unit)?;
        let Some(DecoderUnitConstruction::Body {
            common: spec,
            parameters,
        }) = self.construction.get(&owner)
        else {
            return Err(Error::backend(
                "decoder body has no retained construction scope",
            ));
        };
        let mut spec = spec.clone();
        if self.schedule.has_terminal(global_unit) {
            let operation = self.schedule.body_operation(global_unit) + 1;
            let owner = self.schedule.owner(operation);
            let Some(DecoderUnitConstruction::Terminal(terminal)) = self.construction.get(&owner)
            else {
                return Err(Error::backend(
                    "decoder terminal has no retained construction scope",
                ));
            };
            spec.terminal = Some(terminal.clone());
        }
        TransformerBlock::from_prepared::<C, P>(
            &self.args,
            config,
            global_unit,
            &spec,
            parameters,
            context,
        )
    }
    /// Materializes exactly one declared body or stateless terminal resource.
    pub fn construct_unit(
        &self,
        operation: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DecoderUnit<B, P::FeedForward>, Error> {
        let equation = self.schedule.operation(operation)?;
        let layer = equation.layer();
        let config = self.geometry.block(&self.args, operation, layer)?;
        match (equation, self.construction.get(&operation)) {
            (
                DecoderOperation::Body { .. },
                Some(DecoderUnitConstruction::Body {
                    common: spec,
                    parameters,
                }),
            ) => Ok(DecoderUnit::Body(TransformerBlockBody::from_prepared::<
                C,
                P,
            >(
                &self.args, config, layer, spec, parameters, context,
            )?)),
            (DecoderOperation::Terminal { .. }, Some(DecoderUnitConstruction::Terminal(spec))) => {
                Ok(DecoderUnit::Terminal(B::normalization(
                    spec.clone(),
                    context,
                )?))
            }
            _ => Err(Error::backend(
                "decoder operation has no retained construction scope",
            )),
        }
    }
}

/// Projects physical factory geometry from a complete invocation map and a selected PP range.
pub fn partition_local_geometry<C: PartitionedConfig>(
    config: &C,
    decoder_scopes: &DecoderDeclaration,
    invocations: &eredu_runtime::ExecutionUnitAddressMap,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry<C>, ParallelPlanError> {
    partition_local_geometry_with(
        config,
        decoder_scopes,
        invocations,
        layout,
        owned_units,
        |config, unit, layout| config.local_block_config(unit, layout),
    )
}

/// Derives partition-local decoder geometry with a family-owned block localizer.
///
/// This remains family-blind: the callback is responsible only for translating
/// an already selected physical layout and routed realization into one local
/// configuration. Static ownership and state geometry stay centralized here.
pub(crate) fn partition_local_geometry_with<C, F>(
    config: &C,
    decoder_scopes: &DecoderDeclaration,
    invocations: &eredu_runtime::ExecutionUnitAddressMap,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    mut localize: F,
) -> Result<PartitionLocalGeometry<C>, ParallelPlanError>
where
    C: PartitionedConfig,
    F: FnMut(&C, usize, &LocalModelLayout) -> Result<C, Error>,
{
    decoder_scopes
        .validate_for(config)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let schedule = DecoderSchedule::new(config)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let count = schedule.len();
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "decoder local unit range {owned_units:?} is outside {count} layers"
        )));
    }
    let invalid = |reason: String| ParallelPlanError::InvalidGroup(reason);
    let group = (0..invocations.storage_layout().group_count())
        .find(|&group| {
            invocations
                .storage_layout()
                .group_id(group)
                .is_some_and(|id| id.as_str() == TEXT_DECODER_EXECUTION_GROUP)
        })
        .ok_or_else(|| invalid("decoder geometry has no canonical execution group".into()))?;
    let logical = invocations.logical_addresses().collect::<Vec<_>>();
    let range = invocations
        .group_range(group)
        .expect("declared invocation group");
    if range.len() != count
        || logical[range.clone()]
            .iter()
            .enumerate()
            .any(|(unit, address)| address.index() != unit)
    {
        return Err(invalid(
            "decoder geometry has a different logical layer layout".into(),
        ));
    }
    let selected = invocations
        .select_invocations(owned_units.clone().map(|unit| logical[range.start + unit]))
        .map_err(|error| invalid(error.to_string()))?;
    let mut physical = std::collections::BTreeMap::new();
    for ordinal in invocations
        .storage_layout()
        .group_range(group)
        .expect("declared storage group")
    {
        let owner = invocations
            .storage_layout()
            .address(ordinal)
            .expect("declared physical owner")
            .index();
        let local = match schedule
            .operation(owner)
            .map_err(|error| invalid(error.to_string()))?
        {
            DecoderOperation::Body { layer } => match decoder_scopes.block_parameters(layer) {
                block if block.feed_forward.is_some() => {
                    dense_local_block_config(config, layer, layout, Some(block))
                }
                _ => localize(config, layer, layout),
            }
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            DecoderOperation::Terminal { .. } => config.clone(),
        };
        physical.insert(owner, local);
    }
    let mut attention_geometry = Vec::with_capacity(count);
    for layer in 0..config.num_hidden_layers() as usize {
        let ordinal = range.start + schedule.body_operation(layer);
        let address = logical[ordinal];
        let owner = invocations
            .resolve(ordinal, address)
            .expect("validated storage mapping")
            .storage_location()
            .address()
            .index();
        let DecoderOperation::Body {
            layer: parameter_layer,
        } = schedule
            .operation(owner)
            .map_err(|error| invalid(error.to_string()))?
        else {
            return Err(invalid(
                "decoder body resolves to a stateless parameter owner".into(),
            ));
        };
        let block = decoder_scopes.block_parameters(parameter_layer);
        let local = &physical[&owner];
        attention_geometry.push(
            block
                .with_geometry(
                    eredu_nn::AttentionHeadGeometry::new(
                        local.num_attention_heads(),
                        local.num_key_value_heads(),
                        block.attention.geometry.key_dimensions(),
                        block.attention.geometry.value_dimensions(),
                    )
                    .map_err(|e| invalid(e.to_string()))?,
                    block
                        .feed_forward
                        .as_ref()
                        .map(|_| local.intermediate_size()),
                    config.query_key_norm_per_head_weights(),
                )
                .map_err(|e| invalid(e.to_string()))?
                .attention
                .geometry,
        );
    }
    let mut blocks = std::collections::BTreeMap::new();
    for ordinal in selected
        .storage_layout()
        .group_range(group)
        .expect("selected storage group")
    {
        let owner = selected
            .storage_layout()
            .address(ordinal)
            .expect("selected owner")
            .index();
        blocks.insert(
            owner,
            physical.remove(&owner).expect("localized physical owner"),
        );
    }
    let complete_state_layout = StateLayout::new(
        cache_layout_with_attention_geometry(config, attention_geometry)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let root = config.parameter_root();
    let statics = &decoder_scopes.statics;
    let (embedding_range, output_range) = statics.vocabulary_ranges(
        layout,
        &format!("{root}.embed_tokens"),
        &format!("{root}.norm"),
        "lm_head",
    )?;
    Ok(PartitionLocalGeometry {
        owned_state: schedule.state_range(owned_units.clone()),
        owned_units,
        blocks,
        embedding_range,
        output_range,
        complete_state_layout,
    })
}

fn dense_local_block_config<C: PartitionedConfig>(
    config: &C,
    layer: usize,
    layout: &LocalModelLayout,
    scope: Option<&topology::BlockParameters>,
) -> Result<C, Error> {
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    let fields = config.block_parameter_fields().validate()?;
    let tensor = |path: String| {
        layout
            .tensor(&path)
            .ok_or_else(|| Error::backend(format!("missing local decoder layout for {path}")))
    };
    let generated;
    let attention = match scope {
        Some(block) => &block.attention,
        None => {
            generated = topology::AttentionParameters::new(config, layer)?;
            &generated
        }
    };
    let geometry = attention.geometry;
    let (query_width, key_width) = match &attention.input {
        topology::AttentionInputParameters::Split { query, key, .. } => (
            i32::try_from(tensor(query.weight.id.to_string())?.local_shape()[0])
                .map_err(Error::backend)?,
            i32::try_from(tensor(key.weight.id.to_string())?.local_shape()[0])
                .map_err(Error::backend)?,
        ),
        topology::AttentionInputParameters::Fused { projection, .. } => {
            let fused = tensor(projection.weight.id.to_string())?;
            let heads = fused
                .expanded_logical_range(geometry.key_value_heads() as usize)
                .map_err(Error::backend)?
                .len();
            let local = eredu_nn::AttentionHeadGeometry::new(
                i32::try_from(heads)
                    .map_err(Error::backend)?
                    .checked_mul(geometry.query_heads() / geometry.key_value_heads())
                    .ok_or_else(|| Error::backend("local fused query heads overflowed"))?,
                i32::try_from(heads).map_err(Error::backend)?,
                geometry.key_dimensions(),
                geometry.value_dimensions(),
            )?;
            let width = local
                .query_width()
                .checked_add(local.key_width())
                .and_then(|width| width.checked_add(local.value_width()))
                .ok_or_else(|| Error::backend("local fused QKV width overflowed"))?;
            if width as usize != fused.local_shape()[0] {
                return Err(Error::backend(
                    "local fused QKV shape differs from its head partition",
                ));
            }
            (local.query_width(), local.key_width())
        }
    };
    let intermediate = if let Some(block) = scope {
        let feed = block
            .feed_forward
            .as_ref()
            .ok_or_else(|| Error::backend("dense localizer has no declared feed-forward"))?;
        let (weight, fused) = match feed {
            eredu_nn::GatedProductSpec::Split(spec) => (&spec.gate.weight, false),
            eredu_nn::GatedProductSpec::Fused(spec) => (&spec.gate_up.weight, true),
        };
        let width = i32::try_from(tensor(weight.id.to_string())?.local_shape()[0])
            .map_err(Error::backend)?;
        if fused && (width <= 0 || width % 2 != 0) {
            return Err(Error::backend(
                "local fused gate/up shape splits a component",
            ));
        }
        if fused {
            width / 2
        } else {
            width
        }
    } else {
        match config.gated_projection_layout() {
            GatedProjectionLayout::Split => {
                let gate = tensor(format!(
                    "{prefix}.{}.{}.weight",
                    fields.feed_forward, fields.feed_forward_gate
                ))?;
                i32::try_from(gate.local_shape()[0]).map_err(Error::backend)?
            }
            GatedProjectionLayout::Fused { field } => {
                let gate = tensor(format!("{prefix}.{}.{field}.weight", fields.feed_forward))?;
                let width = i32::try_from(gate.local_shape()[0]).map_err(Error::backend)?;
                if width <= 0 || width % 2 != 0 {
                    return Err(Error::backend(
                        "local fused gate/up shape splits a component",
                    ));
                }
                width / 2
            }
        }
    };
    if query_width <= 0
        || key_width <= 0
        || query_width % geometry.key_dimensions() != 0
        || key_width % geometry.key_dimensions() != 0
    {
        return Err(Error::backend("invalid local decoder attention geometry"));
    }
    let mut local = config.clone();
    local.set_local_geometry(
        query_width / geometry.key_dimensions(),
        key_width / geometry.key_dimensions(),
        intermediate,
    )?;
    Ok(local)
}

impl<B, C, P, M, G> DecoderKernel<B, C, P, M, G>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
{
    pub(crate) fn finish_local(
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

    pub(crate) fn finish_local_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let logits = self
            .static_modules
            .decoder_resources_mut()
            .finish_parallel(hidden, parallel, context, instrumentation)?;
        softcap_logits(logits, self.args.output_softcap(), context)
    }
}

impl<B, C, P, M, G> eredu_runtime::ComponentKernel<B> for DecoderKernel<B, C, P, M, G>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;
    type Unit = DecoderUnit<B, P::FeedForward>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::Error> {
        state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(address.index(), context)
    }
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}
impl<B, C, P, M, G> eredu_runtime::ComponentPinnedResources<B> for DecoderKernel<B, C, P, M, G>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
{
    type Modules = M;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        self.static_modules.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        self.static_modules.decoder_resources_mut().visit(visitor)
    }
}
impl<B, C, P, S, M, G> eredu_runtime::ComponentEquations<B, S> for DecoderKernel<B, C, P, M, G>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = self
            .static_modules
            .decoder_resources_mut()
            .input_embedding()?
            .forward(input.tokens, context)?;
        let hidden = scale_token_embeddings(hidden, self.args.embedding_scale(), context)?;
        self.begin_embedded(hidden, input.mask, state, context)
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        ComponentInstrumentation::apply_embedding(self.begin(input, state, context)?, observer)
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let layer = self.schedule.operation(address.index())?.layer();
        let cache = state
            .layer(self.geometry.state_index(layer)?)
            .map_err(Error::backend)?;
        body.forward(None, forward.attention_input(hidden, cache), context)
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let index = self.schedule.operation(address.index())?.layer();
        let path = format!("{}.layers.{}", self.args.parameter_root(), index);
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        body.forward_observed(
            None,
            forward.attention_input(hidden, cache),
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_hidden_instrumented(hidden, context, &mut ComponentInstrumentation::disabled())
    }
    fn projects_final_text_position() -> bool {
        G::FINAL_TEXT_POSITION
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if G::FINAL_TEXT_POSITION {
            self.finish_hidden_instrumented(
                &final_hidden_position(hidden, context)?,
                context,
                &mut ComponentInstrumentation::disabled(),
            )
        } else {
            self.finish(hidden, state, forward, context)
        }
    }
    fn finish_observed<O>(
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
}
impl<B, C, P, S, M, G> eredu_runtime::ParallelComponentKernel<B, S> for DecoderKernel<B, C, P, M, G>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.geometry.parallel_state_layout()?;
        let hidden = B::vocabulary_parallel_lookup(
            self.static_modules
                .decoder_resources_mut()
                .input_embedding()?,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let hidden = scale_token_embeddings(hidden, self.args.embedding_scale(), context)?;
        begin_decoder_hidden::<B, S>(hidden, input.mask, None, state, Some(expected), 0, context)
    }
    fn begin_parallel_observed<'a, O>(
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
        ComponentInstrumentation::apply_embedding(
            self.begin_parallel(input, state, parallel, context)?,
            observer,
        )
    }
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let layer = self.schedule.operation(address.index())?.layer();
        let cache = state
            .layer(self.geometry.state_index(layer)?)
            .map_err(Error::backend)?;
        body.forward_tensor_parallel(
            None,
            forward.attention_input(hidden, cache),
            parallel,
            context,
        )
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let index = self.schedule.operation(address.index())?.layer();
        let path = format!("{}.layers.{}", self.args.parameter_root(), index);
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        body.forward_tensor_parallel_observed(
            None,
            forward.attention_input(hidden, cache),
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, Some(parallel), context)
    }
    fn finish_parallel_observed<O>(
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
impl<B, C, P, S, M, G> eredu_runtime::RoutedComponentKernel<B, S> for DecoderKernel<B, C, P, M, G>
where
    B: GroupedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
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
    fn routing_points(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Self::Error> {
        match self.schedule.operation(address.index())? {
            DecoderOperation::Body { layer } => Ok(self.args.routed_observation_points(
                &format!("{}.layers.{layer}", self.args.parameter_root()),
                layer,
            )),
            DecoderOperation::Terminal { .. } => Ok(None),
        }
    }
    fn execute_routed<R>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::ParameterProvider<B>,
        R::Error: std::fmt::Display,
    {
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let index = self.schedule.operation(address.index())?.layer();
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        body.forward_routed(
            None,
            index,
            forward.attention_input(hidden, cache),
            pass,
            provider,
            context,
        )
    }
    fn execute_routed_observed<R, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        R: eredu_runtime::ParameterProvider<B>,
        R::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let index = self.schedule.operation(address.index())?.layer();
        let path = format!("{}.layers.{}", self.args.parameter_root(), index);
        let points = self.args.routed_observation_points(&path, index);
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        body.forward_routed_observed(
            None,
            index,
            forward.attention_input(hidden, cache),
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            points,
        )
    }
}
impl<B, C, P, S, M, G> eredu_runtime::ParallelRoutedComponentKernel<B, S>
    for DecoderKernel<B, C, P, M, G>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
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
    fn execute_parallel_routed<R>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        R: eredu_runtime::TensorParallelParameterProvider<B>,
        R::Error: std::fmt::Display,
    {
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let index = self.schedule.operation(address.index())?.layer();
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        body.forward_routed_parallel(
            None,
            index,
            forward.attention_input(hidden, cache),
            pass,
            provider,
            parallel,
            context,
        )
    }
    fn execute_parallel_routed_observed<R, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        R: eredu_runtime::TensorParallelParameterProvider<B>,
        R::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let DecoderUnit::Body(body) = unit else {
            let DecoderUnit::Terminal(norm) = unit else {
                unreachable!()
            };
            return norm.forward(hidden, context);
        };
        let index = self.schedule.operation(address.index())?.layer();
        let path = format!("{}.layers.{}", self.args.parameter_root(), index);
        let points = self.args.routed_observation_points(&path, index);
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        body.forward_routed_parallel_observed(
            None,
            index,
            forward.attention_input(hidden, cache),
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            points,
        )
    }
}
impl<B, C, P, S, M, G> eredu_runtime::PartitionComponentKernel<B, S>
    for DecoderKernel<B, C, P, M, G>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
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
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args.hidden_size(),
        ))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = selected.expected;
        let context = selected.native;
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let embedding = self
                    .static_modules
                    .decoder_resources_mut()
                    .input_embedding()?;
                let hidden = match selected.parallel {
                    Some(parallel) => B::vocabulary_parallel_lookup(
                        embedding,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                    None => embedding.forward(tokens, context)?,
                };
                scale_token_embeddings(hidden, self.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        begin_decoder_hidden::<B, S>(
            hidden,
            mask,
            None,
            state,
            expected,
            self.geometry
                .partition_state_ordinal(selected.first_state_ordinal),
            context,
        )
    }
    fn begin_partition_observed<O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = self.begin_partition(input, mask, state, selected)?;
        if embedded {
            forward = ComponentInstrumentation::apply_embedding(forward, observer)?;
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
impl<B, C, P, M, G> eredu_runtime::TextComponentKernel<B> for DecoderKernel<B, C, P, M, G>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
{
    fn supports_chunked_prefill() -> bool {
        G::CHUNKED_PREFILL
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}
impl<B, C, P, S, M, G> eredu_runtime::PipelineComponentKernel<B, S> for DecoderKernel<B, C, P, M, G>
where
    B: eredu_nn::DistributedNeuralBackend,
    C: PartitionedConfig,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
    P::FeedForward: TensorParallelProjectionOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        Ok((input.tokens.dim(0), input.tokens.dim(1)))
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, eredu_runtime::NoAuxiliaryBoundary)>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let transported = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input.tokens),
        };
        self.begin_partition_observed(transported, input.mask, state, selected, observer)
    }
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
    fn output_width(&self) -> i32 {
        self.args.vocabulary_size()
    }
    fn routed_bank_order(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
    ) -> Vec<eredu_runtime::RoutedBankId> {
        match self.schedule.operation(address.index()) {
            Ok(DecoderOperation::Body { layer }) => self.args.routed_bank_order(layer),
            _ => Vec::new(),
        }
    }
    fn routed_bank_tensor_reductions(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        bank: eredu_runtime::RoutedBankId,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Error> {
        let DecoderOperation::Body { layer } = self.schedule.operation(address.index())? else {
            return Err(Error::backend(
                "stateless decoder terminal has no routed bank",
            ));
        };
        self.args
            .routed_bank_tensor_reductions(layer, bank)
            .map(|(before, after)| eredu_runtime::RoutedTensorReductions::hidden(before, after))
    }
}

fn validate_complete_definition<C: Config>(
    args: &C,
    definition: &eredu_runtime::ComponentDefinition,
    state: &StateLayout,
) -> Result<(), Error> {
    let schedule = DecoderSchedule::new(args)?;
    if definition.parameters().graph().groups().len() != 1
        || definition.primary_group() != TEXT_DECODER_EXECUTION_GROUP
        || definition.unit_count(0)? != schedule.len()
        || definition.state_layout() != state
    {
        return Err(eredu_runtime::ComponentDefinitionError::Invalid {
            component: definition.id().into(),
            resource: "decoder construction geometry".into(),
        }
        .into());
    }
    for index in 0..schedule.len() {
        let unit = definition.unit(0, index)?;
        let slots = match schedule.operation(index)? {
            DecoderOperation::Body { layer } => layer..layer + 1,
            DecoderOperation::Terminal { .. } => 0..0,
        };
        if unit.path != schedule.path(args, index)?
            || unit.state != slots
            || definition
                .unit_observation_boundaries(0, index)?
                .is_kernel_owned()
        {
            return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                component: definition.id().into(),
                resource: format!("decoder operation {index} equation geometry"),
            }
            .into());
        }
    }
    Ok(())
}

impl<B, C, P> CompleteDecoderKernel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    /// Materializes pinned resources from an already validated semantic definition.
    pub fn from_definition(
        args: C,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        decoder_scopes: &DecoderDeclaration,
        projection_declarations: Option<&ProjectionDeclarations<P::Parameters>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B, C, P>, Error> {
        args.validate_config()?;
        if let Some(declarations) = projection_declarations {
            declarations.validate_against(&args, definition.parameters())?;
        }
        P::validate(&args)?;
        let expected = P::parameter_description(&args, decoder_scopes, projection_declarations)?;
        definition.validate_parameter_topology(&expected)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        decoder_scopes.validate_for(&args)?;
        let state = decoder_scopes.state_layout().clone();
        validate_complete_definition(&args, &definition, &state)?;
        let geometry = CompleteDecoderGeometry {
            ordinary_state: definition.state_layout().clone(),
            parallel: None,
            state: definition.state_layout().clone(),
            units: definition.parameters().unit_layout().len(),
        };
        let construction = decoder_unit_construction(
            &args,
            &geometry,
            decoder_scopes,
            projection_declarations,
            prepare_projection::<B, C, P>,
        )?;
        let statics = decoder_scopes.statics.clone();
        let static_modules = StaticModules::from_construction(statics, context)?;
        let kernel = Self {
            schedule: DecoderSchedule::new(&args)?,
            args,
            static_modules,
            geometry,
            construction,
            block_factory: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }

    /// Materializes the retained declaration with planner-derived rank-local geometry.
    pub fn from_parallel_definition(
        args: C,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        decoder_scopes: &DecoderDeclaration,
        projection_declarations: Option<&ProjectionDeclarations<P::Parameters>>,
        geometry: LocalGeometry<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B, C, P>, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
        C: PartitionedConfig,
    {
        args.validate_config()?;
        if let Some(declarations) = projection_declarations {
            declarations.validate_against(&args, definition.parameters())?;
        }
        P::validate(&args)?;
        let expected = P::parameter_description(&args, decoder_scopes, projection_declarations)?;
        definition.validate_parameter_topology(&expected)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        geometry
            .validate_with_declaration(&args, decoder_scopes)
            .map_err(Error::backend)?;
        let ordinary_state = definition.state_layout().clone();
        let definition = definition.select_state_layout(geometry.state_layout())?;
        validate_complete_definition(&args, &definition, geometry.state_layout())?;
        let geometry = CompleteDecoderGeometry {
            ordinary_state,
            parallel: Some(std::sync::Arc::new(geometry)),
            state: definition.state_layout().clone(),
            units: definition.parameters().unit_layout().len(),
        };
        let construction = decoder_unit_construction(
            &args,
            &geometry,
            decoder_scopes,
            projection_declarations,
            prepare_projection::<B, C, P>,
        )?;
        let statics = decoder_scopes.statics.clone();
        let parallel = geometry
            .parallel
            .as_ref()
            .expect("selected parallel geometry");
        let static_modules = StaticModules::from_parallel_construction(
            statics,
            parallel.embedding_range.clone(),
            parallel.output_range.clone(),
            context,
        )?;
        let kernel = Self {
            schedule: DecoderSchedule::new(&args)?,
            args,
            static_modules,
            geometry,
            construction,
            block_factory: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }

    /// Returns planner-derived geometry when this is a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry<C>> {
        match self.geometry.parallel.as_ref() {
            Some(geometry) => Some(geometry.as_ref()),
            None => None,
        }
    }

    /// Shares the authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry<C>>> {
        self.geometry.parallel.as_ref().map(std::sync::Arc::clone)
    }
}

impl<B, C, P, M, G> DecoderKernel<B, C, P, M, G>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    M: DecoderStaticResources<B>,
    G: DecoderGeometry<C>,
{
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
        let expected = self.geometry.ordinary_state_layout(&self.args)?;
        begin_decoder_hidden::<B, S>(
            hidden,
            supplied_mask,
            None,
            state,
            Some(expected.as_ref()),
            0,
            context,
        )
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
    pub fn prepare_partition<S>(
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
                let hidden = self
                    .static_modules
                    .decoder_resources_mut()
                    .input_embedding()?
                    .forward(tokens, context)?;
                scale_token_embeddings(hidden, self.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        begin_decoder_hidden::<B, S>(
            hidden,
            supplied_mask,
            None,
            state,
            Some(expected),
            first_state_ordinal,
            context,
        )
    }

    /// Starts one tensor-parallel pipeline partition through the same neutral
    /// entry point, including vocabulary-parallel embedding on the input rank.
    pub fn prepare_partition_parallel<S>(
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
                    self.static_modules
                        .decoder_resources_mut()
                        .input_embedding()?,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                scale_token_embeddings(hidden, self.args.embedding_scale(), context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        begin_decoder_hidden::<B, S>(
            hidden,
            supplied_mask,
            None,
            state,
            Some(expected),
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
        begin_decoder_hidden::<B, S>(
            hidden,
            supplied_mask,
            rotary_embeddings,
            state,
            Some(expected),
            0,
            context,
        )
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
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        block.body.forward(
            block.output_norm.as_mut(),
            forward.attention_input(hidden, cache),
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
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        block.body.forward_with_feed_forward(
            block.output_norm.as_mut(),
            forward.attention_input(hidden, cache),
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
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        block.body.forward_tensor_parallel(
            block.output_norm.as_mut(),
            forward.attention_input(hidden, cache),
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
        let cache = state
            .layer(self.geometry.state_index(index)?)
            .map_err(Error::backend)?;
        block.body.forward_tensor_parallel_with_feed_forward(
            block.output_norm.as_mut(),
            forward.attention_input(hidden, cache),
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
        self.finish_local_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
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
