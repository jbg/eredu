//! Shared backend-neutral decoder mechanics.
//!
//! Architecture families retain configuration, checkpoint naming, identity, and
//! policy while reusing these statically dispatched decoder operations.

use std::ops::Range;

// Shared by cold architecture requirements and the concrete layered executor.
pub(crate) const CHUNKED_TEXT_PREFILL: bool = true;

pub(crate) mod attention_partition;
mod boundary_modules;
pub use boundary_modules::BoundaryModules;
/// Shared-weight repeated stacks with independent state for each invocation.
pub(crate) mod repeated;
pub(crate) mod topology;

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
use eredu_runtime::PartitionComponentKernel as _;
use eredu_runtime::{
    module_parameter_group, partitioned_module_parameter_group, partitioned_projection_group,
    segmented_projection_group, ArchitectureParameterDescription, ExecutionGraph,
    ExecutionUnitLayout, LayerRuntimeState, LayeredForwardState, LayeredPartitionInput,
    LocalModelLayout, MemberSharding, OwnedParameterGroupSpec, ParallelPlanError,
    ParameterGroupOwner, ParameterGroupSpec, ParameterRole, ProjectionSharding, StateLayout,
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
        dense_local_block_config(self, layer, layout)
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
        .map(|index| local_block(config, config.block_parameter_owner(index), layout))
        .collect::<Result<Vec<_>, _>>()?;
    let key_value_heads = blocks.iter().map(Config::num_key_value_heads);
    let state_layout = StateLayout::new(
        cache_layout_with_key_value_heads(config, key_value_heads)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let root = config.parameter_root();
    let (embedding_range, output_range) = StaticModuleSpec::from_config(config).vocabulary_ranges(
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
    geometry.validate_for(config)?;
    Ok(geometry)
}

/// Declares the complete mutable-state geometry consumed by resident or bounded execution.
pub fn cache_layout_with_key_value_heads<C: Config>(
    config: &C,
    key_value_heads: impl IntoIterator<Item = i32>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    let key_value_heads = key_value_heads.into_iter().collect::<Vec<_>>();
    if key_value_heads.len() != layers {
        return Err(Error::backend(format!(
            "decoder cache geometry has {} layers, expected {layers}",
            key_value_heads.len()
        )));
    }
    let policies = config
        .attention_schedule()
        .iter()
        .zip(key_value_heads)
        .map(|(attention, key_value_heads)| {
            LayerCachePolicy::key_value(*attention, key_value_heads, config.head_dim())
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
        let fields = config.block_parameter_fields().validate()?;
        let prefix = format!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.attention
        );
        let head = config.head_dim();
        let query_heads = config.num_attention_heads();
        let key_value_heads = config.num_key_value_heads();
        let projection_specs = topology::attention_projections(config, layer)?;
        let linear = |field: &str| {
            let spec = projection_specs
                .iter()
                .find(|(name, _)| name == field)
                .ok_or_else(|| Error::backend("attention construction projection is absent"))?;
            B::linear(spec.1.clone(), context)
        };
        let policy = config.attention_schedule().get(layer).ok_or_else(|| {
            Error::backend(format!(
                "decoder attention schedule has no policy for layer {layer}"
            ))
        })?;
        let query_width = query_heads
            .checked_mul(head)
            .ok_or_else(|| Error::backend("decoder query projection width overflowed"))?;
        let key_value_width = key_value_heads
            .checked_mul(head)
            .ok_or_else(|| Error::backend("decoder key/value projection width overflowed"))?;
        let input_projection = match config.attention_projection_layout() {
            AttentionProjectionLayout::Split if config.external_attention_value(layer) => {
                AttentionInputProjection::ExternalValue {
                    query: linear(fields.attention_query)?,
                    key: linear(fields.attention_key)?,
                }
            }
            AttentionProjectionLayout::Split => AttentionInputProjection::Split {
                query: linear(fields.attention_query)?,
                key: linear(fields.attention_key)?,
                value: linear(fields.attention_value)?,
            },
            AttentionProjectionLayout::Fused { field } => {
                if field.trim().is_empty() {
                    return Err(Error::backend(
                        "fused QKV projection field must not be empty",
                    ));
                }
                let biases = [
                    config.attention_bias(AttentionProjection::Query),
                    config.attention_bias(AttentionProjection::Key),
                    config.attention_bias(AttentionProjection::Value),
                ];
                if biases.iter().any(|bias| *bias != biases[0]) {
                    return Err(Error::backend(
                        "fused QKV projection requires identical query/key/value bias policy",
                    ));
                }
                let layout = FusedProjectionLayout::new([
                    FusedProjectionSegment::new("query", query_width)?,
                    FusedProjectionSegment::new("key", key_value_width)?,
                    FusedProjectionSegment::new("value", key_value_width)?,
                ])?;
                let projection = linear(field)?;
                AttentionInputProjection::Fused(FusedAttentionProjection { projection, layout })
            }
        };
        Ok(Self {
            query_position_scaling: config.query_position_scaling(),
            output_gate: config
                .attention_output_gate()
                .map(|(field, _)| linear(field))
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
            output: linear(fields.attention_output)?,
            sinks: config
                .learned_attention_sinks()
                .then(|| {
                    Parameter::unloaded(
                        parameter_spec(config, format!("{prefix}.{}", fields.attention_sinks))
                            .map_err(Error::backend)?,
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
                        parameter_spec(
                            config,
                            format!("{prefix}.{}.weight", fields.attention_query_norm),
                        )
                        .map_err(Error::backend)?,
                    );
                    if config.query_key_norm_per_head_weights() {
                        spec = spec.with_groups(query_heads)?;
                    } else {
                        spec.groups = config.query_key_normalization_groups();
                    }
                    spec.scale = normalization_scale(
                        parameter_spec(
                            config,
                            format!("{prefix}.{}.weight", fields.attention_query_norm),
                        )
                        .map_err(Error::backend)?,
                        config.query_key_normalization_offset(),
                    );
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
                        parameter_spec(
                            config,
                            format!("{prefix}.{}.weight", fields.attention_key_norm),
                        )
                        .map_err(Error::backend)?,
                    );
                    if config.query_key_norm_per_head_weights() {
                        spec = spec.with_groups(key_value_heads)?;
                    } else {
                        spec.groups = config.query_key_normalization_groups();
                    }
                    spec.scale = normalization_scale(
                        parameter_spec(
                            config,
                            format!("{prefix}.{}.weight", fields.attention_key_norm),
                        )
                        .map_err(Error::backend)?,
                        config.query_key_normalization_offset(),
                    );
                    B::normalization(spec, context)
                })
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
            return match output_gate {
                Some(gate) => attended.multiply(
                    &self.output_gate_activation.apply::<B>(gate, context)?,
                    context,
                ),
                None => Ok(attended),
            };
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

/// Shared dense SwiGLU feed-forward network.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: NeuralBackend> {
    /// Split or fused gate/up input projection.
    pub input_projection: GatedInputProjection<B>,
    /// Down projection.
    pub down: B::Linear,
    /// Optional shared pre-activation bound.
    #[parameter(skip)]
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
        let fields = config.block_parameter_fields().validate()?;
        let projection_specs = topology::gated_projections(config, layer)?;
        let build = |field: &str| {
            let spec = projection_specs
                .iter()
                .find(|(name, _)| name == field)
                .ok_or_else(|| Error::backend("gated construction projection is absent"))?;
            B::linear(spec.1.clone(), context)
        };
        let input_projection = match config.gated_projection_layout() {
            GatedProjectionLayout::Split => GatedInputProjection::Split {
                gate: build(fields.feed_forward_gate)?,
                up: build(fields.feed_forward_up)?,
            },
            GatedProjectionLayout::Fused { field } => {
                if field.trim().is_empty() {
                    return Err(Error::backend(
                        "fused gate/up projection field must not be empty",
                    ));
                }
                let layout = FusedProjectionLayout::new([
                    FusedProjectionSegment::new("gate", config.intermediate_size())?,
                    FusedProjectionSegment::new("up", config.intermediate_size())?,
                ])?;
                GatedInputProjection::Fused(FusedGatedProjection {
                    projection: build(field)?,
                    layout,
                })
            }
        };
        Ok(Self {
            input_projection,
            down: build(fields.feed_forward_output)?,
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
        Self::from_factory::<C, DenseBlockFactory>(config, config, layer, context)
    }
}

impl<B: NeuralBackend, F> TransformerBlockBody<B, F> {
    /// Constructs a body from shared attention/residual policy and a typed
    /// feed-forward factory. Global routing and rank-local bank geometry remain distinct.
    pub fn from_factory<C: Config, P: BlockFactory<B, C, FeedForward = F>>(
        global: &C,
        local: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        P::validate_partitioned(global, local)?;
        Self::construct_with_terminal(
            local,
            layer,
            context,
            || Ok(()),
            || P::build_partitioned_feed_forward(global, local, layer, context),
        )
        .map(|(body, ())| body)
    }

    fn construct_with_terminal<C: Config, T>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
        terminal: impl FnOnce() -> Result<T, Error>,
        feed_forward: impl FnOnce() -> Result<F, Error>,
    ) -> Result<(Self, T), Error> {
        let fields = config.block_parameter_fields().validate()?;
        let norm = |name: String| {
            B::normalization(
                NormalizationConstructionSpec {
                    reduction: eredu_nn::NormalizationReduction::Native,
                    groups: config.normalization_groups(),
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    scale: normalization_scale(
                        parameter_spec(config, name).map_err(Error::backend)?,
                        config.normalization_offset(),
                    ),
                },
                context,
            )
        };
        let attention_output_norm = config
            .attention_output_normalization(layer)
            .map(&norm)
            .transpose()?;
        let feed_forward_output_norm = config
            .feed_forward_output_normalization(layer)
            .map(&norm)
            .transpose()?;
        let terminal = terminal()?;
        Ok((
            Self {
                attention_output_norm,
                feed_forward_output_norm,
                self_attention: Attention::new(config, layer, context)?,
                mlp: feed_forward()?,
                input_norm: norm(format!(
                    "{}.layers.{layer}.{}.weight",
                    config.parameter_root(),
                    fields.input_norm
                ))?,
                post_attention_norm: norm(format!(
                    "{}.layers.{layer}.{}.weight",
                    config.parameter_root(),
                    fields.post_attention_norm
                ))?,
            },
            terminal,
        ))
    }
}

/// Constructs only a declared invocation's terminal normalization.
/// Body parameters and mutable attention state are not allocated by this operation.
pub fn terminal_normalization<B: NeuralBackend>(
    config: &impl Config,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<B::Normalization>, Error> {
    config
        .block_output_normalization(layer)
        .map(|name| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    config.hidden_size(),
                    config.rms_norm_epsilon(),
                    parameter_spec(config, name).map_err(Error::backend)?,
                ),
                context,
            )
        })
        .transpose()
}

impl<B: NeuralBackend> TransformerBlock<B, Mlp<B>> {
    /// Constructs one unloaded dense body and its optional terminal normalization.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_factory::<C, DenseBlockFactory>(config, config, layer, context)
    }
}

impl<B: NeuralBackend, F> TransformerBlock<B, F> {
    /// Assembles the shared body and declared terminal without changing their
    /// independent construction interfaces or canonical parameter names.
    pub fn from_factory<C: Config, P: BlockFactory<B, C, FeedForward = F>>(
        global: &C,
        local: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        P::validate_partitioned(global, local)?;
        let (body, output_norm) = TransformerBlockBody::construct_with_terminal(
            local,
            layer,
            context,
            || terminal_normalization::<B>(local, layer, context),
            || P::build_partitioned_feed_forward(global, local, layer, context),
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

/// Declares attention and normalization groups shared by dense and routed blocks.
pub fn body_common_parallel_parameter_groups<B: NeuralBackend, F>(
    block: &TransformerBlockBody<B, F>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let attention_prefix = format!("{prefix}.{}", fields.attention);
    let query_heads = usize::try_from(config.num_attention_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder query-head count exceeds usize".into())
    })?;
    let key_value_heads = usize::try_from(config.num_key_value_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder key/value-head count exceeds usize".into())
    })?;
    let head_dimension = usize::try_from(config.head_dim()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder head dimension exceeds usize".into())
    })?;
    if head_dimension == 0 || key_value_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "decoder attention geometry q={query_heads}, kv={key_value_heads}, dim={head_dimension} does not form positive integral GQA groups"
        )));
    }
    let head_partition = attention_partition::AttentionPartition::new(config, layer)?;
    let attention_units = head_partition.preferred_units();
    let attention = match &block.self_attention.input_projection {
        AttentionInputProjection::ExternalValue { query, key } => {
            let mut projections = vec![
                (query, ProjectionSharding::Column),
                (key, ProjectionSharding::Column),
                (&block.self_attention.output, ProjectionSharding::Row),
            ];
            if let Some(gate) = &block.self_attention.output_gate {
                projections.push((gate, ProjectionSharding::Column));
            }
            partitioned_projection_group::<B::Tensor, B::Linear>(
                format!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &projections,
                attention_units,
            )?
        }
        AttentionInputProjection::Split { query, key, value } => {
            let mut projections = vec![
                (query, ProjectionSharding::Column),
                (key, ProjectionSharding::Column),
                (value, ProjectionSharding::Column),
                (&block.self_attention.output, ProjectionSharding::Row),
            ];
            if let Some(gate) = &block.self_attention.output_gate {
                projections.push((gate, ProjectionSharding::Column));
            }
            partitioned_projection_group::<B::Tensor, B::Linear>(
                format!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &projections,
                attention_units,
            )?
        }
        AttentionInputProjection::Fused(fused) => {
            segmented_projection_group::<B::Tensor, B::Linear>(
                format!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &fused.projection,
                &block.self_attention.output,
                fused_projection_ranges(&fused.layout)?,
                attention_units,
            )?
        }
    };

    let input_norm = module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.{}", fields.input_norm),
        ParameterRole::Replicated,
        &block.input_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?;
    let post_attention_norm = module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.{}", fields.post_attention_norm),
        ParameterRole::Replicated,
        &block.post_attention_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?;
    let mut groups = vec![attention];
    if let Some(sinks) = &block.self_attention.sinks {
        groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
            format!("{attention_prefix}.{}", fields.attention_sinks),
            ParameterRole::AttentionHeads,
            query_heads,
            sinks,
            |_, shape| {
                if shape != [query_heads] {
                    return Err(ParallelPlanError::InvalidTensor(format!(
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
            let name = format!("{attention_prefix}.{field}");
            groups.push(if config.query_key_norm_per_head_weights() {
                partitioned_module_parameter_group::<B::Tensor, _>(
                    name,
                    ParameterRole::AttentionHeads,
                    heads,
                    norm,
                    |_, _| Ok(MemberSharding::Partitioned { axis: 0 }),
                )?
            } else {
                module_parameter_group::<B::Tensor, _>(
                    name,
                    ParameterRole::Replicated,
                    norm,
                    |_, _| Ok(MemberSharding::Replicated),
                )?
            });
        }
    }
    for (name, norm) in [
        (
            config.attention_output_normalization(layer),
            &block.attention_output_norm,
        ),
        (
            config.feed_forward_output_normalization(layer),
            &block.feed_forward_output_norm,
        ),
    ] {
        if let (Some(name), Some(norm)) = (name, norm) {
            groups.push(module_parameter_group::<B::Tensor, _>(
                name.trim_end_matches(".weight"),
                ParameterRole::Replicated,
                norm,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
        }
    }
    groups.extend([input_norm, post_attention_norm]);
    groups
        .into_iter()
        .map(|group| head_partition.apply(group, |name| config.linear_format(name)))
        .collect()
}

/// Projects common body resources and the enclosing invocation's terminal normalization.
pub fn block_common_parallel_parameter_groups<B: NeuralBackend, F>(
    block: &TransformerBlock<B, F>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    with_terminal_parameter_group::<B>(
        body_common_parallel_parameter_groups(&block.body, config, layer)?,
        block.output_norm.as_ref(),
        config,
        layer,
    )
}

fn with_terminal_parameter_group<B: NeuralBackend>(
    mut groups: Vec<ParameterGroupSpec>,
    terminal: Option<&B::Normalization>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    if let Some(norm) = terminal {
        let name = config.block_output_normalization(layer).ok_or_else(|| {
            ParallelPlanError::InvalidGroup(
                "block output normalization is absent from configuration".into(),
            )
        })?;
        groups.insert(
            1,
            module_parameter_group::<B::Tensor, _>(
                name.trim_end_matches(".weight"),
                ParameterRole::Replicated,
                norm,
                |_, _| Ok(MemberSharding::Replicated),
            )?,
        );
    }
    Ok(groups)
}

/// Projects every parameter owned by one dense body, excluding invocation terminals.
pub fn body_parallel_parameter_groups<B: NeuralBackend>(
    body: &TransformerBlockBody<B>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = body_common_parallel_parameter_groups(body, config, layer)?;
    groups.push(dense_mlp_parallel_parameter_group(
        &body.mlp, config, layer,
    )?);
    Ok(groups)
}

/// Declares the dense SwiGLU placement group shared by dense decoder families.
pub fn dense_mlp_parallel_parameter_group<B: NeuralBackend>(
    mlp: &Mlp<B>,
    config: &impl Config,
    layer: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let prefix = format!(
        "{}.layers.{layer}.{}",
        config.parameter_root(),
        fields.feed_forward
    );
    let intermediate = usize::try_from(config.intermediate_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder feed-forward width exceeds usize".into())
    })?;
    let output_format =
        config.linear_format(&format!("{prefix}.{}.weight", fields.feed_forward_output));
    let units =
        crate::linear_format::input_partition_units(&prefix, intermediate, 1, output_format)?;
    let group = match &mlp.input_projection {
        GatedInputProjection::Split { gate, up } => {
            partitioned_projection_group::<B::Tensor, B::Linear>(
                format!("{prefix}.projections"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, ProjectionSharding::Column),
                    (up, ProjectionSharding::Column),
                    (&mlp.down, ProjectionSharding::Row),
                ],
                units,
            )
        }
        GatedInputProjection::Fused(fused) => segmented_projection_group::<B::Tensor, B::Linear>(
            format!("{prefix}.projections"),
            ParameterRole::FeedForwardIntermediate,
            &fused.projection,
            &mlp.down,
            fused_projection_ranges(&fused.layout)?,
            units,
        ),
    }?;
    crate::linear_format::dense_ffn_partition_tail(group, intermediate, output_format, |name| {
        config.linear_format(name)
    })
}

fn fused_projection_ranges(
    layout: &FusedProjectionLayout,
) -> Result<Vec<std::ops::Range<usize>>, ParallelPlanError> {
    let mut start = 0usize;
    layout
        .segments()
        .iter()
        .map(|segment| {
            let width = usize::try_from(segment.width()).map_err(|_| {
                ParallelPlanError::InvalidTensor(format!(
                    "fused projection segment {} exceeds usize",
                    segment.name()
                ))
            })?;
            let end = start.checked_add(width).ok_or_else(|| {
                ParallelPlanError::InvalidTensor(
                    "fused projection segment ranges overflowed usize".into(),
                )
            })?;
            let range = start..end;
            start = end;
            Ok(range)
        })
        .collect()
}

/// Declares every rank-local placement group for one dense shared decoder block.
pub fn layer_parallel_parameter_groups<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    with_terminal_parameter_group::<B>(
        body_parallel_parameter_groups(&block.body, config, layer)?,
        block.output_norm.as_ref(),
        config,
        layer,
    )
}

/// Derives the complete dense-decoder parameter topology from normalized
/// configuration without constructing backend modules.
pub fn dense_parameter_description(
    config: &impl Config,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    parameter_description(config, |_| true, |_| Ok(None))
}

/// Declares a decoder with optional provider-owned values and
/// replacement feed-forward groups, without constructing backend tensors.
pub(crate) fn parameter_description(
    config: &impl Config,
    ordinary_value_projection: impl Fn(usize) -> bool,
    replacement_groups: impl Fn(usize) -> Result<Option<Vec<ParameterGroupSpec>>, ParallelPlanError>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    parameter_description_with_vocabulary(
        config,
        ordinary_value_projection,
        replacement_groups,
        true,
    )
}

/// Neutral router and packed-bank declaration for one ordinary routed block.
pub(crate) struct RoutedBlockParameters {
    pub selector_name: String,
    pub bank_name: String,
    pub selector: eredu_nn::TopKGroupSelectorSpec,
    pub bank: eredu_nn::GroupedGatedProductSpec,
}

/// Projects ordinary routed blocks into the common decoder topology.
pub(crate) fn routed_parameter_description(
    config: &impl Config,
    block: impl Fn(usize) -> Result<RoutedBlockParameters, Error>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let invalid = |error: Error| ParallelPlanError::InvalidGroup(error.to_string());
    config.validate_config().map_err(invalid)?;
    let layers = usize::try_from(config.num_hidden_layers())
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let mut groups = Vec::with_capacity(layers);
    let mut formats = std::collections::BTreeMap::new();
    for layer in 0..layers {
        let block = block(layer).map_err(invalid)?;
        block.selector.validate().map_err(invalid)?;
        block.bank.validate().map_err(invalid)?;
        if block.selector.selection().group_count() != block.bank.group_count()
            || block.selector.input_dimensions() != block.bank.input_dimensions()
            || block.bank.input_dimensions() != config.hidden_size()
            || block.bank.output_dimensions() != config.hidden_size()
        {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "routed decoder unit {layer} has incompatible selector and bank geometry"
            )));
        }
        formats.insert(
            block.selector.weight().id.as_str().to_owned(),
            block.selector.format().encoding(),
        );
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = block.bank.layout()
        else {
            return Err(ParallelPlanError::InvalidGroup(
                "ordinary routed decoder topology requires packed expert projections".into(),
            ));
        };
        for projection in [gate_up, down] {
            formats.insert(
                projection.weight().id.as_str().to_owned(),
                projection.format().encoding(),
            );
        }
        let units = eredu_runtime::aligned_partition_units(
            &block.bank_name,
            block.bank.intermediate_dimensions() as usize,
            1,
            crate::linear_format::input_partition_alignment(down.format().encoding()) as usize,
        )?;
        groups.push(vec![
            eredu_runtime::top_k_group_selector_logical_group(
                block.selector_name,
                &block.selector,
            )?,
            eredu_runtime::packed_gated_product_logical_group(block.bank_name, &block.bank, units)?,
        ]);
    }
    parameter_description_with_formats(
        config,
        |_| true,
        |layer| Ok(Some(groups[layer].clone())),
        true,
        |name| {
            formats
                .get(name)
                .copied()
                .unwrap_or_else(|| config.linear_format(name))
        },
    )
}

/// Derives block and final-normalization topology for supplied-embedding execution,
/// without declaring a vocabulary table or a language-model output head.
pub fn embedded_parameter_description(
    config: &impl Config,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    parameter_description_with_vocabulary(config, |_| true, |_| Ok(None), false)
}

fn parameter_description_with_vocabulary(
    config: &impl Config,
    ordinary_value_projection: impl Fn(usize) -> bool,
    replacement_groups: impl Fn(usize) -> Result<Option<Vec<ParameterGroupSpec>>, ParallelPlanError>,
    include_vocabulary: bool,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    parameter_description_with_formats(
        config,
        ordinary_value_projection,
        replacement_groups,
        include_vocabulary,
        |name| config.linear_format(name),
    )
}

pub(crate) fn parameter_description_with_formats(
    config: &impl Config,
    ordinary_value_projection: impl Fn(usize) -> bool,
    replacement_groups: impl Fn(usize) -> Result<Option<Vec<ParameterGroupSpec>>, ParallelPlanError>,
    include_vocabulary: bool,
    format: impl Fn(&str) -> LinearFormat,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    config
        .validate_config()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let dimension = |name: &str, value: i32| {
        usize::try_from(value)
            .map_err(|_| ParallelPlanError::InvalidTensor(format!("decoder {name} exceeds usize")))
    };
    let hidden = dimension("hidden width", config.hidden_size())?;
    let schedule = DecoderSchedule::new(config)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
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
    let layout = ExecutionUnitLayout::new(&graph, [schedule.len()])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let group_id = layout
        .group_id(0)
        .expect("dense decoder layout contains its text group")
        .clone();
    let root = config.parameter_root();
    let statics = StaticModuleSpec::from_config(config);
    let mut owned = if include_vocabulary {
        statics
            .logical_pinned_parameter_groups(
                &format!("{root}.embed_tokens"),
                &format!("{root}.norm"),
                "lm_head",
            )?
            .into_iter()
            .map(|(role, group)| {
                let owner = if role == "embedding" && statics.tied_head {
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
            ParameterGroupSpec::new(
                format!("{root}.norm"),
                ParameterRole::Replicated,
                [eredu_runtime::ParameterMemberSpec::new(
                    &statics.normalization_weight,
                    vec![hidden],
                    MemberSharding::Replicated,
                )],
            )?,
        )]
    };

    for (owner, consumers) in schedule.resources() {
        let operation = schedule
            .operation(owner)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let layer = operation.layer();
        if let DecoderOperation::Terminal { .. } = operation {
            let name = config
                .block_output_normalization(layer)
                .expect("declared terminal");
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit_consumers(group_id.clone(), owner, consumers),
                ParameterGroupSpec::new(
                    name.trim_end_matches(".weight"),
                    ParameterRole::Replicated,
                    [eredu_runtime::ParameterMemberSpec::new(
                        &name,
                        vec![hidden],
                        MemberSharding::Replicated,
                    )],
                )?,
            ));
            continue;
        }
        let prefix = format!("{root}.layers.{layer}");
        let attention_prefix = format!("{prefix}.{}", fields.attention);
        let head_partition = attention_partition::AttentionPartition::new(config, layer)?;
        let attention_units = head_partition.preferred_units();
        let mut attention_members = Vec::new();
        let mut projection =
            |field: &str, shape: Vec<usize>, sharding: MemberSharding, bias: bool| {
                let output_width = shape[0];
                attention_members.push(eredu_runtime::ParameterMemberSpec::new(
                    format!("{attention_prefix}.{field}.weight"),
                    shape,
                    sharding.clone(),
                ));
                if bias {
                    let bias_sharding = if matches!(
                        sharding,
                        MemberSharding::Partitioned { axis: 0 }
                            | MemberSharding::PartitionedSegments { axis: 0, .. }
                    ) {
                        sharding
                    } else {
                        MemberSharding::Replicated
                    };
                    attention_members.push(eredu_runtime::ParameterMemberSpec::new(
                        format!("{attention_prefix}.{field}.bias"),
                        vec![output_width],
                        bias_sharding,
                    ));
                }
            };
        match config.attention_projection_layout() {
            AttentionProjectionLayout::Split => {
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
            }
            AttentionProjectionLayout::Fused { field } => {
                if !ordinary_value_projection(layer) {
                    return Err(ParallelPlanError::InvalidGroup(
                        "fused QKV requires an ordinary value projection".into(),
                    ));
                }
                let key_end = query_width.checked_add(key_value_width).ok_or_else(|| {
                    ParallelPlanError::InvalidTensor("fused QK width overflowed usize".into())
                })?;
                let end = key_end.checked_add(key_value_width).ok_or_else(|| {
                    ParallelPlanError::InvalidTensor("fused QKV width overflowed usize".into())
                })?;
                projection(
                    field,
                    vec![end, hidden],
                    MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: vec![0..query_width, query_width..key_end, key_end..end],
                    },
                    config.attention_bias(AttentionProjection::Query),
                );
            }
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

        let feed_forward_groups = match replacement_groups(layer)? {
            Some(groups) => groups,
            None => {
                let feed_forward_prefix = format!("{prefix}.{}", fields.feed_forward);
                let feed_forward_units = crate::linear_format::input_partition_units(
                    &feed_forward_prefix,
                    intermediate,
                    1,
                    format(&format!(
                        "{feed_forward_prefix}.{}.weight",
                        fields.feed_forward_output
                    )),
                )?;
                let mut feed_forward_members = Vec::new();
                let inputs = match config.gated_projection_layout() {
                    GatedProjectionLayout::Split => vec![
                        (
                            fields.feed_forward_gate,
                            intermediate,
                            MemberSharding::Partitioned { axis: 0 },
                        ),
                        (
                            fields.feed_forward_up,
                            intermediate,
                            MemberSharding::Partitioned { axis: 0 },
                        ),
                    ],
                    GatedProjectionLayout::Fused { field } => {
                        let width = intermediate.checked_mul(2).ok_or_else(|| {
                            ParallelPlanError::InvalidTensor(
                                "fused gate/up width overflowed usize".into(),
                            )
                        })?;
                        vec![(
                            field,
                            width,
                            MemberSharding::PartitionedSegments {
                                axis: 0,
                                segments: vec![0..intermediate, intermediate..width],
                            },
                        )]
                    }
                };
                for (field, width, sharding) in inputs {
                    feed_forward_members.push(eredu_runtime::ParameterMemberSpec::new(
                        format!("{feed_forward_prefix}.{field}.weight"),
                        vec![width, hidden],
                        sharding.clone(),
                    ));
                    if config.mlp_bias() {
                        feed_forward_members.push(eredu_runtime::ParameterMemberSpec::new(
                            format!("{feed_forward_prefix}.{field}.bias"),
                            vec![width],
                            sharding,
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
                    format(&format!(
                        "{feed_forward_prefix}.{}.weight",
                        fields.feed_forward_output
                    )),
                    |name| format(name),
                )?;
                vec![feed_forward]
            }
        };
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
        if config.learned_attention_sinks() {
            let name = format!("{attention_prefix}.{}", fields.attention_sinks);
            unit_groups.push(ParameterGroupSpec::partitioned(
                name.clone(),
                ParameterRole::AttentionHeads,
                query_heads,
                [eredu_runtime::ParameterMemberSpec::new(
                    name,
                    vec![query_heads],
                    MemberSharding::Partitioned { axis: 0 },
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
        for name in [
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
        unit_groups.extend([input_norm, post_attention_norm]);
        unit_groups.extend(feed_forward_groups);
        for group in unit_groups {
            let group = head_partition.apply(group, |name| format(name))?;
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit_consumers(
                    group_id.clone(),
                    owner,
                    consumers.iter().copied(),
                ),
                group,
            ));
        }
    }

    let expanded = eredu_runtime::expand_owned_linear_format_parameter_groups(owned, |member| {
        crate::linear_format::standard_parallel_linear_format(member, format(member.target()))
    })?;
    ArchitectureParameterDescription::from_owned(&graph, &layout, expanded)
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
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{parameter_root}.embed_tokens"),
            ParameterRole::Vocabulary,
            embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "decoder embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            format!("{parameter_root}.norm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    if let Some(head) = head {
        groups.push(module_parameter_group::<B::Tensor, _>(
            "lm_head",
            ParameterRole::Vocabulary,
            head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "decoder language-model head parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
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
pub(crate) struct StaticModuleConstruction {
    embedding: EmbeddingSpec,
    head: Option<LinearSpec>,
    normalization: NormalizationConstructionSpec,
}

impl StaticModuleConstruction {
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
        Ok(Self {
            embedding,
            head,
            normalization,
        })
    }

    /// Projects logical embedding, normalization and readout resources before
    /// encoding expansion. Families retain group names and format admission;
    /// explicit resource roles remain independent of group ordering.
    pub(crate) fn logical_pinned_parameter_groups(
        &self,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
        let embedding = &self.embedding;
        let head = &self.head;
        let normalization_weight = match &self.normalization.scale {
            eredu_nn::NormalizationScale::Learned(weight)
            | eredu_nn::NormalizationScale::LearnedOffset { weight, .. } => weight,
            eredu_nn::NormalizationScale::Unit => {
                return Err(ParallelPlanError::InvalidGroup(
                    "pinned normalization catalog requires a learned scale".into(),
                ));
            }
        };
        let group = |name: &str, role, target: &str, shape, sharding| {
            ParameterGroupSpec::new(
                name,
                role,
                [eredu_runtime::ParameterMemberSpec::new(
                    target, shape, sharding,
                )],
            )
        };
        let mut groups = vec![
            (
                "embedding",
                group(
                    embedding_group,
                    ParameterRole::Vocabulary,
                    embedding.weight.id.as_str(),
                    vec![embedding.vocabulary as usize, embedding.dimensions as usize],
                    MemberSharding::Balanced { axis: 0 },
                )?,
            ),
            (
                "norm",
                group(
                    normalization_group,
                    ParameterRole::Replicated,
                    normalization_weight.id.as_str(),
                    vec![embedding.dimensions as usize],
                    MemberSharding::Replicated,
                )?,
            ),
        ];
        if let Some(head) = head {
            let mut members = vec![eredu_runtime::ParameterMemberSpec::new(
                head.weight.id.as_str(),
                vec![head.output as usize, head.input as usize],
                MemberSharding::Balanced { axis: 0 },
            )];
            if let Some(bias) = &head.bias {
                members.push(eredu_runtime::ParameterMemberSpec::new(
                    bias.id.as_str(),
                    vec![head.output as usize],
                    MemberSharding::Balanced { axis: 0 },
                ));
            }
            groups.push((
                "output",
                ParameterGroupSpec::new(output_group, ParameterRole::Vocabulary, members)?,
            ));
        }
        Ok(groups)
    }

    /// Projects physical pinned groups from the same admitted formats used by
    /// construction. Resource roles survive encoding and companion expansion.
    pub(crate) fn pinned_parameter_groups(
        &self,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
        eredu_runtime::expand_tagged_linear_format_parameter_groups(
            self.logical_pinned_parameter_groups(
                embedding_group,
                normalization_group,
                output_group,
            )?,
            |member| {
                let format = if member.target() == self.embedding.weight.id.as_str() {
                    Some(self.embedding.format.clone())
                } else {
                    self.head
                        .as_ref()
                        .filter(|head| member.target() == head.weight.id.as_str())
                        .map(|head| head.format.clone())
                };
                Ok(format)
            },
        )
    }
}

/// Physical ownership and placement of one ordinary pinned vocabulary resource.
pub(crate) enum StaticModulePlacement {
    Omitted,
    Local,
    VocabularyParallel(VocabularyParallelRange),
}
impl StaticModulePlacement {
    fn validate_rows(&self, rows: i32) -> Result<(), Error> {
        if let Self::VocabularyParallel(range) = self {
            range.validate_global_rows(rows)?;
        }
        Ok(())
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

    /// Projects logical pinned roles from the validated construction declaration.
    pub(crate) fn logical_pinned_parameter_groups(
        &self,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
        self.construction_specs()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .logical_pinned_parameter_groups(embedding_group, normalization_group, output_group)
    }

    /// Projects physical storage and companions from the same construction declaration.
    pub(crate) fn pinned_parameter_groups(
        &self,
        embedding_group: &str,
        normalization_group: &str,
        output_group: &str,
    ) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
        self.construction_specs()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .pinned_parameter_groups(embedding_group, normalization_group, output_group)
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
        let groups =
            self.pinned_parameter_groups(embedding_group, normalization_group, output_group)?;
        let global_vocabulary = usize::try_from(self.vocabulary)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let range = |role| -> Result<_, ParallelPlanError> {
            let (_, group) = groups
                .iter()
                .find(|(resource, _)| *resource == role)
                .ok_or_else(|| {
                    ParallelPlanError::InvalidGroup(format!(
                        "static declaration has no {role} resource"
                    ))
                })?;
            Ok(VocabularyParallelRange {
                global_vocabulary,
                local: eredu_runtime::balanced_parameter_group_range(
                    group,
                    layout,
                    0,
                    global_vocabulary,
                )?,
            })
        };
        Ok((
            range("embedding")?,
            (!self.tied_head).then(|| range("output")).transpose()?,
        ))
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

    /// Actual output projection, including the embedding owner and encoding when tied.
    pub(crate) fn output_topology(
        &self,
    ) -> Result<eredu_runtime::execution_topology::ProjectionTopology, Error> {
        if self.hidden_size <= 0 || self.vocabulary <= 0 {
            return Err(Error::backend("static module dimensions must be positive"));
        }
        Ok(eredu_runtime::execution_topology::ProjectionTopology {
            arithmetic: eredu_nn::LinearArithmetic::Native,
            input: self.hidden_size as u64,
            output: self.vocabulary as u64,
            format: if self.tied_head {
                self.embedding_format
            } else {
                self.head_format
            },
            bias: false,
            parameter: if self.tied_head {
                self.embedding_weight.clone()
            } else {
                self.head_weight.clone()
            },
        })
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

fn validate_vocabulary_specs(
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
        let StaticModuleConstruction {
            embedding,
            head,
            normalization,
        } = spec.construction_specs()?;
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
        let StaticModuleConstruction {
            embedding,
            head,
            normalization,
        } = spec.construction_specs()?;
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

    /// Builds the family-selected projection resources inside an ordinary body.
    fn build_feed_forward(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error>;

    /// Retains global routing cardinality while constructing owner-local banks.
    /// Ordinary dense projection policies consume only the localized configuration.
    fn build_partitioned_feed_forward(
        global: &C,
        local: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        let _ = global;
        Self::build_feed_forward(local, layer, context)
    }

    /// Declares canonical global parameter topology without native resources.
    fn parameter_description(
        config: &C,
    ) -> Result<ArchitectureParameterDescription, ParallelPlanError>;
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

    fn build_feed_forward(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        Mlp::new(config, layer, context)
    }

    fn parameter_description(
        config: &C,
    ) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
        dense_parameter_description(config)
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
    pub(crate) fn from_selected_spec(
        spec: StaticModuleSpec,
        vocabulary: Option<(VocabularyParallelRange, Option<VocabularyParallelRange>)>,
        owns: impl Fn(&str) -> bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let construction = spec.construction_specs()?;
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
                .map_or(StaticModulePlacement::Local, |(range, _)| {
                    StaticModulePlacement::VocabularyParallel(range.clone())
                })
        } else {
            StaticModulePlacement::Omitted
        };
        let norm = owns("norm");
        let head = if construction.head.is_some() && owns("output") {
            vocabulary
                .as_ref()
                .map_or(StaticModulePlacement::Local, |(_, range)| {
                    StaticModulePlacement::VocabularyParallel(
                        range.clone().expect("validated output range"),
                    )
                })
        } else {
            StaticModulePlacement::Omitted
        };
        Self::from_construction(construction, embedding, head, norm, context)
    }

    /// Materializes validated resources with independently selected vocabulary owners.
    pub(crate) fn from_construction(
        construction: StaticModuleConstruction,
        embedding: StaticModulePlacement,
        head: StaticModulePlacement,
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
        } else if !matches!(head, StaticModulePlacement::Omitted) {
            return Err(Error::backend(
                "selected output has no vocabulary projection",
            ));
        }
        let embeddings = match embedding {
            StaticModulePlacement::Omitted => None,
            StaticModulePlacement::Local => Some(B::embedding(embedding_spec, context)?),
            StaticModulePlacement::VocabularyParallel(range) => Some(
                B::vocabulary_parallel_embedding(embedding_spec, range, context)?,
            ),
        };
        let norm = owns_norm
            .then(|| B::normalization(normalization, context))
            .transpose()?;
        let lm_head = match (head, head_spec) {
            (StaticModulePlacement::Omitted, _) => None,
            (StaticModulePlacement::Local, Some(spec)) => Some(B::linear(spec, context)?),
            (StaticModulePlacement::VocabularyParallel(range), Some(spec)) => {
                Some(B::vocabulary_parallel_linear(spec, range, context)?)
            }
            (_, None) => unreachable!("validated output ownership"),
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
        global: &C,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        match self.parallel {
            Some(_) => state_layout(global).map(std::borrow::Cow::Owned),
            None => Ok(std::borrow::Cow::Borrowed(&self.state)),
        }
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
    let parameter_uses = parameter_uses(args, &schedule, &parameters)?;
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
    .and_then(|definition| definition.with_parameter_uses(parameter_uses))
    .and_then(|definition| {
        definition.map_unit_observation_boundaries(|address, _| {
            schedule.boundaries(args, address.index())
        })
    })
    .map_err(Error::from)
}

fn parameter_uses<C: Config>(
    args: &C,
    schedule: &DecoderSchedule,
    parameters: &ArchitectureParameterDescription,
) -> Result<Vec<(String, eredu_runtime::ComponentParameterUse)>, Error> {
    let mut parameter_uses = Vec::new();
    for owned in parameters.groups() {
        let ParameterGroupOwner::ExecutionUnit {
            group,
            global_unit,
            consumers,
            ..
        } = owned.owner()
        else {
            continue;
        };
        let group_index = parameters
            .graph()
            .group_index(group.as_str())
            .expect("declared group");
        let canonical = schedule.path(args, *global_unit)?;
        for &consumer in consumers {
            let prefix = schedule.path(args, consumer)?;
            let ordinal = parameters
                .unit_layout()
                .ordinal(group_index, consumer)
                .expect("declared invocation");
            let invocation = parameters
                .unit_layout()
                .address(ordinal)
                .expect("declared invocation");
            for member in owned.members() {
                let suffix = member
                    .target()
                    .strip_prefix(&canonical)
                    .filter(|suffix| suffix.starts_with('.'))
                    .ok_or_else(|| {
                        Error::backend(format!(
                            "decoder resource {} is outside its declared owner {canonical}",
                            member.target()
                        ))
                    })?;
                parameter_uses.push((
                    format!("{prefix}{suffix}"),
                    eredu_runtime::ComponentParameterUse {
                        invocation,
                        target: member.target().into(),
                    },
                ));
            }
        }
    }
    Ok(parameter_uses)
}

/// Retains semantic ownership while selecting exact family-authored physical formats.
pub fn project_parameters<C: Config>(
    args: &C,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
    parameters: ArchitectureParameterDescription,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    if definition.parameters() == &parameters {
        return Ok(definition.clone());
    }
    let uses = parameter_uses(args, &DecoderSchedule::new(args)?, &parameters)?;
    definition
        .as_ref()
        .clone()
        .with_parameters(parameters, uses)
        .map(std::sync::Arc::new)
        .map_err(Error::from)
}

/// Typed decoder equations over exact admitted pipeline geometry.
pub struct DecoderKernel<
    B: NeuralBackend,
    C: Config,
    P = DenseBlockFactory,
    M = PartitionStaticModules<B>,
    G = PartitionLocalGeometry<C>,
> {
    args: C,
    schedule: DecoderSchedule,
    static_modules: M,
    geometry: G,
    block_factory: std::marker::PhantomData<fn() -> (B, P)>,
}

impl<B: NeuralBackend, C: Config + Clone, P, M: Clone, G: Clone> Clone
    for DecoderKernel<B, C, P, M, G>
{
    fn clone(&self) -> Self {
        Self {
            args: self.args.clone(),
            schedule: self.schedule.clone(),
            static_modules: self.static_modules.clone(),
            geometry: self.geometry.clone(),
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
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry<C>, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedLayeredModel<B, C, P>, Error> {
        args.validate_config()?;
        P::validate(&args)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        let expected = args
            .partition_parameter_description()
            .map_err(Error::backend_source)?;
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
                ))
            }
        }
        let geometry = partition.local_geometry().clone();
        let definition = definition.select_state_layout(geometry.complete_state_layout())?;
        validate_complete_definition(&args, &definition, geometry.complete_state_layout())?;
        let selection = definition.select_partition(partition)?;
        let static_modules = PartitionStaticModules::from_selected_spec(
            StaticModuleSpec::from_config(&args),
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
        TransformerBlock::from_factory::<C, P>(&self.args, config, global_unit, context)
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
        match equation {
            DecoderOperation::Body { .. } => Ok(DecoderUnit::Body(
                TransformerBlockBody::from_factory::<C, P>(&self.args, config, layer, context)?,
            )),
            DecoderOperation::Terminal { .. } => Ok(DecoderUnit::Terminal(
                terminal_normalization::<B>(config, layer, context)?.ok_or_else(|| {
                    Error::backend("decoder terminal has no normalization policy")
                })?,
            )),
        }
    }
}

/// Projects physical factory geometry from a complete invocation map and a selected PP range.
pub fn partition_local_geometry<C: PartitionedConfig>(
    config: &C,
    invocations: &eredu_runtime::ExecutionUnitAddressMap,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry<C>, ParallelPlanError> {
    partition_local_geometry_with(
        config,
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
    invocations: &eredu_runtime::ExecutionUnitAddressMap,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
    mut localize: F,
) -> Result<PartitionLocalGeometry<C>, ParallelPlanError>
where
    C: PartitionedConfig,
    F: FnMut(&C, usize, &LocalModelLayout) -> Result<C, Error>,
{
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
            DecoderOperation::Body { layer } => localize(config, layer, layout)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            DecoderOperation::Terminal { .. } => config.clone(),
        };
        physical.insert(owner, local);
    }
    let mut local_key_value_heads = Vec::with_capacity(count);
    for layer in 0..config.num_hidden_layers() as usize {
        let ordinal = range.start + schedule.body_operation(layer);
        let address = logical[ordinal];
        let owner = invocations
            .resolve(ordinal, address)
            .expect("validated storage mapping")
            .storage_location()
            .address()
            .index();
        local_key_value_heads.push(physical[&owner].num_key_value_heads());
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
        cache_layout_with_key_value_heads(config, local_key_value_heads)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let root = config.parameter_root();
    let (embedding_range, output_range) = StaticModuleSpec::from_config(config).vocabulary_ranges(
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
) -> Result<C, Error> {
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    let fields = config.block_parameter_fields().validate()?;
    let tensor = |path: String| {
        layout
            .tensor(&path)
            .ok_or_else(|| Error::backend(format!("missing local decoder layout for {path}")))
    };
    let (query_width, key_width) = match config.attention_projection_layout() {
        AttentionProjectionLayout::Split => {
            let query = tensor(format!(
                "{prefix}.{}.{}.weight",
                fields.attention, fields.attention_query
            ))?;
            let key = tensor(format!(
                "{prefix}.{}.{}.weight",
                fields.attention, fields.attention_key
            ))?;
            (
                i32::try_from(query.local_shape()[0]).map_err(Error::backend)?,
                i32::try_from(key.local_shape()[0]).map_err(Error::backend)?,
            )
        }
        AttentionProjectionLayout::Fused { field } => {
            let fused = tensor(format!("{prefix}.{}.{field}.weight", fields.attention))?;
            let heads = fused
                .expanded_logical_range(config.num_key_value_heads() as usize)
                .map_err(Error::backend)?
                .len();
            let key_width = i32::try_from(heads)
                .map_err(Error::backend)?
                .checked_mul(config.head_dim())
                .ok_or_else(|| Error::backend("local fused KV width overflowed"))?;
            let query_width = key_width
                .checked_mul(config.num_attention_heads() / config.num_key_value_heads())
                .ok_or_else(|| Error::backend("local fused query width overflowed"))?;
            let width = key_width
                .checked_mul(2)
                .and_then(|width| query_width.checked_add(width))
                .ok_or_else(|| Error::backend("local fused QKV width overflowed"))?;
            if usize::try_from(width).map_err(Error::backend)? != fused.local_shape()[0] {
                return Err(Error::backend(
                    "local fused QKV shape differs from its head partition",
                ));
            }
            (query_width, key_width)
        }
    };
    let intermediate = match config.gated_projection_layout() {
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
    };
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
        let mut forward = self.begin(input, state, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }
    fn begin_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(initial.clone())
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
        let mut forward = self.begin_parallel(input, state, parallel, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
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
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B, C, P>, Error> {
        args.validate_config()?;
        P::validate(&args)?;
        crate::operator_requirements::require::<B>(
            "shared decoder equations",
            operator_requirements(&args),
        )?;
        validate_complete_definition(&args, &definition, &state_layout(&args)?)?;
        let static_modules = StaticModules::new(&args, context)?;
        let geometry = CompleteDecoderGeometry {
            parallel: None,
            state: definition.state_layout().clone(),
            units: definition.parameters().unit_layout().len(),
        };
        let kernel = Self {
            schedule: DecoderSchedule::new(&args)?,
            args,
            static_modules,
            geometry,
            block_factory: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }

    /// Materializes the retained declaration with planner-derived rank-local geometry.
    pub fn from_parallel_definition(
        args: C,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: LocalGeometry<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B, C, P>, Error>
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
        let definition = definition.select_state_layout(geometry.state_layout())?;
        validate_complete_definition(&args, &definition, geometry.state_layout())?;
        let static_modules = StaticModules::new_parallel(&args, &geometry, context)?;
        let geometry = CompleteDecoderGeometry {
            parallel: Some(std::sync::Arc::new(geometry)),
            state: definition.state_layout().clone(),
            units: definition.parameters().unit_layout().len(),
        };
        let kernel = Self {
            schedule: DecoderSchedule::new(&args)?,
            args,
            static_modules,
            geometry,
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
