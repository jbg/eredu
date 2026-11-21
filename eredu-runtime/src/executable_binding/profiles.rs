//! Statically typed state implementations and shared construction visitors.

use std::marker::PhantomData;

use crate::{LayerRuntimeState, RuntimeAppendStreams, RuntimeStateComponents};
use eredu_nn::{AttentionCache, BlockwiseAttentionBackend, CompressedAttentionCache};

use super::ReplicatedTextArchitectureVisitor;

/// Concrete state implementations for each architecture-selected access profile.
///
/// One implementation can serve several profiles when it satisfies their exact
/// access bounds. An arbitrary runtime state cannot stand in for those bounds:
///
/// ```compile_fail
/// use std::marker::PhantomData;
/// use eredu_runtime::executable_binding::ReplicatedTextStateProfiles;
/// use eredu_nn::BlockwiseAttentionBackend;
/// use eredu_runtime::LayerRuntimeState;
/// struct UncheckedProfiles<B, S>(PhantomData<(B, S)>);
/// impl<B, S> ReplicatedTextStateProfiles<B> for UncheckedProfiles<B, S>
/// where
///     B: BlockwiseAttentionBackend,
///     S: LayerRuntimeState<B>,
/// {
///     type StatelessState = S;
///     type AttentionState = S;
///     type ComponentState = S;
///     type AttentionComponentState = S;
///     type AttentionStreamState = S;
///     type CompressedState = S;
///     type CompressedComponentState = S;
/// }
/// ```
pub trait ReplicatedTextStateProfiles<B: BlockwiseAttentionBackend> {
    /// State used by an architecture without mutable token state.
    type StatelessState: LayerRuntimeState<B> + 'static;
    /// State exposing ordinary attention cache access.
    type AttentionState: LayerRuntimeState<B, LayerState: AttentionCache<B::Tensor>> + 'static;
    /// State exposing named fixed components.
    type ComponentState: LayerRuntimeState<B, LayerState: RuntimeStateComponents<B>> + 'static;
    /// State exposing ordinary attention and named fixed components.
    type AttentionComponentState: LayerRuntimeState<B, LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>>
        + 'static;
    /// State exposing ordinary attention, fixed components and named append streams.
    type AttentionStreamState: LayerRuntimeState<
            B,
            LayerState: AttentionCache<B::Tensor>
                            + RuntimeStateComponents<B>
                            + RuntimeAppendStreams<B>,
        > + 'static;
    /// State exposing compressed attention access.
    type CompressedState: LayerRuntimeState<B, LayerState: CompressedAttentionCache<B::Tensor>>
        + 'static;
    /// State exposing compressed attention and named fixed components.
    type CompressedComponentState: LayerRuntimeState<
            B,
            LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        > + 'static;
}

/// The runtime state owners matching the architecture's ordinary state profiles.
pub struct PortableReplicatedStateProfiles<C>(PhantomData<fn() -> C>);

impl<B, C> ReplicatedTextStateProfiles<B> for PortableReplicatedStateProfiles<C>
where
    B: BlockwiseAttentionBackend<Tensor = C::Value>,
    C: crate::state::HybridCacheStorage,
{
    type StatelessState = crate::state::HybridState<C>;
    type AttentionState = crate::state::KeyValueState<C>;
    type ComponentState = crate::state::HybridState<C>;
    type AttentionComponentState = crate::state::HybridState<C>;
    type AttentionStreamState = crate::state::HybridState<C>;
    type CompressedState = crate::state::HybridState<C>;
    type CompressedComponentState = crate::state::HybridState<C>;
}

/// Shares one portable construction visitor across the exact state profiles.
pub struct SharedReplicatedTextVisitor<P, V> {
    visitor: V,
    profiles: PhantomData<fn() -> P>,
}

impl<P, V> SharedReplicatedTextVisitor<P, V> {
    /// Binds a construction visitor to a concrete state-profile set.
    pub const fn new(visitor: V) -> Self {
        Self {
            visitor,
            profiles: PhantomData,
        }
    }
}

impl<B, P, V, O, E> ReplicatedTextProfileDispatcher<B> for SharedReplicatedTextVisitor<P, V>
where
    B: BlockwiseAttentionBackend
        + eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        >,
    P: ReplicatedTextStateProfiles<B>,
    V: ReplicatedTextArchitectureVisitor<B, P::StatelessState, Output = O, Error = E>
        + ReplicatedTextArchitectureVisitor<B, P::AttentionState, Output = O, Error = E>
        + ReplicatedTextArchitectureVisitor<B, P::ComponentState, Output = O, Error = E>
        + ReplicatedTextArchitectureVisitor<B, P::AttentionComponentState, Output = O, Error = E>
        + ReplicatedTextArchitectureVisitor<B, P::AttentionStreamState, Output = O, Error = E>
        + ReplicatedTextArchitectureVisitor<B, P::CompressedState, Output = O, Error = E>
        + ReplicatedTextArchitectureVisitor<B, P::CompressedComponentState, Output = O, Error = E>,
{
    type Output = O;
    type Error = E;
    type StatelessState = P::StatelessState;
    type AttentionState = P::AttentionState;
    type ComponentState = P::ComponentState;
    type AttentionComponentState = P::AttentionComponentState;
    type AttentionStreamState = P::AttentionStreamState;
    type CompressedState = P::CompressedState;
    type CompressedComponentState = P::CompressedComponentState;
    type StatelessVisitor = V;
    type AttentionVisitor = V;
    type ComponentVisitor = V;
    type AttentionComponentVisitor = V;
    type AttentionStreamVisitor = V;
    type CompressedVisitor = V;
    type CompressedComponentVisitor = V;

    fn into_stateless_visitor(self) -> Self::StatelessVisitor {
        self.visitor
    }

    fn into_attention_visitor(self) -> Self::AttentionVisitor {
        self.visitor
    }

    fn into_component_visitor(self) -> Self::ComponentVisitor {
        self.visitor
    }

    fn into_attention_stream_visitor(self) -> Self::AttentionStreamVisitor {
        self.visitor
    }

    fn into_attention_component_visitor(self) -> Self::AttentionComponentVisitor {
        self.visitor
    }

    fn into_compressed_visitor(self) -> Self::CompressedVisitor {
        self.visitor
    }

    fn into_compressed_component_visitor(self) -> Self::CompressedComponentVisitor {
        self.visitor
    }
}

/// Supplies typed construction visitors while architecture dispatch selects the exact state profile.
///
/// This dispatcher supplies a concrete state and construction visitor for
/// each portable access profile.
pub trait ReplicatedTextProfileDispatcher<B>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        >,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;
    /// State representation used when no token state is retained.
    type StatelessState: LayerRuntimeState<B> + 'static;
    /// State representation used by ordinary key/value attention.
    type AttentionState: LayerRuntimeState<B> + 'static;
    /// State representation used by recurrent or convolutional components.
    type ComponentState: LayerRuntimeState<B> + 'static;
    /// State representation used by key/value attention with fixed components.
    type AttentionComponentState: LayerRuntimeState<B> + 'static;
    /// State exposing the combined attention, fixed and named-stream contract.
    type AttentionStreamState: LayerRuntimeState<
            B,
            LayerState: AttentionCache<B::Tensor>
                            + crate::RuntimeStateComponents<B>
                            + crate::RuntimeAppendStreams<B>,
        > + 'static;
    /// Visitor preserving the concrete combined-state access bounds.
    type AttentionStreamVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::AttentionStreamState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Consumes this adapter into its attention-with-streams visitor.
    fn into_attention_stream_visitor(self) -> Self::AttentionStreamVisitor;
    /// State representation used by compressed attention.
    type CompressedState: LayerRuntimeState<B> + 'static;
    /// State representation used by compressed attention with fixed components.
    type CompressedComponentState: LayerRuntimeState<B> + 'static;

    /// Visitor for a stateless architecture.
    type StatelessVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::StatelessState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for an ordinary key/value architecture.
    type AttentionVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::AttentionState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for a fixed-component architecture.
    type ComponentVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::ComponentState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for key/value attention with fixed components.
    type AttentionComponentVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::AttentionComponentState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for compressed attention.
    type CompressedVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::CompressedState,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Visitor for compressed attention with fixed components.
    type CompressedComponentVisitor: ReplicatedTextArchitectureVisitor<
        B,
        Self::CompressedComponentState,
        Output = Self::Output,
        Error = Self::Error,
    >;

    /// Consumes the adapter into its stateless visitor.
    fn into_stateless_visitor(self) -> Self::StatelessVisitor;
    /// Consumes the adapter into its ordinary attention visitor.
    fn into_attention_visitor(self) -> Self::AttentionVisitor;
    /// Consumes the adapter into its fixed-component visitor.
    fn into_component_visitor(self) -> Self::ComponentVisitor;
    /// Consumes the adapter into its attention-with-components visitor.
    fn into_attention_component_visitor(self) -> Self::AttentionComponentVisitor;
    /// Consumes the adapter into its compressed-attention visitor.
    fn into_compressed_visitor(self) -> Self::CompressedVisitor;
    /// Consumes the adapter into its compressed-attention-with-components visitor.
    fn into_compressed_component_visitor(self) -> Self::CompressedComponentVisitor;
}
