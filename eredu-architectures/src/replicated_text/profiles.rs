//! Statically typed state implementations and shared construction visitors.

use std::marker::PhantomData;

use eredu_nn::{AttentionCache, BlockwiseAttentionBackend, CompressedAttentionCache};
use eredu_runtime::{LayerRuntimeState, RuntimeStateComponents};

use super::{ReplicatedTextArchitectureVisitor, ReplicatedTextProfileDispatcher};

/// Concrete state implementations for each architecture-selected access profile.
///
/// One implementation can serve several profiles when it satisfies their exact
/// access bounds. An arbitrary runtime state cannot stand in for those bounds:
///
/// ```compile_fail
/// use std::marker::PhantomData;
/// use eredu_architectures::replicated_text::ReplicatedTextStateProfiles;
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
///     type CompressedState = S;
///     type CompressedComponentState = S;
/// }
/// ```
pub trait ReplicatedTextStateProfiles<B: BlockwiseAttentionBackend> {
    /// State used by an architecture without mutable token state.
    type StatelessState: LayerRuntimeState<B>;
    /// State exposing ordinary attention cache access.
    type AttentionState: LayerRuntimeState<B, LayerState: AttentionCache<B::Tensor>>;
    /// State exposing named fixed components.
    type ComponentState: LayerRuntimeState<B, LayerState: RuntimeStateComponents<B>>;
    /// State exposing ordinary attention and named fixed components.
    type AttentionComponentState: LayerRuntimeState<
        B,
        LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    >;
    /// State exposing compressed attention access.
    type CompressedState: LayerRuntimeState<B, LayerState: CompressedAttentionCache<B::Tensor>>;
    /// State exposing compressed attention and named fixed components.
    type CompressedComponentState: LayerRuntimeState<
        B,
        LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    >;
}

/// Converts one construction visitor while retaining its concrete type.
pub trait ReplicatedTextVisitorConversion<V> {
    /// Converted construction visitor.
    type Visitor;

    /// Consumes the conversion and original visitor exactly once.
    fn convert(self, visitor: V) -> Self::Visitor;
}

/// Default conversion that keeps the original visitor unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityReplicatedTextVisitor;

impl<V> ReplicatedTextVisitorConversion<V> for IdentityReplicatedTextVisitor {
    type Visitor = V;

    fn convert(self, visitor: V) -> V {
        visitor
    }
}

impl<V, F, W> ReplicatedTextVisitorConversion<V> for F
where
    F: FnOnce(V) -> W,
{
    type Visitor = W;

    fn convert(self, visitor: V) -> W {
        self(visitor)
    }
}

/// Reuses one construction visitor for every profile in a concrete state set.
///
/// Architecture dispatch still selects the exact profile and calls the
/// corresponding statically typed visitor. This adapter consumes the original
/// visitor once; it does not clone it, erase state, or alter construction-start
/// callbacks. A backend can replace only the conversions that need different
/// behavior with the `with_*_conversion` methods; all others remain identity
/// conversions. Overrides still have to satisfy their exact typed profile.
pub struct SharedReplicatedTextVisitor<
    P,
    V,
    S = IdentityReplicatedTextVisitor,
    A = IdentityReplicatedTextVisitor,
    F = IdentityReplicatedTextVisitor,
    AF = IdentityReplicatedTextVisitor,
    C = IdentityReplicatedTextVisitor,
    CF = IdentityReplicatedTextVisitor,
> {
    visitor: V,
    profiles: PhantomData<fn() -> P>,
    stateless: S,
    attention: A,
    component: F,
    attention_component: AF,
    compressed: C,
    compressed_component: CF,
}

impl<P, V> SharedReplicatedTextVisitor<P, V> {
    /// Binds a construction visitor to a concrete state-profile set.
    pub const fn new(visitor: V) -> Self {
        Self {
            visitor,
            profiles: PhantomData,
            stateless: IdentityReplicatedTextVisitor,
            attention: IdentityReplicatedTextVisitor,
            component: IdentityReplicatedTextVisitor,
            attention_component: IdentityReplicatedTextVisitor,
            compressed: IdentityReplicatedTextVisitor,
            compressed_component: IdentityReplicatedTextVisitor,
        }
    }
}

impl<P, V, S, A, F, AF, C, CF> SharedReplicatedTextVisitor<P, V, S, A, F, AF, C, CF> {
    /// Replaces only the stateless visitor conversion.
    pub fn with_stateless_conversion<N>(
        self,
        conversion: N,
    ) -> SharedReplicatedTextVisitor<P, V, N, A, F, AF, C, CF> {
        SharedReplicatedTextVisitor {
            visitor: self.visitor,
            profiles: self.profiles,
            stateless: conversion,
            attention: self.attention,
            component: self.component,
            attention_component: self.attention_component,
            compressed: self.compressed,
            compressed_component: self.compressed_component,
        }
    }

    /// Replaces only the ordinary attention visitor conversion.
    pub fn with_attention_conversion<N>(
        self,
        conversion: N,
    ) -> SharedReplicatedTextVisitor<P, V, S, N, F, AF, C, CF> {
        SharedReplicatedTextVisitor {
            visitor: self.visitor,
            profiles: self.profiles,
            stateless: self.stateless,
            attention: conversion,
            component: self.component,
            attention_component: self.attention_component,
            compressed: self.compressed,
            compressed_component: self.compressed_component,
        }
    }

    /// Replaces only the fixed-component visitor conversion.
    pub fn with_component_conversion<N>(
        self,
        conversion: N,
    ) -> SharedReplicatedTextVisitor<P, V, S, A, N, AF, C, CF> {
        SharedReplicatedTextVisitor {
            visitor: self.visitor,
            profiles: self.profiles,
            stateless: self.stateless,
            attention: self.attention,
            component: conversion,
            attention_component: self.attention_component,
            compressed: self.compressed,
            compressed_component: self.compressed_component,
        }
    }

    /// Replaces only the attention-with-components visitor conversion.
    pub fn with_attention_component_conversion<N>(
        self,
        conversion: N,
    ) -> SharedReplicatedTextVisitor<P, V, S, A, F, N, C, CF> {
        SharedReplicatedTextVisitor {
            visitor: self.visitor,
            profiles: self.profiles,
            stateless: self.stateless,
            attention: self.attention,
            component: self.component,
            attention_component: conversion,
            compressed: self.compressed,
            compressed_component: self.compressed_component,
        }
    }

    /// Replaces only the compressed-attention visitor conversion.
    pub fn with_compressed_conversion<N>(
        self,
        conversion: N,
    ) -> SharedReplicatedTextVisitor<P, V, S, A, F, AF, N, CF> {
        SharedReplicatedTextVisitor {
            visitor: self.visitor,
            profiles: self.profiles,
            stateless: self.stateless,
            attention: self.attention,
            component: self.component,
            attention_component: self.attention_component,
            compressed: conversion,
            compressed_component: self.compressed_component,
        }
    }

    /// Replaces only the compressed-attention-with-components visitor conversion.
    pub fn with_compressed_component_conversion<N>(
        self,
        conversion: N,
    ) -> SharedReplicatedTextVisitor<P, V, S, A, F, AF, C, N> {
        SharedReplicatedTextVisitor {
            visitor: self.visitor,
            profiles: self.profiles,
            stateless: self.stateless,
            attention: self.attention,
            component: self.component,
            attention_component: self.attention_component,
            compressed: self.compressed,
            compressed_component: conversion,
        }
    }
}

impl<B, P, V, O, E, S, A, F, AF, C, CF> ReplicatedTextProfileDispatcher<B>
    for SharedReplicatedTextVisitor<P, V, S, A, F, AF, C, CF>
where
    B: BlockwiseAttentionBackend,
    P: ReplicatedTextStateProfiles<B>,
    S: ReplicatedTextVisitorConversion<V>,
    A: ReplicatedTextVisitorConversion<V>,
    F: ReplicatedTextVisitorConversion<V>,
    AF: ReplicatedTextVisitorConversion<V>,
    C: ReplicatedTextVisitorConversion<V>,
    CF: ReplicatedTextVisitorConversion<V>,
    S::Visitor: ReplicatedTextArchitectureVisitor<B, P::StatelessState, Output = O, Error = E>,
    A::Visitor: ReplicatedTextArchitectureVisitor<B, P::AttentionState, Output = O, Error = E>,
    F::Visitor: ReplicatedTextArchitectureVisitor<B, P::ComponentState, Output = O, Error = E>,
    AF::Visitor:
        ReplicatedTextArchitectureVisitor<B, P::AttentionComponentState, Output = O, Error = E>,
    C::Visitor: ReplicatedTextArchitectureVisitor<B, P::CompressedState, Output = O, Error = E>,
    CF::Visitor:
        ReplicatedTextArchitectureVisitor<B, P::CompressedComponentState, Output = O, Error = E>,
{
    type Output = O;
    type Error = E;
    type StatelessState = P::StatelessState;
    type AttentionState = P::AttentionState;
    type ComponentState = P::ComponentState;
    type AttentionComponentState = P::AttentionComponentState;
    type CompressedState = P::CompressedState;
    type CompressedComponentState = P::CompressedComponentState;
    type StatelessVisitor = S::Visitor;
    type AttentionVisitor = A::Visitor;
    type ComponentVisitor = F::Visitor;
    type AttentionComponentVisitor = AF::Visitor;
    type CompressedVisitor = C::Visitor;
    type CompressedComponentVisitor = CF::Visitor;

    fn into_stateless_visitor(self) -> Self::StatelessVisitor {
        self.stateless.convert(self.visitor)
    }

    fn into_attention_visitor(self) -> Self::AttentionVisitor {
        self.attention.convert(self.visitor)
    }

    fn into_component_visitor(self) -> Self::ComponentVisitor {
        self.component.convert(self.visitor)
    }

    fn into_attention_component_visitor(self) -> Self::AttentionComponentVisitor {
        self.attention_component.convert(self.visitor)
    }

    fn into_compressed_visitor(self) -> Self::CompressedVisitor {
        self.compressed.convert(self.visitor)
    }

    fn into_compressed_component_visitor(self) -> Self::CompressedComponentVisitor {
        self.compressed_component.convert(self.visitor)
    }
}
