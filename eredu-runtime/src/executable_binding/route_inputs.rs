//! Typed binding inputs supplied by the backend to portable construction.

use eredu_nn::{NeuralBackend, Tensor};
use std::marker::PhantomData;

/// Ordinary key/value construction requiring only the base neural backend.
///
/// Other state profiles and prediction extensions are rejected before any model
/// constructor or binding visitor runs. The full text route suite requires additional mechanisms.
pub struct KeyValueRoute<'a, B: NeuralBackend, S, V> {
    /// Native execution context.
    pub context: &'a <B::Tensor as Tensor>::Context,
    /// Native key/value visitor.
    pub visitor: V,
    state: PhantomData<fn() -> S>,
}

impl<'a, B: NeuralBackend, S, V> KeyValueRoute<'a, B, S, V> {
    /// Binds the native context and one ordinary attention-state visitor.
    pub fn new(context: &'a <B::Tensor as Tensor>::Context, visitor: V) -> Self {
        Self {
            context,
            visitor,
            state: PhantomData,
        }
    }
}
/// An absent optional prediction construction capability.
#[derive(Clone, Copy, Debug, Default)]
pub struct WithoutPrediction;

/// Native prediction materializer and typed target visitor factory.
///
/// Construction routes create this value through their `with_prediction` methods.
pub struct PredictionMechanisms<M, S, V> {
    #[doc(hidden)]
    pub visitor: V,
    marker: PhantomData<fn() -> (M, S)>,
}
impl<M, S, V> PredictionMechanisms<M, S, V> {
    /// Pairs the supplied materializer and visitor.
    pub fn new(visitor: V) -> Self {
        Self {
            visitor,
            marker: PhantomData,
        }
    }
}
/// Ordinary replicated construction with architecture-selected typed state access.
pub struct ReplicatedRoute<'a, B: NeuralBackend, D, P = WithoutPrediction> {
    #[doc(hidden)]
    pub context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub source_context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub dispatcher: D,
    #[doc(hidden)]
    pub prediction: P,
}

impl<'a, B: NeuralBackend, D> ReplicatedRoute<'a, B, D> {
    /// Binds native contexts and a typed profile visitor without selecting a profile.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        dispatcher: D,
    ) -> Self {
        Self {
            context,
            source_context,
            dispatcher,
            prediction: WithoutPrediction,
        }
    }
}

impl<'a, B: NeuralBackend, D, P> ReplicatedRoute<'a, B, D, P> {
    /// Adds a native materialization context and a prediction-aware target visitor.
    pub fn with_prediction<M, V>(
        self,
        visitor: V,
    ) -> ReplicatedRoute<'a, B, D, PredictionMechanisms<M, (), V>> {
        ReplicatedRoute {
            context: self.context,
            source_context: self.source_context,
            dispatcher: self.dispatcher,
            prediction: PredictionMechanisms::new(visitor),
        }
    }
}

/// Routed text construction with architecture-owned equation and state dispatch.
pub struct RoutedRoute<'a, B: NeuralBackend, S, PS, SS, G, R, T, U, P = WithoutPrediction> {
    #[doc(hidden)]
    pub context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub source_context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub gated: G,
    #[doc(hidden)]
    pub relu2: R,
    #[doc(hidden)]
    pub pooling: T,
    #[doc(hidden)]
    pub streams: U,
    #[doc(hidden)]
    pub prediction: P,
    states: PhantomData<fn() -> (S, PS, SS)>,
}

impl<'a, B: NeuralBackend, S, PS, SS, G, R, T, U> RoutedRoute<'a, B, S, PS, SS, G, R, T, U> {
    /// Binds exact gated, ReLU-squared, pooling and append-stream visitors.
    /// Architecture code chooses the selected state profile.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        gated: G,
        relu2: R,
        pooling: T,
        streams: U,
    ) -> Self {
        Self {
            context,
            source_context,
            gated,
            relu2,
            pooling,
            streams,
            prediction: WithoutPrediction,
            states: PhantomData,
        }
    }
}

impl<'a, B: NeuralBackend, S, PS, SS, G, R, T, U, P> RoutedRoute<'a, B, S, PS, SS, G, R, T, U, P> {
    /// Adds native extension materialization and a routed prediction-profile visitor.
    pub fn with_prediction<M, V>(
        self,
        visitor: V,
    ) -> RoutedRoute<'a, B, S, PS, SS, G, R, T, U, PredictionMechanisms<M, (S, PS), V>> {
        RoutedRoute {
            context: self.context,
            source_context: self.source_context,
            gated: self.gated,
            relu2: self.relu2,
            pooling: self.pooling,
            streams: self.streams,
            prediction: PredictionMechanisms::new(visitor),
            states: PhantomData,
        }
    }
}

/// Composite text construction with one selected processor and ingress contract.
pub struct CompositeRoute<'a, B: NeuralBackend, S, V, P = WithoutPrediction> {
    #[doc(hidden)]
    pub context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub source_context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub visitor: V,
    #[doc(hidden)]
    pub prediction: P,
    state: PhantomData<fn() -> S>,
}
impl<'a, B: NeuralBackend, S, V> CompositeRoute<'a, B, S, V> {
    /// Binds one native visitor to architecture-owned composite construction.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        visitor: V,
    ) -> Self {
        Self {
            context,
            source_context,
            visitor,
            prediction: WithoutPrediction,
            state: PhantomData,
        }
    }
}
impl<'a, B: NeuralBackend, S, V, P> CompositeRoute<'a, B, S, V, P> {
    /// Adds native extension materialization and a composite prediction visitor.
    pub fn with_prediction<M, V2>(
        self,
        visitor: V2,
    ) -> CompositeRoute<'a, B, S, V, PredictionMechanisms<M, S, V2>> {
        CompositeRoute {
            context: self.context,
            source_context: self.source_context,
            visitor: self.visitor,
            prediction: PredictionMechanisms::new(visitor),
            state: PhantomData,
        }
    }
}

/// Direct partitioned construction from an exact native visitor factory.
pub struct PartitionedDenseRoute<'a, B: NeuralBackend, S, V, P = WithoutPrediction> {
    #[doc(hidden)]
    pub context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub source_context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub visitor: V,
    #[doc(hidden)]
    pub prediction: P,
    state: PhantomData<fn() -> S>,
}
impl<'a, B: NeuralBackend, S, V> PartitionedDenseRoute<'a, B, S, V> {
    /// Binds a factory receiving only checked native communication and source roles.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        visitor: V,
    ) -> Self {
        Self {
            context,
            source_context,
            visitor,
            prediction: WithoutPrediction,
            state: PhantomData,
        }
    }
}
impl<'a, B: NeuralBackend, S, V, P> PartitionedDenseRoute<'a, B, S, V, P> {
    /// Adds native extension materialization and a partitioned prediction visitor factory.
    pub fn with_prediction<M, V2>(
        self,
        visitor: V2,
    ) -> PartitionedDenseRoute<'a, B, S, V, PredictionMechanisms<M, S, V2>> {
        PartitionedDenseRoute {
            context: self.context,
            source_context: self.source_context,
            visitor: self.visitor,
            prediction: PredictionMechanisms::new(visitor),
            state: PhantomData,
        }
    }
}

/// Routed partitioned construction with architecture-owned operator/state selection.
pub struct PartitionedRoutedRoute<'a, B: NeuralBackend, S, PS, SS, G, T, U, P = WithoutPrediction> {
    #[doc(hidden)]
    pub context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub source_context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub gated: G,
    #[doc(hidden)]
    pub pooling: T,
    #[doc(hidden)]
    pub streams: U,
    #[doc(hidden)]
    pub prediction: P,
    states: PhantomData<fn() -> (S, PS, SS)>,
}
impl<'a, B: NeuralBackend, S, PS, SS, G, T, U> PartitionedRoutedRoute<'a, B, S, PS, SS, G, T, U> {
    /// Binds factories for mixed-state grouped execution and pooling execution.
    ///
    /// The first visitor implements both gated-product and ReLU-squared contracts.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        gated: G,
        pooling: T,
        streams: U,
    ) -> Self {
        Self {
            context,
            source_context,
            gated,
            pooling,
            streams,
            prediction: WithoutPrediction,
            states: PhantomData,
        }
    }
}

impl<'a, B: NeuralBackend, S, PS, SS, G, T, U, P>
    PartitionedRoutedRoute<'a, B, S, PS, SS, G, T, U, P>
{
    /// Adds native extension materialization and a typed prediction visitor factory.
    pub fn with_prediction<M, V>(
        self,
        visitor: V,
    ) -> PartitionedRoutedRoute<'a, B, S, PS, SS, G, T, U, PredictionMechanisms<M, (S, PS), V>>
    {
        PartitionedRoutedRoute {
            context: self.context,
            source_context: self.source_context,
            gated: self.gated,
            pooling: self.pooling,
            streams: self.streams,
            prediction: PredictionMechanisms::new(visitor),
            states: PhantomData,
        }
    }
}

/// Composite partitioned construction from checked native resources.
pub struct PartitionedCompositeRoute<'a, B: NeuralBackend, S, V, P = WithoutPrediction> {
    #[doc(hidden)]
    pub context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub source_context: &'a <B::Tensor as Tensor>::Context,
    #[doc(hidden)]
    pub visitor: V,
    #[doc(hidden)]
    pub prediction: P,
    state: PhantomData<fn() -> S>,
}
impl<'a, B: NeuralBackend, S, V> PartitionedCompositeRoute<'a, B, S, V> {
    /// Binds a visitor factory receiving only checked communication and sources.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        visitor: V,
    ) -> Self {
        Self {
            context,
            source_context,
            visitor,
            prediction: WithoutPrediction,
            state: PhantomData,
        }
    }
}
impl<'a, B: NeuralBackend, S, V, P> PartitionedCompositeRoute<'a, B, S, V, P> {
    /// Adds native extension materialization and composite prediction binding.
    pub fn with_prediction<M, V2>(
        self,
        visitor: V2,
    ) -> PartitionedCompositeRoute<'a, B, S, V, PredictionMechanisms<M, S, V2>> {
        PartitionedCompositeRoute {
            context: self.context,
            source_context: self.source_context,
            visitor: self.visitor,
            prediction: PredictionMechanisms::new(visitor),
            state: PhantomData,
        }
    }
}
