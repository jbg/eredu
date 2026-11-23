//! Shared assembly of typed model routes from native binding operations.

use super::{
    composite_partition_binding::{CompositePartitionPredictionVisitor, CompositePartitionVisitor},
    dense_partition::{DensePartitionPredictionVisitor, DensePartitionVisitor},
    routed_partition_binding::{RoutedPartitionPredictionVisitor, RoutedPartitionVisitor},
    CompositeRoute, PartitionedCompositeRoute, PartitionedDenseRoute, PartitionedRoutedRoute,
    PredictionBinding, PredictionMechanisms, PreparedExecutionRoutes,
    PreparedPartitionPredictionResources, PreparedPartitionResources, ReplicatedRoute, RoutedRoute,
    SharedReplicatedTextVisitor, Unavailable,
};
use eredu_core::checkpoint_mechanisms::{
    LocalTextPlacement, RankedTextPlacement, TextBindingContext, TextBindingResources,
};
use eredu_nn::{BlockwiseAttentionBackend, NeuralBackend, Tensor};
use std::collections::BTreeSet;

mod local;

/// Native resources for a retained rank-local composite.
pub struct RetainedCompositeRank<'a, B: NeuralBackend, S, V> {
    /// Execution context.
    pub context: &'a Context<B>,
    /// Native binding resources.
    pub visitor: V,
    pub(crate) state: std::marker::PhantomData<fn() -> S>,
}
mod prediction;
pub(super) mod weights;
use prediction::LocalPrediction;

type Factory<'a, I, V> = Box<dyn FnOnce(I) -> V + 'a>;
type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

type RankFactory<'a, C, V> = Factory<'a, PreparedPartitionResources<C>, V>;
type RankPrediction<'a, C, M, S, V> =
    PredictionMechanisms<M, S, Factory<'a, PreparedPartitionPredictionResources<C>, V>>;

type DensePartition<'a, M> = PartitionedDenseRoute<
    'a,
    Neural<M>,
    State<M>,
    RankFactory<'a, Communication<M>, DensePartitionVisitor<Local<'a, M>, Communication<M>>>,
    RankPrediction<
        'a,
        Communication<M>,
        Materializer<M>,
        State<M>,
        DensePartitionPredictionVisitor<Local<'a, M>, Communication<M>>,
    >,
>;
type RoutedPartition<'a, M> = PartitionedRoutedRoute<
    'a,
    Neural<M>,
    State<M>,
    Pooling<M>,
    State<M>,
    RankFactory<'a, Communication<M>, RoutedPartitionVisitor<Local<'a, M>, Communication<M>>>,
    RankFactory<'a, Communication<M>, RoutedPartitionVisitor<Local<'a, M>, Communication<M>>>,
    RankFactory<'a, Communication<M>, RoutedPartitionVisitor<Local<'a, M>, Communication<M>, true>>,
    RankPrediction<
        'a,
        Communication<M>,
        Materializer<M>,
        (State<M>, Pooling<M>),
        RoutedPartitionPredictionVisitor<Local<'a, M>, Communication<M>>,
    >,
>;
type CompositePartition<'a, M> = PartitionedCompositeRoute<
    'a,
    Neural<M>,
    State<M>,
    RankFactory<'a, Communication<M>, CompositePartitionVisitor<Local<'a, M>, Communication<M>>>,
    RankPrediction<
        'a,
        Communication<M>,
        Materializer<M>,
        State<M>,
        CompositePartitionPredictionVisitor<Local<'a, M>, Communication<M>>,
    >,
>;

/// Portable rank routes available for an exact native placement.
pub trait TextRankRoutes<'a, M: TextBindingContext<'a>>: private::RankPlacement {
    /// Dense partition route.
    type Dense;
    /// Routed partition route.
    type Routed;
    /// Composite partition route.
    type Composite;
    /// Retained composite rank route.
    type RetainedComposite;

    /// Assembles portable rank routes from an already selected native context.
    fn bind(
        execution: &'a Context<M::Neural>,
        weights: &'a Context<M::Neural>,
        local: M::TextResources,
    ) -> (
        Self::Dense,
        Self::Routed,
        Self::Composite,
        Self::RetainedComposite,
    );
}

mod private {
    pub trait RankPlacement {}
    impl RankPlacement for super::LocalTextPlacement {}
    impl RankPlacement for super::RankedTextPlacement {}
}

impl<'a, M: TextBindingContext<'a>> TextRankRoutes<'a, M> for LocalTextPlacement {
    type Dense = Unavailable;
    type Routed = Unavailable;
    type Composite = Unavailable;
    type RetainedComposite = Unavailable;

    fn bind(
        _: &'a Context<M::Neural>,
        _: &'a Context<M::Neural>,
        _: M::TextResources,
    ) -> (
        Self::Dense,
        Self::Routed,
        Self::Composite,
        Self::RetainedComposite,
    ) {
        (Unavailable, Unavailable, Unavailable, Unavailable)
    }
}

impl<'a, M: TextBindingContext<'a>> TextRankRoutes<'a, M> for RankedTextPlacement
where
    M::Neural: BlockwiseAttentionBackend,
    M::CacheStorage: crate::state::HybridCacheStorage<Value = <M::Neural as NeuralBackend>::Tensor>
        + eredu_core::cache::mechanisms::PoolingCacheStorage,
    M::Communication: 'a,
{
    type Dense = DensePartition<'a, M>;
    type Routed = RoutedPartition<'a, M>;
    type Composite = CompositePartition<'a, M>;
    type RetainedComposite = RetainedCompositeRank<'a, Neural<M>, State<M>, Local<'a, M>>;

    fn bind(
        execution: &'a Context<M::Neural>,
        weights: &'a Context<M::Neural>,
        local: M::TextResources,
    ) -> (
        Self::Dense,
        Self::Routed,
        Self::Composite,
        Self::RetainedComposite,
    ) {
        let (dense, routed, composite) = bind_partitions::<M>(execution, weights, local.clone());
        let retained = RetainedCompositeRank {
            context: execution,
            visitor: local,
            state: std::marker::PhantomData,
        };
        (dense, routed, composite, retained)
    }
}

type Rank<M> = <M as TextBindingResources>::RankPlacement;

fn bind_partitions<'a, M: TextBindingContext<'a>>(
    execution: &'a Context<M::Neural>,
    weights: &'a Context<M::Neural>,
    native: Local<'a, M>,
) -> (
    DensePartition<'a, M>,
    RoutedPartition<'a, M>,
    CompositePartition<'a, M>,
)
where
    M::Neural: BlockwiseAttentionBackend,
    M::CacheStorage: crate::state::HybridCacheStorage<Value = <M::Neural as NeuralBackend>::Tensor>
        + eredu_core::cache::mechanisms::PoolingCacheStorage,
    M::Communication: 'a,
{
    let mechanisms = native.clone();
    let dense_ordinary: RankFactory<'a, M::Communication, _> = Box::new(move |resources| {
        let (communication, _, extensions) = resources.into_parts();
        DensePartitionVisitor::new(mechanisms, communication, extensions)
    });
    let mechanisms = native.clone();
    let dense_prediction: Factory<'a, PreparedPartitionPredictionResources<M::Communication>, _> =
        Box::new(move |facts| {
            let (resources, facts) = facts.into_parts();
            let (communication, _, extensions) = resources.into_parts();
            DensePartitionPredictionVisitor::new(mechanisms, communication, extensions, facts)
        });
    let dense = PartitionedDenseRoute::new(execution, weights, dense_ordinary)
        .with_prediction(dense_prediction);

    let mechanisms = native.clone();
    let routed_ordinary: RankFactory<'a, M::Communication, _> = Box::new(move |resources| {
        let (communication, _, extensions) = resources.into_parts();
        RoutedPartitionVisitor::new(mechanisms, communication, extensions)
    });
    let mechanisms = native.clone();
    let pooling: RankFactory<'a, M::Communication, _> = Box::new(move |resources| {
        let (communication, _, _) = resources.into_parts();
        RoutedPartitionVisitor::new(mechanisms, communication, BTreeSet::new())
    });
    let mechanisms = native.clone();
    let streams: RankFactory<'a, M::Communication, _> = Box::new(move |resources| {
        let (communication, _, extensions) = resources.into_parts();
        RoutedPartitionVisitor::new(mechanisms, communication, extensions)
    });
    let mechanisms = native.clone();
    let routed_prediction: Factory<'a, PreparedPartitionPredictionResources<M::Communication>, _> =
        Box::new(move |facts| {
            let (resources, prediction) = facts.into_parts();
            let (communication, _, extensions) = resources.into_parts();
            RoutedPartitionPredictionVisitor::new(mechanisms, communication, extensions, prediction)
        });
    let routed = PartitionedRoutedRoute::new(execution, weights, routed_ordinary, pooling, streams)
        .with_prediction(routed_prediction);

    let mechanisms = native.clone();
    let composite_ordinary: RankFactory<'a, M::Communication, _> = Box::new(move |resources| {
        let (communication, target, _) = resources.into_parts();
        CompositePartitionVisitor::new(mechanisms, communication, target)
    });
    let composite_prediction: Factory<
        'a,
        PreparedPartitionPredictionResources<M::Communication>,
        _,
    > = Box::new(move |facts| {
        let (resources, prediction) = facts.into_parts();
        let (communication, target, _) = resources.into_parts();
        CompositePartitionPredictionVisitor::new(native, communication, target, prediction)
    });
    let composite = PartitionedCompositeRoute::new(execution, weights, composite_ordinary)
        .with_prediction(composite_prediction);

    (dense, routed, composite)
}

type Neural<M> = <M as TextBindingResources>::Neural;
type Communication<M> =
    <M as eredu_core::checkpoint_mechanisms::ModelBindingResources>::Communication;
type Local<'a, M> = <M as TextBindingContext<'a>>::TextResources;
type Profiles<M> =
    super::profiles::PortableReplicatedStateProfiles<<M as TextBindingResources>::CacheStorage>;
type State<M> = crate::state::HybridState<<M as TextBindingResources>::CacheStorage>;
type Pooling<M> = crate::state::PoolingAttentionState<
    <M as TextBindingResources>::Neural,
    <M as TextBindingResources>::CacheStorage,
>;
type Materializer<M> = <M as TextBindingResources>::PredictionMaterializer;
type PredictionVisitor<'a, M> = LocalPrediction<Local<'a, M>, State<M>, Pooling<M>>;
type OrdinaryPrediction<'a, M, S> = PredictionMechanisms<
    Materializer<M>,
    S,
    Factory<'a, PredictionBinding, PredictionVisitor<'a, M>>,
>;

/// Complete typed route bundle derived from a backend's native binding operations.
pub type BoundExecutionRoutes<'a, M> = PreparedExecutionRoutes<
    ReplicatedRoute<
        'a,
        <M as TextBindingResources>::Neural,
        SharedReplicatedTextVisitor<Profiles<M>, Local<'a, M>>,
        OrdinaryPrediction<'a, M, ()>,
    >,
    RoutedRoute<
        'a,
        <M as TextBindingResources>::Neural,
        State<M>,
        Pooling<M>,
        State<M>,
        Local<'a, M>,
        Local<'a, M>,
        Local<'a, M>,
        Local<'a, M>,
        OrdinaryPrediction<'a, M, (State<M>, Pooling<M>)>,
    >,
    CompositeRoute<
        'a,
        <M as TextBindingResources>::Neural,
        State<M>,
        Local<'a, M>,
        OrdinaryPrediction<'a, M, State<M>>,
    >,
    <Rank<M> as TextRankRoutes<'a, M>>::Dense,
    <Rank<M> as TextRankRoutes<'a, M>>::Routed,
    <Rank<M> as TextRankRoutes<'a, M>>::Composite,
    <Rank<M> as TextRankRoutes<'a, M>>::RetainedComposite,
>;

pub(super) fn bind_text_routes<'a, M>(native: &'a M) -> BoundExecutionRoutes<'a, M>
where
    M: TextBindingContext<'a, RankPlacement: TextRankRoutes<'a, M>>,
    M::Neural: BlockwiseAttentionBackend,
    M::CacheStorage: crate::state::HybridCacheStorage<Value = <M::Neural as NeuralBackend>::Tensor>
        + eredu_core::cache::mechanisms::PoolingCacheStorage,
{
    let execution = native.text_execution();
    let weights = native.text_weights();
    let local = native.text_resources();
    let prediction = || -> Factory<'a, PredictionBinding, PredictionVisitor<'a, M>> {
        let resources = local.clone();
        Box::new(move |facts| LocalPrediction::new(resources, facts))
    };
    let ordinary = (
        ReplicatedRoute::new(
            execution,
            weights,
            SharedReplicatedTextVisitor::new(local.clone()),
        )
        .with_prediction(prediction()),
        RoutedRoute::new(
            execution,
            weights,
            local.clone(),
            local.clone(),
            local.clone(),
            local.clone(),
        )
        .with_prediction(prediction()),
        CompositeRoute::new(execution, weights, local.clone()).with_prediction(prediction()),
    );
    let partitioned = M::RankPlacement::bind(execution, weights, local);
    PreparedExecutionRoutes::from_parts(ordinary, partitioned)
}
