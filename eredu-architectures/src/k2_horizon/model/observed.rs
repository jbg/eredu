//! Actual routed and shared FFN values inside the shared decoder observer.
use super::*;
use crate::decoder::ComponentInstrumentation;
use eredu_nn::GroupSelection;

fn completed<B: GroupedNeuralBackend>(
    point: &eredu_runtime::expert::RoutedObservationPoint,
    routes: &GroupSelection<B::Tensor>,
    routed: B::Tensor,
    shared: Option<B::Tensor>,
    shape: &[i32],
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let routed = routed.reshape(shape, context)?;
    let shared = shared
        .map(|value| value.reshape(shape, context))
        .transpose()?;
    if instrumentation.observer().is_none() {
        return match shared {
            Some(shared) => routed.add(&shared, context),
            None => Ok(routed),
        };
    }
    let combined = match &shared {
        Some(shared) => routed.add(shared, context)?,
        None => routed.clone(),
    };
    let combined = if let Some(observer) = instrumentation.observer() {
        observer.observe_routing(eredu_runtime::RoutingObservation {
            path: point.path(),
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: shared.as_ref(),
            combined_output: shared.as_ref().map(|_| &combined),
            expert_count: point.expert_count(),
        })?;
        eredu_runtime::observe_and_intervene(
            observer,
            &format!("{}.output", point.path()),
            &combined,
        )?
    } else {
        combined
    };
    Ok(combined)
}

impl<B: GroupedNeuralBackend> Projections<B> {
    pub(super) fn values_instrumented<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let Some(values) = &mut self.values else {
            return Ok(None);
        };
        let point = self
            .points
            .as_ref()
            .and_then(|p| p.bank(ExpertBank::AttentionValue.id()))
            .expect("constructed value routing point");
        let mut shape = input.shape().to_vec();
        let hidden = *shape
            .last()
            .ok_or_else(|| Error::backend("value input is scalar"))?;
        let flat = input.reshape(&[-1, hidden], context)?;
        let routes = match instrumentation.observer() {
            Some(observer) => eredu_runtime::select_routes_with_observer(
                &mut values.router,
                &flat,
                context,
                point.path(),
                observer,
            )?,
            None => eredu_runtime::select_routes_with_provider::<B, P>(
                &mut values.router,
                &flat,
                context,
                provider,
                ExpertBank::AttentionValue.id(),
            )?,
        };
        let output = provider
            .forward_linear_routed(
                &mut values.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    layer: self.layer,
                    bank: ExpertBank::AttentionValue.id(),
                    input: &flat,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(Error::backend_retained_source)?;
        *shape.last_mut().expect("validated value shape") = values.output_width;
        completed::<B>(
            point,
            &routes,
            output,
            None,
            &shape,
            context,
            instrumentation,
        )
        .map(Some)
    }

    pub(super) fn feed_forward_instrumented<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let FeedForward::Routed {
            router,
            experts,
            shared,
        } = &mut self.feed_forward
        else {
            let FeedForward::Dense(mlp) = &mut self.feed_forward else {
                unreachable!()
            };
            return mlp.forward_feed_forward_observed(input, context, instrumentation);
        };
        let point = self
            .points
            .as_ref()
            .and_then(|p| p.bank(ExpertBank::FeedForward.id()))
            .expect("constructed feed-forward routing point");
        let shape = input.shape();
        let flat = input.reshape(
            &[
                -1,
                *shape
                    .last()
                    .ok_or_else(|| Error::backend("feed-forward input is scalar"))?,
            ],
            context,
        )?;
        let routes = match instrumentation.observer() {
            Some(observer) => eredu_runtime::select_routes_with_observer(
                router,
                &flat,
                context,
                point.path(),
                observer,
            )?,
            None => eredu_runtime::select_routes_with_provider::<B, P>(
                router,
                &flat,
                context,
                provider,
                ExpertBank::FeedForward.id(),
            )?,
        };
        let request = RoutedExpertRequest {
            unit_observer: None,
            layer: self.layer,
            bank: ExpertBank::FeedForward.id(),
            input: &flat,
            routes: &routes,
            pass,
        };
        let routed = match instrumentation.observer() {
            Some(observer) => eredu_runtime::with_routed_unit_observer(
                observer,
                point.path(),
                request,
                |request| provider.forward_grouped(experts, request, context),
            )
            .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?,
            None => provider
                .forward_grouped(experts, request, context)
                .map_err(Error::backend_retained_source)?,
        };
        let shared = shared
            .as_mut()
            .map(|shared| {
                if instrumentation.observer().is_none() {
                    return shared.forward_feed_forward(input, context);
                }
                instrumentation.with_scope("shared", |instrumentation| {
                    let input = instrumentation.apply("feed_forward.input", input.clone())?;
                    let value =
                        shared.forward_feed_forward_observed(&input, context, instrumentation)?;
                    let value = instrumentation.apply("feed_forward.write", value)?;
                    instrumentation.apply("feed_forward.output", value)
                })
            })
            .transpose()?;
        completed::<B>(
            point,
            &routes,
            routed,
            shared,
            shape,
            context,
            instrumentation,
        )
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    Projections<B>
{
    pub(super) fn feed_forward_parallel_instrumented<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let FeedForward::Routed {
            router,
            experts,
            shared,
        } = &mut self.feed_forward
        else {
            let FeedForward::Dense(mlp) = &mut self.feed_forward else {
                unreachable!()
            };
            return mlp.forward_feed_forward_parallel_observed(
                input,
                parallel,
                context,
                instrumentation,
            );
        };
        let point = self
            .points
            .as_ref()
            .and_then(|p| p.bank(ExpertBank::FeedForward.id()))
            .expect("constructed feed-forward routing point");
        let shape = input.shape();
        let flat = input.reshape(
            &[
                -1,
                *shape
                    .last()
                    .ok_or_else(|| Error::backend("feed-forward input is scalar"))?,
            ],
            context,
        )?;
        let routes = match instrumentation.observer() {
            Some(observer) => eredu_runtime::select_routes_with_observer(
                router,
                &flat,
                context,
                point.path(),
                observer,
            )?,
            None => eredu_runtime::select_routes_with_provider::<B, P>(
                router,
                &flat,
                context,
                provider,
                ExpertBank::FeedForward.id(),
            )?,
        };
        let request = RoutedExpertRequest {
            unit_observer: None,
            layer: self.layer,
            bank: ExpertBank::FeedForward.id(),
            input: &flat,
            routes: &routes,
            pass,
        };
        let routed = match instrumentation.observer() {
            Some(observer) => eredu_runtime::with_routed_unit_observer(
                observer,
                point.path(),
                request,
                |request| {
                    provider.forward_grouped_tensor_parallel(
                        experts,
                        request,
                        B::parallel_size(parallel),
                        context,
                    )
                },
            )
            .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?,
            None => provider
                .forward_grouped_tensor_parallel(
                    experts,
                    request,
                    B::parallel_size(parallel),
                    context,
                )
                .map_err(Error::backend_retained_source)?,
        };
        let routed =
            eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(routed, parallel, context)?;
        let shared = shared
            .as_mut()
            .map(|shared| {
                if instrumentation.observer().is_none() {
                    return shared.forward_feed_forward_parallel(input, parallel, context);
                }
                instrumentation.with_scope("shared", |instrumentation| {
                    let input = instrumentation.apply("feed_forward.input", input.clone())?;
                    let value = shared.forward_feed_forward_parallel_observed(
                        &input,
                        parallel,
                        context,
                        instrumentation,
                    )?;
                    let value = instrumentation.apply("feed_forward.write", value)?;
                    instrumentation.apply("feed_forward.output", value)
                })
            })
            .transpose()?;
        completed::<B>(
            point,
            &routes,
            routed,
            shared,
            shape,
            context,
            instrumentation,
        )
    }
}
