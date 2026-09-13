use super::*;
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    RoutedExpertTensorParallelOutput, TensorParallelRoutedExpertProvider,
};

#[derive(Clone, Copy, Debug)]
enum ProviderMode {
    Complete,
    Partial,
    PostBias,
}
struct Provider(ProviderMode);
impl RoutedExpertProvider<NumericBackend> for Provider {
    type Error = Error;
    fn forward_grouped(
        &mut self,
        bank: &mut <NumericBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        <ResidentExpertProvider as RoutedExpertProvider<NumericBackend>>::forward_grouped(
            &mut ResidentExpertProvider,
            bank,
            request,
            context,
        )
    }
    fn forward_linear_routed(
        &mut self,
        bank: &mut <NumericBackend as eredu_nn::GroupedNeuralBackend>::LinearGroups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        <ResidentExpertProvider as RoutedExpertProvider<NumericBackend>>::forward_linear_routed(
            &mut ResidentExpertProvider,
            bank,
            request,
            context,
        )
    }
    fn forward_relu2_routed(
        &mut self,
        bank: &mut <NumericBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        <ResidentExpertProvider as RoutedExpertProvider<NumericBackend>>::forward_relu2_routed(
            &mut ResidentExpertProvider,
            bank,
            request,
            context,
        )
    }
}
impl TensorParallelRoutedExpertProvider<NumericBackend> for Provider {
    fn forward_grouped_tensor_parallel(
        &mut self,
        bank: &mut <NumericBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        partitions: usize,
        context: &NumericContext,
    ) -> Result<RoutedExpertTensorParallelOutput<NumericTensor>, Error> {
        assert_eq!(
            partitions, 1,
            "retained provider owns local partition geometry"
        );
        let value = self.forward_grouped(bank, request, context)?;
        Ok(match self.0 {
            ProviderMode::Complete => RoutedExpertTensorParallelOutput::Complete(value),
            ProviderMode::Partial | ProviderMode::PostBias => {
                let bias = matches!(self.0, ProviderMode::PostBias).then(|| {
                    NumericTensor::new(value.shape.clone(), vec![0.375; value.data.len()])
                });
                RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(value, bias),
                )
            }
        })
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        bank: &mut <NumericBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        partitions: usize,
        context: &NumericContext,
    ) -> Result<RoutedExpertTensorParallelOutput<NumericTensor>, Error> {
        <ResidentExpertProvider as TensorParallelRoutedExpertProvider<NumericBackend>>::forward_relu2_routed_tensor_parallel(&mut ResidentExpertProvider, bank, request, partitions, context)
    }
}

#[test]
fn observed_shared_terms_preserve_fused_reduction_and_literal_post_bias() {
    let context = NumericContext::default();
    let mut config = v3_config(None);
    config["num_hidden_layers"] = 2.into();
    let args = deepseek::parse_v3_config(&config).unwrap();
    let policy = deepseek::v3::moe_policy(&args, 1).unwrap();
    let block = deepseek::moe::RoutedPlusShared::<NumericBackend>::new(&policy, &context).unwrap();
    let input = NumericTensor::new(
        [1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.3 - 1.1).sin()).collect(),
    );
    for mode in [
        ProviderMode::Complete,
        ProviderMode::Partial,
        ProviderMode::PostBias,
    ] {
        for target in [
            None,
            Some("sparse.shared.input"),
            Some("sparse.shared.units"),
            Some("sparse.shared.write"),
            Some("sparse.shared.output"),
        ] {
            let make_observer = || Components {
                zero: target.map(|path| (path, 1, 2)),
                ..Default::default()
            };
            let mut ordinary_capture = SharedCapture(make_observer());
            block
                .clone()
                .forward_with_provider_observed(
                    "sparse",
                    &input,
                    deepseek::moe::RouteSource::Learned,
                    ExpertPass::Prefill,
                    &mut ResidentExpertProvider,
                    &context,
                    &mut ordinary_capture,
                )
                .unwrap();
            let routed = &ordinary_capture.0.values["routed"];
            let shared = &ordinary_capture.0.values["shared"];
            // Three equal native terms model the collective without a diagnostic
            // reduction. The post-bias must be added once after that collective.
            let expected = NumericTensor::new(
                input.shape.clone(),
                routed
                    .data
                    .iter()
                    .zip(&shared.data)
                    .map(|(r, s)| match mode {
                        ProviderMode::Complete => r + 3.0 * s,
                        ProviderMode::Partial => 3.0 * (r + s),
                        ProviderMode::PostBias => 3.0 * (r + s) + 0.375,
                    })
                    .collect(),
            );
            let mut capture = make_observer();
            let mut reductions = 0;
            let actual = block
                .clone()
                .forward_tensor_parallel_with_provider_observed(
                    "sparse",
                    &input,
                    deepseek::moe::RouteSource::Learned,
                    ExpertPass::Prefill,
                    &mut Provider(mode),
                    &context,
                    &mut capture,
                    |mut value, _| {
                        reductions += 1;
                        value.data.iter_mut().for_each(|value| *value *= 3.0);
                        Ok(value)
                    },
                )
                .unwrap();
            assert_eq!(reductions, 1, "{mode:?} {target:?}");
            assert_tensor_exact(
                &actual,
                &expected,
                "actual edited shared term enters ordinary fused sum",
            );
            assert_tensor_exact(
                &capture.values["sparse.shared.output.effective"],
                shared,
                "shared callback retains its actual unreduced term",
            );
            assert_tensor_exact(
                &capture.values["sparse.output"],
                &expected,
                "combined seam is complete",
            );
            if target.is_none() {
                let mut ordinary_reductions = 0;
                let ordinary = block
                    .clone()
                    .forward_tensor_parallel_with_provider(
                        &input,
                        deepseek::moe::RouteSource::Learned,
                        ExpertPass::Prefill,
                        &mut Provider(mode),
                        &context,
                        |mut value, _| {
                            ordinary_reductions += 1;
                            value.data.iter_mut().for_each(|value| *value *= 3.0);
                            Ok(value)
                        },
                    )
                    .unwrap();
                assert_eq!(ordinary_reductions, reductions);
                assert_tensor_exact(
                    &ordinary,
                    &actual,
                    "disabled and no-op observed work preserve fused equation",
                );
            }
        }
    }
}
