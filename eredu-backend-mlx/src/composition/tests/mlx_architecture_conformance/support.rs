fn mlx_execution() -> ExecutionContext {
    static AVAILABLE: OnceLock<()> = OnceLock::new();
    AVAILABLE.get_or_init(|| {
        #[cfg(feature = "metal")]
        {
            match safemlx::metal::is_available() {
                Ok(true) => {}
                Ok(false) => panic!("native MLX Metal execution is required for conformance tests"),
                Err(error) => panic!("MLX Metal availability probe failed: {error}"),
            }
        }
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        safemlx::ops::zeros::<f32>(&[1], execution.stream())
            .and_then(|probe| probe.evaluated().map(|_| ()))
            .unwrap_or_else(|error| {
                panic!("native MLX execution is required for conformance tests: {error}")
            });
    });
    ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
}

macro_rules! execute_group {
    ($architecture_ty:ty, $state_ty:ty, $architecture:expr, $state:expr, $context:expr, $group:expr, $initial:expr, $dependencies:expr, $stream:expr) => {{
        let mut hidden = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::begin_execution_group(
            &mut $architecture,
            $group,
            $initial,
            $dependencies,
            &mut $state,
            &mut $context,
            $stream,
        )
        .unwrap();
        let unit_count =
            <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::group_unit_count(
                &$architecture,
                $group,
            )
            .unwrap();
        for index in 0..unit_count {
            let mut unit =
                <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::build_unit(
                    &$architecture,
                    $group,
                    index,
                    $stream,
                )
                .unwrap();
            hidden =
                <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::forward_unit(
                    &mut $architecture,
                    $group,
                    index,
                    &mut unit,
                    &hidden,
                    &mut $state,
                    &mut $context,
                    $stream,
                )
                .unwrap();
        }
        hidden = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::complete_execution_group(
            &mut $architecture,
            $group,
            &hidden,
            &mut $state,
            &mut $context,
            $stream,
        )
        .unwrap();
        hidden
    }};
}

macro_rules! execute_target_group {
    ($architecture_ty:ty, $state_ty:ty, $architecture:expr, $state:expr, $input:expr, $shape:expr, $stream:expr) => {{
        let token_validation_scope = TokenValidationScope::begin().unwrap();
        let LayeredForwardState {
            hidden: initial,
            mut context,
        } = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::begin_forward(
            &mut $architecture,
            $input,
            &mut $state,
            $stream,
        )
        .unwrap();
        let hidden = execute_group!(
            $architecture_ty,
            $state_ty,
            $architecture,
            $state,
            context,
            0,
            &initial,
            &[],
            $stream
        );
        let logits =
            <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::finish_forward(
                &mut $architecture,
                &hidden,
                &mut $state,
                &context,
                $stream,
            )
            .unwrap();
        assert_eq!(logits.shape(), $shape);
        let token_validations = token_validation_scope.finish();
        async_eval_with_event(std::iter::once(logits.as_array()).chain(token_validations.arrays()))
            .unwrap()
            .synchronize()
            .unwrap();
        token_validations.validate_completed().unwrap();
    }};
}

macro_rules! execute_vision_text_groups {
    ($architecture_ty:ty, $state_ty:ty, $architecture:expr, $state:expr, $input:expr, $shape:expr, $stream:expr) => {{
        let token_validation_scope = TokenValidationScope::begin().unwrap();
        let LayeredForwardState {
            hidden: initial,
            mut context,
        } = <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::begin_forward(
            &mut $architecture,
            $input,
            &mut $state,
            $stream,
        )
        .unwrap();
        let vision = execute_group!(
            $architecture_ty,
            $state_ty,
            $architecture,
            $state,
            context,
            0,
            &initial,
            &[],
            $stream
        );
        let hidden = execute_group!(
            $architecture_ty,
            $state_ty,
            $architecture,
            $state,
            context,
            1,
            &initial,
            &[&vision],
            $stream
        );
        let logits =
            <$architecture_ty as LayeredArchitecture<MlxNeuralBackend, $state_ty>>::finish_forward(
                &mut $architecture,
                &hidden,
                &mut $state,
                &context,
                $stream,
            )
            .unwrap();
        assert_eq!(logits.shape(), $shape);
        let token_validations = token_validation_scope.finish();
        async_eval_with_event(std::iter::once(logits.as_array()).chain(token_validations.arrays()))
            .unwrap()
            .synchronize()
            .unwrap();
        token_validations.validate_completed().unwrap();
    }};
}
