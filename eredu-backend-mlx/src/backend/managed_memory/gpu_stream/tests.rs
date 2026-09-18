use super::*;
use crate::backend::{MlxAcceleratorFamily, MlxBackend, MlxDeviceIdentity};
use safemlx::{Array, Device, DeviceType};

fn isolated(name: &str, body: impl FnOnce()) {
    if std::env::var("EREDU_GPU_STREAM_CASE").as_deref() == Ok(name) {
        body();
        println!("GPU_STREAM_OK:{name}");
        return;
    }
    let test = format!("backend::managed_memory::gpu_stream::tests::{name}");
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--nocapture"])
        .env("EREDU_GPU_STREAM_CASE", name)
        .output()
        .unwrap();
    let output = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{output}\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.contains(&format!("GPU_STREAM_OK:{name}")),
        "{output}"
    );
}
fn qualified() -> bool {
    let layout = PreparedGpuStream::<SharedNativeInitializationCustody>::layout();
    if std::env::var_os("EREDU_REQUIRE_GPU_STREAM_QUALIFICATION").is_some() {
        assert!(layout.is_ok(), "{layout:?}");
    }
    match layout {
        Ok(layout) => {
            assert_eq!(layout.platform_objects(), (1, 1, 1));
            assert!(layout
                .requests()
                .into_iter()
                .all(|(bytes, alignment)| bytes > 0 && alignment.is_power_of_two()));
            true
        }
        Err(GpuStreamRegistrationCause::UnknownLayout) => false,
        Err(error) => panic!("unexpected GPU stream qualification: {error}"),
    }
}
#[test]
fn gpu_stream_birth_preserves_short_source_and_factory_accounts_execute_nonzero() {
    isolated(
        "gpu_stream_birth_preserves_short_source_and_factory_accounts_execute_nonzero",
        || {
            if !qualified() {
                return;
            }
            let pool = super::super::domain();
            super::super::input_allocator::prepare_admitted(&pool).unwrap();
            let target = GpuStreamTarget::for_initialized(
                super::super::metal_device::admitted_owner(&pool).unwrap(),
                super::super::scheduler::admitted_owner(&pool).unwrap(),
            )
            .unwrap();
            let layout = PreparedGpuStream::<SharedNativeInitializationCustody>::layout().unwrap();
            let plan = Initializer { layout, target };
            let bytes =
                WorkingMemoryPool::shared_native_initialization_required_bytes(&plan).unwrap();
            let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
            let failure = short.initialize_shared_native(plan).unwrap_err();
            assert!(matches!(
                failure.accounting_failure(),
                Some(WorkingMemoryError::BudgetExceeded { .. })
            ));
            assert!(failure.rejected_plan().is_some());
            assert!(failure.constructor_failure().is_none());
            assert_eq!(short.used_bytes().unwrap(), 0);
            drop(failure);

            let before = pool.used_bytes().unwrap();
            let streams = PreparedExecutionStreams::for_factory(&pool)
                .unwrap()
                .expect("qualified fresh creator");
            streams.validate_pool(&pool).unwrap();
            assert!(matches!(
                streams.validate_pool(&short),
                Err(MlxStreamOwnershipError::Accounting(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            let ExecutionStream::Gpu(execution)=&streams.execution else{panic!("actual GPU execution owner")};
            let held = execution.0.original_bytes()
                + streams.source.registration_owner().original_bytes()
                + streams.worker.worker_owner().original_bytes();
            assert_eq!(execution.0.original_bytes(), bytes);
            assert_eq!(pool.used_bytes().unwrap(), before + held);
            let identity = MlxDeviceIdentity::from_realized_device(
                &Device::new(DeviceType::Gpu, 0),
                Some(MlxAcceleratorFamily::Metal),
            )
            .unwrap();
            let backend = MlxBackend::for_prepared_execution_plan(streams, identity);
            backend.validate_original_stream_owners().unwrap();
            // These are actual ordinary numerical operations using the newly owned
            // queues. Their later payload/task allocations are separate from birth.
            let left = Array::from_slice(&[3.0f32, 5.0], &[2]);
            let right = Array::from_slice(&[2.0f32, 7.0], &[2]);
            let gpu = left.add(&right, backend.stream()).unwrap();
            assert_eq!(
                gpu.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
                &[5.0, 12.0]
            );
            let cpu = left.multiply(&right, backend.weights_stream()).unwrap();
            assert_eq!(
                cpu.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
                &[6.0, 35.0]
            );
            backend.synchronize().unwrap();
            // Even an ordinary wrapper of these scalars cannot invent the backend's
            // retained source-account capability.
            let borrowed = MlxBackend::new(backend.stream(), backend.weights_stream());
            assert!(matches!(
                borrowed.validate_original_stream_owners(),
                Err(crate::backend::Error::PrefillControl(
                    WorkingMemoryError::UnknownBound
                ))
            ));
            drop((borrowed, cpu, gpu, left, right, backend));
            safemlx::reclaim_allocation_owners();
            assert_eq!(
                pool.used_bytes().unwrap(),
                before + held,
                "process registry/worker still retain birth custody"
            );
            println!("GPU_STREAM_QUALIFIED:factory-and-values");
        },
    );
}
#[test]
fn ordinary_gpu_predecessor_keeps_factory_compatibility_without_original_authority() {
    isolated(
        "ordinary_gpu_predecessor_keeps_factory_compatibility_without_original_authority",
        || {
            if !qualified() {
                return;
            }
            let device = Device::new(DeviceType::Gpu, 0);
            let ordinary = Stream::try_new_with_device(&device).unwrap();
            let pool = super::super::domain();
            assert!(PreparedExecutionStreams::for_factory(&pool)
                .unwrap()
                .is_none());
            let backend = MlxBackend::new(&ordinary, &ordinary);
            assert!(matches!(
                backend.validate_original_stream_owners(),
                Err(crate::backend::Error::PrefillControl(
                    WorkingMemoryError::UnknownBound
                ))
            ));
            let left = Array::from_slice(&[2.0f32, -3.0], &[2]);
            let result = left.add(&left, &ordinary).unwrap();
            assert_eq!(
                result.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
                &[4.0, -6.0]
            );
            println!("GPU_STREAM_QUALIFIED:ordinary-refusal");
        },
    );
}

#[test]
fn retained_copy_environment_outlives_backend_and_keeps_exact_pool_and_host() {
    isolated("retained_copy_environment_outlives_backend_and_keeps_exact_pool_and_host", || {
        if !qualified() { return; }
        use crate::backend::{PreparedOriginalCopyEnvironment, PreparedOriginalCopyEnvironmentError,
            OriginalCopyEnvironmentError};
        use eredu_core::HostPreparationAuthority;
        use eredu_runtime::working_memory::InferenceExecutionIdentity;
        let pool = super::super::domain();
        super::super::input_allocator::prepare_admitted(&pool).unwrap();
        let streams = PreparedExecutionStreams::for_factory(&pool).unwrap().unwrap();
        let identity = MlxDeviceIdentity::from_realized_device(
            &Device::new(DeviceType::Gpu, 0), Some(MlxAcceleratorFamily::Metal)).unwrap();
        let backend = MlxBackend::for_prepared_execution_plan(streams, identity);
        let baseline = pool.used_bytes().unwrap();
        // The policy ceiling covers the actual existing native-source domain;
        // this fixture adds 16 MiB for its separately retained host descriptors.
        let ceiling = baseline.checked_add(1 << 24).unwrap();
        let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), ceiling).unwrap();
        funding.reserve_metadata(HostPreparationAuthority::retention_bytes::<
            eredu_nn::workspace::HostMetadataFunding>().unwrap()).unwrap();
        let host = HostPreparationAuthority::retain(funding.clone());
        let environment = backend.original_copy_environment().unwrap();
        let source_index = environment.stream().get_index().unwrap();
        let source_wrapper = environment.stream().as_ptr().ctx;
        let owner = PreparedOriginalCopyEnvironment::prepare(&environment, &host, &funding).unwrap();
        let alias = owner.clone();
        drop(environment);
        drop((backend, host, funding));
        let loan = alias.loan(&pool).unwrap();
        assert_eq!(loan.stream().get_index().unwrap(), source_index);
        assert_ne!(loan.stream().as_ptr().ctx, source_wrapper);
        drop(loan);
        let other = WorkingMemoryPool::new(1 << 24, 0).unwrap();
        assert!(matches!(alias.loan(&other), Err(PreparedOriginalCopyEnvironmentError::Environment(
            OriginalCopyEnvironmentError::Memory(WorkingMemoryError::IdentityMismatch)))));
        let held = pool.used_bytes().unwrap();
        assert!(held > baseline);
        drop(owner);
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(), held, "last alias retains actual wrapper and metadata");
        drop(alias);
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(), baseline, "metadata retires after native wrapper ownership");
    });
}

#[test]
fn cpu_factory_retains_actual_stream_worker_and_copy_environment() {
    isolated("cpu_factory_retains_actual_stream_worker_and_copy_environment",||{
        if !qualified(){return;}
        let pool=super::super::domain();
        super::super::input_allocator::prepare_admitted(&pool).unwrap();
        let before=pool.used_bytes().unwrap();
        let streams=PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap();
        let ExecutionStream::Cpu(execution)=&streams.execution else{panic!("actual CPU execution owner")};
        let held=execution.original_bytes()+streams.source.registration_owner().original_bytes()
            +streams.worker.worker_owner().original_bytes();
        assert_eq!(pool.used_bytes().unwrap(),before+held);
        let marker=safemlx::StreamCopyPlan::<()>::capture(streams.execution()).unwrap();
        assert_eq!(marker.device_type(),DeviceType::Cpu);
        assert_eq!(marker.device_index(),0);
        assert!(!marker.matches_source(streams.source()));
        streams.observe_idle(&pool).unwrap();
        let foreign=WorkingMemoryPool::new(held,0).unwrap();
        assert!(streams.validate_pool(&foreign).is_err());
        let identity=MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu,0),None).unwrap();
        let backend=MlxBackend::for_prepared_execution_plan(streams,identity);
        backend.validate_original_stream_owners().unwrap();
        let environment=backend.original_copy_environment().unwrap();
        assert!(marker.matches_source(environment.stream()));
        assert!(environment.pool().same_domain(&pool));
        // This validates native ownership, not qualification for a CPU model.
        drop(environment);
        drop(backend);
        safemlx::reclaim_allocation_owners();
        assert!(pool.used_bytes().unwrap() > before, "actual process registrations retain native birth custody");
    });
}

#[test]
fn cpu_factory_retains_selected_matmul_context_and_exact_copy_account() {
    isolated("cpu_factory_retains_selected_matmul_context_and_exact_copy_account",||{
        if !qualified(){return;}
        let pool=super::super::domain();
        super::super::input_allocator::prepare_admitted(&pool).unwrap();
        let choice=crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
            eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let geometry=choice.selected().geometry(2,17,19,23,1).unwrap();
        assert!(choice.eval_layout(geometry,false).unwrap().backing_births()==1);
        let before=pool.used_bytes().unwrap();
        let streams=PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool,choice).unwrap().unwrap();
        let ExecutionStream::Cpu(execution)=&streams.execution else{panic!("actual CPU execution owner")};
        let held=execution.original_bytes()+streams.source.registration_owner().original_bytes()
            +streams.worker.worker_owner().original_bytes();
        assert_eq!(pool.used_bytes().unwrap(),before+held);
        let selected=safemlx::StreamCopyPlan::<()>::capture(streams.execution()).unwrap();
        let retiring=execution.wrapper_control_bytes();
        assert_eq!(selected.cpu_matmul(),safemlx::CpuMatmulKernel::Float32Tiles);
        assert_eq!(safemlx::StreamCopyPlan::<()>::capture(streams.source()).unwrap().cpu_matmul(),safemlx::CpuMatmulKernel::PlatformDefault);
        streams.observe_idle(&pool).unwrap();
        let foreign=WorkingMemoryPool::new(held,0).unwrap();assert!(streams.validate_pool(&foreign).is_err());
        let identity=MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu,0),None).unwrap();
        let backend=MlxBackend::for_prepared_execution_plan(streams,identity);
        let environment=backend.original_copy_environment().unwrap();
        assert!(selected.matches_source(environment.stream()));
        assert_eq!(safemlx::StreamCopyPlan::<()>::capture(environment.stream()).unwrap().cpu_matmul(),safemlx::CpuMatmulKernel::Float32Tiles);
        drop(environment);drop(backend);safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(),before+held-retiring,
            "both local wrapper accounts retire; registered CPU stream and worker accounts survive");
    });
}
