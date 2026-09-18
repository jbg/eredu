use super::*;
use eredu_nn::{GroupedGatedProductSpec,GroupedLinearSpec,GroupedRelu2Spec,GroupedProjectionSpec,
    GatedProductGroupLayout,GatedProductPolicy,GroupedLinearActivation,LinearFormatSpec,ParameterSpec};
use eredu_nn::workspace::{WorkspaceExpertKernel,WorkspaceFloatingType,WorkspaceRepresentation,
    HostMetadataAccount,HostMetadataFundingError};
use std::sync::atomic::{AtomicUsize,Ordering};
#[derive(Debug,Default)]
struct Account(AtomicUsize);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError>{
        self.0.fetch_update(Ordering::SeqCst,Ordering::SeqCst,|total|total.checked_add(bytes))
            .map(|_|()).map_err(|_|HostMetadataFundingError::Overflow)
    }
}
fn projection(name:&str)->GroupedProjectionSpec {
    GroupedProjectionSpec::new(ParameterSpec::trainable(name).unwrap(),None,
        LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap()
}
fn components(recipe:SpeculativeNumericalRecipe)->[u64;6] {
    [recipe.graph_capacity as u64,recipe.record_capacity as u64,recipe.storage.mutable_bytes(),
        recipe.storage.maximum_births() as u64,recipe.kernels as u64,recipe.controls]
}
#[test]
fn expert_branch_census_bounds_actual_recipes_across_chunk_and_singleton_transitions() {
    let _sources=crate::tests::support::test_utils::initialize_original_sources();
    let gated=GroupedGatedProductSpec::new(2,32,32,32,GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed{gate_up:projection("read"),down:projection("write")}).unwrap();
    let linear=GroupedLinearSpec::new(2,32,32,GroupedLinearActivation::Identity,projection("linear")).unwrap();
    let relu=GroupedRelu2Spec::new(2,32,32,projection("up"),projection("down")).unwrap();
    let mechanism=ResidentExecutionMechanisms::Metal(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    for kernel in [WorkspaceExpertKernel::Gated(&gated),WorkspaceExpertKernel::Linear(&linear),WorkspaceExpertKernel::Relu2(&relu)] {
        let funding=HostMetadataFunding::new(Account::default()).unwrap();
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone()).unwrap();
        let view=WorkspaceExpertRegionView{addressable:None,bank:0,unit:0,prefill:true,
            group:eredu_core::CollectiveGroupId::new(0),rank:0,peers:2,source_rows:65,routes_per_row:1,
            owners:&[0,1,0,1],owner_local:&[0,0,1,1],kernel,tensor_partitions:None,provider_tensor_group:None,provider_wave_group:None,
            movement:eredu_nn::workspace::WorkspaceExpertMovementPopulation{
                row_gathers:1,scalar_gathers:2,zeros:1,indexed_adds:1},
            transfers:eredu_nn::workspace::WorkspaceExpertTransfers([None;9])};
        let declaration=view.retain(&context).unwrap();
        let repr=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
        let floating=|shape:&[i32]|context.layout(shape,WorkspaceDtype::Float32).unwrap().with_representation(repr);
        let mut inputs=vec![floating(&[65,32]),context.layout(&[65,1],WorkspaceDtype::Int32).unwrap(),
            floating(&[65,1]),floating(&[65,1])];
        match kernel {
            WorkspaceExpertKernel::Gated(_)=>{inputs.push(floating(&[2,64,32]));inputs.push(floating(&[2,32,32]));},
            WorkspaceExpertKernel::Linear(_)=>inputs.push(floating(&[2,32,32])),
            WorkspaceExpertKernel::Relu2(_)=>{inputs.push(floating(&[2,32,32]));inputs.push(floating(&[2,32,32]));},
        }
        // Cache actual recipes once: the assertion compares the production
        // finite candidate fold with every real interior recipe, not a copy of
        // the candidate-selection algorithm. No tensor/native work executes.
        let actual=(0..=130).map(|rows|components(local_trace(&declaration,&inputs,rows,mechanism,&funding).unwrap_or_else(|cause|panic!("{kernel:?}, actual rows={rows}: {cause}")))).collect::<Vec<_>>();
        for maximum in [0,1,2,31,32,33,63,64,65,66,95,96,97,127,128,129,130] {
            let mut ceiling=actual[0];
            for rows in super::super::super::grouped::expert_row_candidates(kernel,maximum) {
                for (bound,value) in ceiling.iter_mut().zip(actual[rows]){*bound=(*bound).max(value);}
            }
            for (rows,recipe) in actual[..=maximum].iter().enumerate() {
                for (component,(actual,bound)) in recipe.iter().zip(ceiling).enumerate() {
                    assert!(*actual<=bound,"{kernel:?}: rows={rows}, maximum={maximum}, component={component}, actual={actual}, ceiling={bound}");
                }
            }
        }
    }
}

#[test]
fn expert_observed_child_keeps_sparse_sources_inside_its_native_frontier() {
    let _sources=crate::tests::support::test_utils::initialize_original_sources();
    let gated=GroupedGatedProductSpec::new(2,32,32,32,GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed{gate_up:projection("read"),down:projection("write")}).unwrap();
    let kernel=WorkspaceExpertKernel::Gated(&gated);
    let mechanism=ResidentExecutionMechanisms::Metal(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let funding=HostMetadataFunding::new(Account::default()).unwrap();
    let context=WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone()).unwrap();
    let declaration=WorkspaceExpertRegionView{addressable:None,bank:0,unit:0,prefill:true,
        group:eredu_core::CollectiveGroupId::new(0),rank:0,peers:2,source_rows:65,routes_per_row:1,
        owners:&[0,1,0,1],owner_local:&[0,0,1,1],kernel,tensor_partitions:None,
        provider_tensor_group:None,provider_wave_group:None,
        movement:eredu_nn::workspace::WorkspaceExpertMovementPopulation{row_gathers:1,scalar_gathers:2,zeros:1,indexed_adds:1},
        transfers:eredu_nn::workspace::WorkspaceExpertTransfers([None;9])}.retain(&context).unwrap();
    let repr=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
    let floating=|shape:&[i32]|context.layout(shape,WorkspaceDtype::Float32).unwrap().with_representation(repr);
    let inputs=[floating(&[65,32]),context.layout(&[65,1],WorkspaceDtype::Int32).unwrap(),
        floating(&[65,1]),floating(&[65,1]),floating(&[2,64,32]),floating(&[2,32,32])];
    let empty=crate::backend::array_copy::CaptureNativePopulation::default();
    let capture=crate::backend::array_copy::CaptureNativePopulation::routed_partition().unwrap();
    let baseline=ExpertLocalObservationSource::new(empty,empty,empty).unwrap();
    let capture=ExpertLocalObservationSource::new(empty,capture,capture).unwrap();
    for rows in [0usize,1,65] {
        let plain=local_trace_with_outputs(&declaration,&inputs,rows,mechanism,&funding,Some(baseline),|_|Ok(())).unwrap();
        let observed=local_trace_with_outputs(&declaration,&inputs,rows,mechanism,&funding,Some(capture),|_|Ok(())).unwrap();
        let schedule=mechanism.grouped_observation_schedule(declaration.kernel(),rows as u32).unwrap().unwrap();
        let envelope=eredu_nn::workspace::WorkspaceGroupedObservationEnvelope::new(rows,1,32,schedule,false).unwrap();
        let added=10*envelope.maximum_callbacks();
        assert_eq!(observed.completion.nested_completions,plain.completion.nested_completions+added);
        assert_eq!(observed.completion.nested_traversal().unwrap().limits().roots,
            (1+plain.completion.validation_roots+added).max(3));
        assert!(observed.graph_capacity>=plain.graph_capacity);
        assert!(observed.record_capacity>=plain.record_capacity);
        assert!(observed.storage.mutable_bytes()>=plain.storage.mutable_bytes());
        if rows>0 {assert!(observed.controls>plain.controls);}
    }
}

#[test]
fn cpu_expert_local_empty_and_singleton_children_have_complete_typed_sources() {
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected = crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
        eredu_nn::CpuMatmulImplementation::Float32Tiles,
    )
    .unwrap();
    let cpu = crate::backend::nn::workspace::MlxCpuWorkspaceMechanisms::new(
        ordinary.allocation(),
        selected,
    );
    let mechanism = ResidentExecutionMechanisms::Cpu { ordinary, cpu };
    let gated = GroupedGatedProductSpec::new(
        2,
        8,
        8,
        8,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: projection("read"),
            down: projection("write"),
        },
    )
    .unwrap();
    let linear = GroupedLinearSpec::new(
        2,
        8,
        5,
        GroupedLinearActivation::Identity,
        projection("linear"),
    )
    .unwrap();
    let relu = GroupedRelu2Spec::new(2, 8, 8, projection("up"), projection("down")).unwrap();
    for kernel in [
        WorkspaceExpertKernel::Gated(&gated),
        WorkspaceExpertKernel::Linear(&linear),
        WorkspaceExpertKernel::Relu2(&relu),
    ] {
        let funding = HostMetadataFunding::new(Account::default()).unwrap();
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone()).unwrap();
        let declaration = WorkspaceExpertRegionView {
            addressable: None,
            bank: 0,
            unit: 0,
            prefill: true,
            group: eredu_core::CollectiveGroupId::new(0),
            rank: 0,
            peers: 2,
            source_rows: 2,
            routes_per_row: 1,
            owners: &[0, 1, 0, 1],
            owner_local: &[0, 0, 1, 1],
            kernel,
            tensor_partitions: None,
            provider_tensor_group: None,
            provider_wave_group: None,
            movement: eredu_nn::workspace::WorkspaceExpertMovementPopulation {
                row_gathers: 1,
                scalar_gathers: 2,
                zeros: 1,
                indexed_adds: 1,
            },
            transfers: eredu_nn::workspace::WorkspaceExpertTransfers([None; 9]),
        }
        .retain(&context)
        .unwrap();
        let floating = |shape: &[i32]| {
            context
                .layout(shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                )))
        };
        let mut inputs = vec![
            floating(&[2, 8]),
            context.layout(&[2, 1], WorkspaceDtype::Int32).unwrap(),
            floating(&[2, 1]),
            floating(&[2, 1]),
        ];
        match kernel {
            WorkspaceExpertKernel::Gated(_) => {
                inputs.push(floating(&[2, 16, 8]));
                inputs.push(floating(&[2, 8, 8]));
            }
            WorkspaceExpertKernel::Linear(_) => inputs.push(floating(&[2, 5, 8])),
            WorkspaceExpertKernel::Relu2(_) => {
                inputs.push(floating(&[2, 8, 8]));
                inputs.push(floating(&[2, 8, 8]));
            }
        }
        for rows in [0usize, 1, 2, 4] {
            let recipe = local_trace_with_outputs(
                &declaration,
                &inputs,
                rows,
                mechanism,
                &funding,
                None,
                |outputs| {
                    assert_eq!(outputs.len(), 1);
                    assert_eq!(
                        outputs[0].shape(),
                        &[
                            rows as i32,
                            if matches!(kernel, WorkspaceExpertKernel::Linear(_)) {
                                5
                            } else {
                                8
                            }
                        ]
                    );
                    assert_eq!(
                        outputs[0].layout().representation().unwrap().dtype(),
                        WorkspaceFloatingType::Float32
                    );
                    Ok(())
                },
            )
            .unwrap_or_else(|cause| panic!("{kernel:?}, actual rows={rows}: {cause}"));
            assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
            assert_eq!(recipe.kernels, 0);
            assert!(recipe.controls > 0);
            // Even the empty typed-zero worker owns its eager scalar seed;
            // empty output and its reshape never invent a physical output birth.
            assert!(recipe.storage.maximum_births() > 0);
        }
    }
}
