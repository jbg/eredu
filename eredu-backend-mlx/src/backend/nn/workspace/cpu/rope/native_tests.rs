//! Actual CPU YaRN wrapper under the exact shared cold equation recipe.
use super::*;
use crate::{backend::{MlxBackend,MlxDeviceIdentity,
    managed_memory::gpu_stream::PreparedExecutionStreams,nn::shared::MlxNeuralBackend},MlxTensor};
use eredu_nn::{NeuralBackend,Parameterized,RotaryOperator,RotaryPosition,RotarySpec,Tensor};
use safemlx::{Array,Device,DeviceType,OriginalBufferBudget,OriginalScopeObserver,
    PrefillRoots,PrefillRootsRuntime,PreparedOriginalBufferBudget,PreparedPrefillFailure,
    PreparedSubmissionGraphQuota,PreparedSubmissionRecordQuota,PreparedSubmissionScopeOwner,SubmissionScope};
use std::sync::{Arc,atomic::{AtomicBool,Ordering}};

#[derive(Debug)]
struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime {fn drop(&mut self){self.0.store(true,Ordering::SeqCst);}}
fn spec(width:i32)->RotarySpec {
    RotarySpec{arithmetic:RotaryArithmetic::Native,algorithm:RotaryAlgorithm::Yarn{
        factor:4.0,original_max_positions:16,beta_fast:8.0,beta_slow:1.0,amplitude:1.35,truncate:true},
        dimensions:width,traditional:false,base:10000.0}
}

// The pinned MLX CPU simd/math.h uses Cephes degree-7 sine / degree-8
// cosine, after three F32 FMAs reduce the angle by j*pi/4. Above 2^24,
// the rounded quadrant selector can leave |r| > pi/4 (1.106 at 16_777_218).
// A constant small-angle tolerance therefore does not describe this worker.
// Keep libm as the independent oracle and bound the actual approximation:
// coefficient differences from Taylor + its next term, measured phase loss
// from the three reductions, and the F32 roundoff of each polynomial path.
// This is an error envelope, not a copy of the native polynomial evaluator.
fn input_products_oracle_error(theta: f32, left: f32, right: f32, amplitude: f32) -> f64 {
    let j = ((theta.abs() * 1.27323954473516_f32) as u32 + 1) & !1;
    let r = (j as f32).mul_add(-0.78515625_f32, theta.abs());
    let r = (j as f32).mul_add(-2.4187564849853515625e-4_f32, r);
    let r = (j as f32).mul_add(-3.77489497744594108e-8_f32, r);
    let x = f64::from(r.abs());
    // All exercised coordinates stay in the alternating Taylor remainder's
    // decreasing-term interval, including the large-coordinate quadrant case.
    assert!(x <= std::f64::consts::FRAC_PI_2);
    let phase = (f64::from(theta.abs())
        - (f64::from(j) * std::f64::consts::FRAC_PI_4 + f64::from(r))).abs();
    let gamma = |operations: u32| {
        let nu = f64::from(operations) * f64::from(f32::EPSILON) / 2.0;
        nu / (1.0 - nu)
    };
    let sin = [
        f64::from(-1.6666654611e-1_f32),
        f64::from(8.3321608736e-3_f32),
        f64::from(-1.9515295891e-4_f32),
    ];
    let cos = [
        f64::from(4.166664568298827e-2_f32),
        f64::from(-1.388731625493765e-3_f32),
        f64::from(2.443315711809948e-5_f32),
    ];
    // Including the reused rounded x*x, the longest coefficient paths have
    // seven (sine) and ten (cosine) roundings. Absolute coefficient sums also
    // cover cancellation in Horner evaluation.
    let sine = (sin[0] + 1.0/6.0).abs() * x.powi(3)
        + (sin[1] - 1.0/120.0).abs() * x.powi(5)
        + (sin[2] + 1.0/5040.0).abs() * x.powi(7)
        + x.powi(9)/362880.0
        + gamma(7) * (x + sin[0].abs()*x.powi(3)
            + sin[1].abs()*x.powi(5) + sin[2].abs()*x.powi(7));
    let cosine = (cos[0] - 1.0/24.0).abs() * x.powi(4)
        + (cos[1] + 1.0/720.0).abs() * x.powi(6)
        + (cos[2] - 1.0/40320.0).abs() * x.powi(8)
        + x.powi(10)/3628800.0
        + gamma(10) * (1.0 + 0.5*x*x + cos[0].abs()*x.powi(4)
            + cos[1].abs()*x.powi(6) + cos[2].abs()*x.powi(8));
    // Either polynomial may serve sin/cos after quadrant selection. Both
    // trig functions are 1-Lipschitz with respect to the phase discrepancy.
    let trig = sine.max(cosine) + phase;
    let magnitude = f64::from(amplitude.abs())
        * (f64::from(left.abs()) + f64::from(right.abs()));
    // Amplitude multiplication, input multiplication, then sum/difference.
    magnitude * (trig + gamma(3) * (1.0 + trig))
}

#[test]
fn cpu_yarn_source_counts_flattened_batches_and_retained_frequency_operand() {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    for (batch,heads,positions,width) in [(1,1,1,4),(1,2,2,4),(2,3,3,8)] {
        let context=WorkspaceContext::new(cpu);
        let input=WorkspaceTensor::existing(context.layout(&[batch,heads,positions,width],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let mut rope=WorkspaceBackend::rotary(spec(width),&context).unwrap();
        context.begin_span();
        let output=rope.forward(&input,RotaryPosition::Offset(5),&context).unwrap();
        assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
        let report=context.finish_report(&[output]).unwrap();
        let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.seeds,3*(batch*heads) as usize+1);
        assert_eq!(plan.population.hidden_leaves,1);
        assert_eq!(plan.population.maximum_operands,2.max((batch*heads) as usize));
        let captures=5.max(1+2*(batch*heads) as usize);
        assert_eq!(plan.population.maximum_captures,captures);
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.completion.graph.maximum_operands(),4.max((batch*heads) as usize));
        assert_eq!(recipe.completion.traversal.limits().captures,8.max(captures));
        assert_eq!(recipe.storage.maximum_births(),plan.population.births+plan.seeds);
        assert_eq!(recipe.completion.traversal.limits().arrays,
            plan.population.primitives+plan.seeds+plan.population.hidden_leaves+3);
    }
}

#[test]
#[ignore="requires qualified native allocator and selected CPU execution"]
fn original_cpu_yarn_batched_wrapper_matches_ordinary_and_retires_exact_custody() {
    run_native(false);
}
#[test]
#[ignore="requires qualified native allocator and selected CPU execution"]
fn original_cpu_input_products_yarn_preserves_transposed_sources_positions_and_custody() {
    run_native(true);
}
fn run_native(explicit:bool) {
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    let streams=PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool,selected).unwrap().unwrap();
    let backend=MlxBackend::for_prepared_execution_plan(streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu,0),None).unwrap());
    let environment=backend.original_copy_environment().unwrap();
    let stream=environment.stream();
    let runtime=PrefillRootsRuntime::prepare_for_stream(stream,stream).unwrap();
    let allocator=environment.input_runtime().unwrap();
    for (batch,heads,positions,width) in [(1,1,1,4),(1,2,2,4),(2,3,3,8)] {
      for transposed in [false,true] {
        if !explicit&&transposed {continue;}
        let offset=if explicit&&transposed {16_777_216}else{5};
        let mut selected_spec=spec(width);
        if explicit {selected_spec.arithmetic=RotaryArithmetic::InputProducts;}
        let shape=[batch,heads,positions,width];let count=(batch*heads*positions*width) as usize;
        let source_shape=if transposed {[batch,positions,heads,width]}else{shape};
        let data=(0..count).map(|i|((i*7%23) as f32-11.0)*0.125).collect::<Vec<_>>();
        let source=Array::from_slice(&data,&source_shape);
        let input=MlxTensor::from_array(if transposed {source.transpose_axes(&[0,2,1,3],stream).unwrap()}else{source.clone()});
        input.as_array().evaluated().unwrap();
        let mut module=MlxNeuralBackend::rotary(selected_spec,stream).unwrap();
        let mut sources=Vec::new();
        assert!(module.visit_retained_values(&mut |value|sources.push(value.clone())));
        assert_eq!(sources.len(),if explicit {2}else{1});
        for value in &sources {value.as_array().evaluated().unwrap();}
        let frequency=sources.last().unwrap();
        let frequencies=frequency.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let reference=module.forward(&input,RotaryPosition::Offset(offset),stream).unwrap();
        let expected=reference.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap();drop(reference);
        assert!(expected.iter().all(|x|x.is_finite())&&expected.iter().any(|x|x.abs()>1e-5));
        // Actual source axes and absolute integer positions feed the independent
        // scalar oracle. InputProducts scales trig values before multiplication;
        // Native scales input first. Preserve their distinct rounding policies.
        for row in 0..(batch*heads*positions) as usize {for col in 0..width as usize/2 {
            let position=row%positions as usize;
            let coordinate=(offset+position as i32) as f32;
            let theta=if explicit {coordinate*frequencies[col]}else{coordinate/frequencies[col]};
            let head=row/positions as usize%heads as usize;
            let batch_index=row/(positions*heads) as usize;
            let source_row=if transposed {(batch_index*positions as usize+position)*heads as usize+head}else{row};
            let mut left=data[source_row*width as usize+col];
            let mut right=data[source_row*width as usize+col+width as usize/2];
            let (values,tolerance)=if explicit {
                let amplitude=f64::from(1.35_f32);
                let cos=f64::from(theta).cos()*amplitude;
                let sin=f64::from(theta).sin()*amplitude;
                ([f64::from(left)*cos-f64::from(right)*sin,
                    f64::from(right)*cos+f64::from(left)*sin],
                    input_products_oracle_error(theta,left,right,1.35))
            }else{
                left*=1.35;right*=1.35;
                let cos=theta.cos();let sin=theta.sin();
                ([f64::from(left*cos-right*sin),f64::from(right*cos+left*sin)],3e-6)
            };
            for (index,value) in [(col,values[0]),(col+width as usize/2,values[1])] {
                assert!((f64::from(expected[row*width as usize+index])-value).abs()<=tolerance,
                    "arithmetic {:?}, transposed={transposed}, shape={shape:?}, row {row}, index {index}, theta={theta}: actual {}, scalar {value}, envelope {tolerance}",selected_spec.arithmetic,expected[row*width as usize+index]);
            }
        }}
        let retained_facts=sources.iter().map(|value|value.as_array().try_descriptor().unwrap().facts()).collect::<Vec<_>>();
        let context=WorkspaceContext::new(cpu);
        let projected=WorkspaceTensor::existing(context.layout(&source_shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let mut quoted=WorkspaceBackend::rotary(selected_spec,&context).unwrap();context.begin_span();
        // The original leaf is the independently owned dense source. Transpose
        // is an actual quoted alias producer, not an assumed detached leaf.
        let projected=if transposed {projected.transpose_axes(&[0,2,1,3],&context).unwrap()}else{projected};
        let output=quoted.forward(&projected,RotaryPosition::Offset(offset),&context).unwrap();
        // Completion of the rotary output need not evaluate an optimized-out
        // singleton view. The backing/stride witness is its own explicit root.
        let output_roots=if transposed {2}else{1};
        let report=if transposed {context.finish_report(&[output,projected])}
            else{context.finish_report(&[output])}.unwrap();
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,output_roots,ordinary,cpu,&context).unwrap();
        let completion=recipe.completion;
        let physical=OriginalBufferBudget::population_layout(&allocator,
            usize::try_from(recipe.storage.mutable_bytes()).unwrap(),recipe.storage.maximum_births()).unwrap().capacity();
        let released=Arc::new(AtomicBool::new(false));let owner=Arc::new(Lifetime(released.clone()));
        let graph=PreparedSubmissionGraphQuota::try_new(recipe.graph_capacity,owner.clone()).unwrap().try_allocate().unwrap();
        let records=PreparedSubmissionRecordQuota::try_new(recipe.record_capacity,owner.clone()).unwrap().try_allocate().unwrap();
        let budget=PreparedOriginalBufferBudget::try_new(&allocator,physical,owner.clone()).unwrap().try_allocate().unwrap();
        let failure=PreparedPrefillFailure::try_new(owner.clone()).unwrap().try_allocate().unwrap();
        let mut roots=PrefillRoots::new_retained(&runtime,output_roots,&graph,&failure).unwrap();
        let mut scope=SubmissionScope::try_begin_retaining(PreparedSubmissionScopeOwner::try_new(owner.clone())
            .unwrap().with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
        scope.enable_scoped_observation().unwrap();scope.require_original_native_controls().unwrap();
        roots.bind_scope(&scope).unwrap();scope.enable_original_native_controls().unwrap();scope.bind_original_buffer_budget(&budget).unwrap();
        let observer=OriginalScopeObserver::require_current().unwrap();
        OperationEvent::validate_traversal_leaf(&source,&observer)
            .unwrap_or_else(|error|panic!("original rotary input leaf explicit={explicit} transposed={transposed} shape={shape:?}: {error}"));
        for (source_index,value) in sources.iter().enumerate() {
            OperationEvent::validate_traversal_leaf(value.as_array(),&observer)
                .unwrap_or_else(|error|panic!("retained rotary leaf explicit={explicit} transposed={transposed} shape={shape:?} source={source_index}, before={:?}: {error}",retained_facts[source_index]));
        }
        let bank=OperationEvent::prepare_resident_graph(completion.graph,&observer).unwrap();
        let scoped_input=if transposed {
            Some(MlxTensor::from_array(source.transpose_axes(&[0,2,1,3],stream).unwrap()))
        }else{None};
        let actual=module.forward(scoped_input.as_ref().unwrap_or(&input),RotaryPosition::Offset(offset),stream).unwrap();drop(bank);
        roots.append(actual.as_array()).unwrap();
        if let Some(view)=&scoped_input {roots.append(view.as_array()).unwrap();}
        roots.complete_current_scope_on_stream_prepared(stream,&completion.traversal)
            .unwrap_or_else(|error|panic!("rotary completion explicit={explicit} transposed={transposed} shape={shape:?}: {error}; status {:?}; native {}/{physical}; graph {}/{}; records {}/{}; traversal {:?}",
                observer.status(),budget.occupied_bytes(),graph.occupied_bytes(),recipe.graph_capacity,
                records.occupied_bytes(),recipe.record_capacity,completion.traversal.limits()));
        assert!(!observer.status().failed());assert!(budget.occupied_bytes()>0&&budget.occupied_bytes()<=physical);
        assert_eq!(actual.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);
        scope.seal();
        // CPU output readiness can precede the signal task's final accepted
        // frontier. Settle through the actual observer before retiring it;
        // record retirement alone never polls unfinished native work.
        let retirement_deadline=std::time::Instant::now()+std::time::Duration::from_secs(10);
        crate::backend::submission_recovery::wait_for_retirement(||{
            assert!(std::time::Instant::now()<retirement_deadline,
                "rotary settlement deadline explicit={explicit} transposed={transposed} shape={shape:?}");
            let (progress,status)=observer.progress().unwrap();
            assert_eq!(progress,safemlx::ScopedSubmissionProgress::Observed);
            assert!(!status.failed()&&!status.blocked(),
                "rotary settlement explicit={explicit} transposed={transposed} shape={shape:?}");
            status.is_settled()
        });
        assert_eq!(observer.retire_completed_records().unwrap(),safemlx::SubmissionRetirement::CompleteSnapshot);
        safemlx::try_with_submission_retirement(||drop((roots,scope,observer,failure,records,graph,budget))).unwrap();drop(owner);
        safemlx::reclaim_allocation_owners();assert!(!released.load(Ordering::SeqCst));
        if let Some(view)=&scoped_input {
            // Completion preserves the same backing and exact transposed
            // strides; the exercised input has not become a contiguous copy.
            let (allocation,strides)={
                let descriptor=source.try_descriptor().unwrap();
                let strides=descriptor.completed_strides().unwrap();
                (descriptor.facts().allocation().expect("completed original source backing"),
                    [strides[0],strides[2],strides[1],strides[3]])
            };
            let descriptor=view.as_array().try_descriptor().unwrap();
            assert_eq!(descriptor.facts().allocation(),Some(allocation));
            assert_eq!(descriptor.completed_strides().unwrap(),&strides);
            if heads>1&&positions>1 {
                assert_ne!(descriptor.completed_strides().unwrap(),
                    &[(heads*positions*width) as i64,(positions*width) as i64,width as i64,1]);
            }
        }
        drop(scoped_input);
        assert_eq!(actual.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);drop(actual);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            assert!(std::time::Instant::now()<retirement_deadline,
                "rotary retirement deadline explicit={explicit} transposed={transposed} shape={shape:?}");
            safemlx::try_retire_completed_submissions().unwrap();
            MlxNeuralBackend::reclaim_retired_resources();safemlx::reclaim_allocation_owners();released.load(Ordering::SeqCst)
        });
        // Retained initial input/frequencies have independent ownership and must
        // not keep this completed invocation's actual quota alive.
        assert_eq!(frequency.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap(),frequencies);
        assert_eq!(source.evaluated().unwrap().try_to_vec::<f32>().unwrap(),data);
      }
    }
}

#[test]
fn cpu_input_products_yarn_uses_exact_flatten_stride_proof_and_integer_coordinates() {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    for algorithm in [RotaryAlgorithm::Default,RotaryAlgorithm::Linear{factor:4.0},spec(8).algorithm] {
        let mut selected_spec=spec(8);selected_spec.arithmetic=RotaryArithmetic::InputProducts;selected_spec.algorithm=algorithm;
        let mut dense_births=None;
        for transposed in [false,true] {
            let context=WorkspaceContext::new(cpu);
            let source_shape=if transposed {[2,5,3,8]}else{[2,3,5,8]};
            let input=WorkspaceTensor::existing(context.layout(&source_shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            let input=if transposed {input.transpose_axes(&[0,2,1,3],&context).unwrap()}else{input};
            let mut rope=WorkspaceBackend::rotary(selected_spec,&context).unwrap();context.begin_span();
            let output=rope.forward(&input,RotaryPosition::Offset(16_777_216),&context).unwrap();
            assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
            let report=context.finish_report(&[output]).unwrap();
            let operation=report.operations[0].as_view();let plan=cpu.plan(operation).unwrap().unwrap();
            assert_eq!(plan.seeds,1);assert_eq!(plan.population.hidden_leaves,1);
            if transposed {assert_eq!(Some(plan.population.births-1),dense_births);}else{dense_births=Some(plan.population.births);}
            let recipe=SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
            assert_eq!(recipe.storage.maximum_births(),plan.population.births+1);
            assert_eq!(recipe.completion.traversal.limits().arrays,plan.population.primitives+5);
            let unknown=[operation.inputs.get(0).unwrap().with_representation(Some(
                WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false).with_last_axis_contiguous(true)))];
            assert!(cpu.plan(WorkspaceOperationView{inputs:WorkspaceLayoutList::Views(&unknown),..operation}).unwrap().is_none());
            assert!(cpu.plan(WorkspaceOperationView{kind:WorkspaceOperationKindView::Rotary(selected_spec,Some(i32::MAX)),..operation}).unwrap().is_none());
        }
    }
}

#[test]
fn cpu_input_products_recipe_pays_transpose_alias_without_an_extra_backing() {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    let mut external=None;
    for internal in [false,true] {
        let context=WorkspaceContext::new(cpu);
        let input=WorkspaceTensor::existing(context.layout(&[2,5,3,8],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let mut selected_spec=spec(8);selected_spec.arithmetic=RotaryArithmetic::InputProducts;
        let mut rope=WorkspaceBackend::rotary(selected_spec,&context).unwrap();
        if internal {context.begin_span();}
        let input=input.transpose_axes(&[0,2,1,3],&context).unwrap();
        if !internal {context.begin_span();}
        let output=rope.forward(&input,RotaryPosition::Offset(16_777_216),&context).unwrap();
        let output_roots=if internal {2}else{1};
        let report=if internal {context.finish_report(&[output,input])}
            else{context.finish_report(&[output])}.unwrap();
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,output_roots,ordinary,cpu,&context).unwrap();
        if internal {
            assert_eq!(report.operations.len(),2);
            let alias=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(alias.alias_input,Some(0));
            assert_eq!(alias.population.births,0);
            assert_eq!(alias.population.primitives,1);
            assert_eq!(alias.seeds,0);
            let (primitives,births,arrays)=external.unwrap();
            assert_eq!(recipe.completion.graph.primitives(),primitives+1);
            assert_eq!(recipe.storage.maximum_births(),births);
            // Alias descriptor, input occurrence, and the independently
            // completed view root, without another Data allocation.
            assert_eq!(recipe.completion.traversal.limits().arrays,arrays+3);
            assert_eq!(recipe.completion.traversal.limits().roots,2);
        }else{
            assert_eq!(report.operations.len(),1);
            external=Some((recipe.completion.graph.primitives(),recipe.storage.maximum_births(),
                recipe.completion.traversal.limits().arrays));
        }
    }
}
