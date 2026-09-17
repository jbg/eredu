use super::*;

#[test]
fn gpu_eval_prologue_population_counts_root_edges_once_or_reports_unknown() {
    let layout = OperationEvent::gpu_eval_prologue_layout(3, 0);
    if std::env::var("EREDU_REQUIRE_GPU_EVAL_PROLOGUE_QUALIFICATION").as_deref() == Ok("1") {
        assert!(layout.is_some());
    }
    let Some(layout) = layout else {
        assert!(OperationEvent::gpu_eval_prologue_layout(0, 0).is_none());
        return;
    };
    assert_eq!(
        std::mem::size_of::<safemlx_sys::mlx_gpu_eval_prologue_layout>(),
        26 * std::mem::size_of::<usize>()
    );
    let population = layout.population(11, 15, 0).unwrap(); // 2 ops + 3 root edges
    assert_eq!(population.evaluations(), 11);
    assert_eq!(population.input_edges(), 15);
    assert_eq!(population.vector_allocations(), 11);
    let one = OperationEvent::gpu_eval_prologue_layout(1, 0).unwrap();
    assert_eq!(population.data_slot_bytes(), 15 * one.requests()[2].0);
    assert_eq!(
        population.fixed_payload_bytes(),
        11 * (layout.requests()[0].0 + layout.requests()[1].0)
    );
    assert!(layout.population(1, 4, 0).is_none());
    assert!(layout.population(usize::MAX, 0, 0).is_none());
    let zero = OperationEvent::gpu_eval_prologue_layout(0, 0).unwrap();
    assert_eq!(zero.blocks(), 3);
    assert_eq!(zero.population(1, 0, 0).unwrap().data_slot_bytes(), 0);
    assert!(OperationEvent::gpu_eval_prologue_layout(usize::MAX, 1).is_none());
    assert_eq!(population.output_array_slots(), 11);
    assert_eq!(population.tracer_array_slots(), 0);
    assert_eq!(population.invocation_vector_allocations(), 11);
    let traced = OperationEvent::gpu_eval_prologue_layout_with_tracing(3, 0, true).unwrap();
    assert_eq!(traced.blocks(), 5);
    let traced = traced.population(11, 15, 0).unwrap();
    assert_eq!(traced.output_array_slots(), 11);
    assert_eq!(traced.tracer_array_slots(), 15);
    assert_eq!(traced.invocation_vector_allocations(), 22);
    assert_eq!(traced.output_array_bytes(), 11 * one.requests()[3].0);
    assert_eq!(traced.tracer_array_bytes(), 15 * one.requests()[3].0);
    assert_eq!(zero.population(1, 0, 0).unwrap().output_array_slots(), 1);
    assert!(traced.maximum().population(usize::MAX, 0, 0).is_none());
}
