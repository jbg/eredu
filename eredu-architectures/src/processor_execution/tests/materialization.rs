use super::*;

struct Admit;
impl ProcessorInputAdmission for Admit {
    fn validate<T>(
        &self,
        _: &PreparedModelInput<T>,
        _: &impl PreparedInputInspector<T>,
    ) -> Result<(), ProcessorInputAdmissionError> {
        Ok(())
    }
}

fn budget() -> ProcessorRequestBudget {
    ProcessorRequestBudget {
        decoded_input_bytes: 1 << 20,
        host_buffer_bytes: 1 << 20,
        output_tensor_bytes: 1 << 20,
        planning_items: 4096,
        decoder_positions: 4096,
    }
}

#[test]
fn ordinary_raw_preparation_needs_no_allocation_accounting_hook() {
    let request = MultimodalRequest::new(vec![
        MultimodalSegment::TokenIds(vec![7, 9]),
        MultimodalSegment::Media(Media::Image(image(32))),
    ])
    .unwrap()
    .tokenize::<Infallible>(|_| unreachable!())
    .unwrap();
    let processor = qwen_processor();
    let mut mechanisms = TestMechanisms::default();
    let plain = processor
        .prepare(&request, &mut mechanisms, &mut |_| {
            Ok::<_, Infallible>(vec![])
        })
        .unwrap();
    assert!(mechanisms.tensors.get() > 0);
    let prepare = |mechanisms: &mut TestMechanisms, budget| {
        processor.prepare_with_admission(
            &request,
            mechanisms,
            &mut |_| Ok::<_, Infallible>(vec![]),
            &Admit,
            budget,
        )
    };
    let prepared = prepare(&mut mechanisms, budget()).unwrap();
    assert_eq!(prepared.identity(), plain.identity());
    assert_eq!(prepared.resources().output_tensor_bytes, 16 + 384 + 12);
    for field in 0..5 {
        let mut denied = budget();
        match field {
            0 => denied.decoded_input_bytes = 0,
            1 => denied.host_buffer_bytes = 0,
            2 => denied.output_tensor_bytes = 0,
            3 => denied.planning_items = 0,
            _ => denied.decoder_positions = 0,
        }
        let before = mechanisms.tensors.get();
        assert!(matches!(
            prepare(&mut mechanisms, denied),
            Err(ProcessorExecutionError::Resources(
                ProcessorResourceError::Exhausted { .. }
            ))
        ));
        assert_eq!(mechanisms.tensors.get(), before);
    }
}
