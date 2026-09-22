use super::*;
struct Destination {
    values: [u64; 2],
    written: usize,
}
impl TransferDestination for Destination {
    type Error = CaptureTensorNativeError;
    fn validate(&self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn push_f32(&mut self, _: f32) -> Result<(), Self::Error> {
        Err(CaptureTensorNativeError::ClaimMismatch)
    }
    fn push_u64(&mut self, value: u64) -> Result<(), Self::Error> {
        let slot = self
            .values
            .get_mut(self.written)
            .ok_or(CaptureTensorNativeError::ShapeMismatch)?;
        *slot = value;
        self.written += 1;
        Ok(())
    }
}
#[test]
fn native_unsigned_preview_and_summary_keep_large_expert_ids_exact() {
    if !crate::tests::support::native_process::enter("unsigned-capture-source") {
        return;
    }
    let stream = stream();
    let values = [16_777_217u32, u32::MAX - 1, 7, 11];
    let source = Array::from_slice(&values, &[4]);
    source.evaluated().unwrap();
    let admitted = admitted_dtype(
        vec![SymbolicDimension::Known(4)],
        CaptureTransform::Preview { max_elements: 2 },
        vec![],
        ObservationDtype::Integer,
    );
    let geometry =
        CaptureTensorGeometry::prepare(&admitted, 0, CapturePhase::Prefill, 0, None).unwrap();
    let plan =
        Selection::with_conversion(&geometry, ConversionMode::actual(source.dtype())).unwrap();
    PreparedCaptureTensor::validate_borrowed_source(&source, &geometry).unwrap();
    let roots = RefCell::new(Vec::new());
    let mut destination = Destination {
        values: [0; 2],
        written: 0,
    };
    execute_selected(
        &source,
        &plan,
        &mut destination,
        &stream,
        &roots,
        CaptureCompletion::Ordinary,
    )
    .unwrap();
    assert_eq!(destination.written, 2);
    assert_eq!(destination.values, [16_777_217, 4_294_967_294]);
    assert!(roots
        .borrow()
        .iter()
        .all(|value| value.dtype() == Dtype::Uint32));
    let admitted = admitted_dtype(
        vec![SymbolicDimension::Known(4)],
        CaptureTransform::Summary,
        vec![CaptureSlice {
            axis: "axis0".into(),
            start: 0,
            end: 2,
            stride: 1,
        }],
        ObservationDtype::Integer,
    );
    let geometry =
        CaptureSummaryGeometry::prepare(&admitted, 0, CapturePhase::Prefill, 0, None).unwrap();
    let summary = PreparedCaptureSummary::from_geometry(&geometry).unwrap();
    let result = summary
        .execute(
            &source,
            &stream,
            CaptureCompletion::Ordinary,
            &roots,
            &mut |_| Ok(()),
        )
        .unwrap();
    assert_eq!(result.min, Some(16_777_217.0));
    assert_eq!(result.max, Some(4_294_967_294.0));
    assert_eq!(result.mean, Some((16_777_217.0 + 4_294_967_294.0) / 2.));
    assert!(roots
        .borrow()
        .iter()
        .all(|value| value.dtype() == Dtype::Uint32));
}
