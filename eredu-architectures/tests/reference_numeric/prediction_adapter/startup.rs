//! Actual architecture membership reaches the destination without an ordinary clone.
use super::*;

#[derive(Default)]
struct Destination {
    source_addresses: Vec<usize>,
    sequential_count: Option<usize>,
    refuse: bool,
}
impl PredictionStateStartupFactory<NumericBackend, CaptureOnlyMaterializer> for Destination {
    type Error = &'static str;
    type Prepared<T> = T;

    fn sequential(&mut self, count: usize) -> Result<Vec<NumericCompressedCache>, Self::Error> {
        self.sequential_count = Some(count);
        if self.refuse {
            return Err("destination capacity");
        }
        Ok((0..count)
            .map(|_| NumericCompressedCache::resident())
            .collect())
    }
    fn pooling(
        &mut self,
        source: &[NumericPoolingCache],
    ) -> Result<Vec<NumericPoolingCache>, Self::Error> {
        self.source_addresses
            .extend(source.iter().map(|slot| std::ptr::from_ref(slot).addr()));
        if self.refuse {
            return Err("destination capacity");
        }
        Ok(source.to_vec())
    }
    fn model(&mut self, _: &CaptureOnlyModelState) -> Result<CaptureOnlyModelState, Self::Error> {
        panic!("this source is a per-layer cache, not a model-state prototype")
    }
}

#[test]
fn prepared_startup_uses_actual_membership_and_preserves_nonzero_prototype_on_refusal() {
    let sequential = MaterializedDeepSeekV3Prediction::<NumericBackend, CaptureOnlyMaterializer> {
        units: vec![
            CaptureOnlyModule::new(),
            CaptureOnlyModule::new(),
            CaptureOnlyModule::new(),
        ],
    };
    let mut destination = Destination::default();
    assert_eq!(
        sequential
            .prepare_new_state(&mut destination)
            .unwrap()
            .len(),
        3
    );
    assert_eq!(destination.sequential_count, Some(3));
    assert!(destination.source_addresses.is_empty());

    let mut state = NumericPoolingCache::new(4, &[]);
    state
        .append_local(
            NumericTensor::new([1, 1, 2], vec![3.0, 7.0]),
            &NumericContext::default(),
        )
        .unwrap();
    let extension =
        MaterializedDeepSeekV4Prediction::<NumericBackend, CaptureOnlyMaterializer>::Sequential {
            units: vec![CaptureOnlyModule::new()],
            state: vec![state],
        };
    let MaterializedDeepSeekV4Prediction::Sequential { state, .. } = &extension else {
        unreachable!()
    };
    let source_address = std::ptr::from_ref(&state[0]).addr();
    let mut destination = Destination {
        refuse: true,
        ..Destination::default()
    };
    assert!(matches!(
        extension.prepare_new_state(&mut destination),
        Err("destination capacity")
    ));
    assert_eq!(destination.source_addresses, [source_address]);
    assert_eq!(state[0].local.as_ref().unwrap().data, [3.0, 7.0]);
    destination.refuse = false;
    let copied = extension.prepare_new_state(&mut destination).unwrap();
    assert_eq!(copied[0].local.as_ref().unwrap().data, [3.0, 7.0]);
    assert_eq!(state[0].local.as_ref().unwrap().data, [3.0, 7.0]);
    assert_ne!(std::ptr::from_ref(&copied[0]).addr(), source_address);
}

impl PredictionStateCopyFactory<NumericBackend, CaptureOnlyMaterializer> for Destination {
    type Error = &'static str;
    type Prepared<T> = T;
    fn sequential(&mut self, source: &[NumericCompressedCache]) -> Result<Vec<NumericCompressedCache>, Self::Error> {
        self.source_addresses.extend(source.iter().map(|slot| std::ptr::from_ref(slot).addr()));
        if self.refuse { Err("destination capacity") } else { Ok(source.to_vec()) }
    }
    fn pooling(&mut self, source: &[NumericPoolingCache]) -> Result<Vec<NumericPoolingCache>, Self::Error> {
        <Self as PredictionStateStartupFactory<NumericBackend, CaptureOnlyMaterializer>>::pooling(self, source)
    }
    fn model(&mut self, _: &CaptureOnlyModelState) -> Result<CaptureOnlyModelState, Self::Error> {
        panic!("this source is a per-layer cache, not a model-state prototype")
    }
}

#[test]
fn prepared_prediction_copy_uses_current_branch_after_prototype_retires() {
    let mut state = NumericPoolingCache::new(4, &[]);
    state.append_local(NumericTensor::new([1, 1, 2], vec![3.0, 7.0]), &NumericContext::default()).unwrap();
    let extension = MaterializedDeepSeekV4Prediction::<NumericBackend, CaptureOnlyMaterializer>::Sequential {
        units: vec![CaptureOnlyModule::new()], state: vec![state],
    };
    let mut destination = Destination::default();
    let mut branch = extension.prepare_new_state(&mut destination).unwrap();
    branch[0].local.as_mut().unwrap().data[0] = 91.0;
    drop(extension);
    let current_address = std::ptr::from_ref(&branch[0]).addr();
    destination.source_addresses.clear();
    destination.refuse = true;
    assert!(matches!(MaterializedDeepSeekV4Prediction::<NumericBackend, CaptureOnlyMaterializer>::prepare_copy_state(&branch, &mut destination), Err("destination capacity")));
    assert_eq!(destination.source_addresses, [current_address]);
    assert_eq!(branch[0].local.as_ref().unwrap().data, [91.0, 7.0]);
    destination.refuse = false;
    let copied = MaterializedDeepSeekV4Prediction::<NumericBackend, CaptureOnlyMaterializer>::prepare_copy_state(&branch, &mut destination).unwrap();
    drop(branch);
    assert_eq!(copied[0].local.as_ref().unwrap().data, [91.0, 7.0]);
}
