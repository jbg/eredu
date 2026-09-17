//! Direct borrowed physical-field/topology bridge; no native metadata queries.
use super::*;

struct Check<'t> {
    topology: &'t dyn NativeParameterTopology,
    count: usize,
    failure: Option<ParameterSourceError>,
}
impl<'a> NativeParameterSourceVisitor<'a> for Check<'_> {
    fn parameter(&mut self, local: &'static str, _: &'a Array, _: bool) {
        if !self.topology.contains_key(local) && self.failure.is_none() {
            self.failure = Some(ParameterSourceError::TopologyMismatch { slot: self.count });
        }
        match self.count.checked_add(1) {
            Some(value) => self.count = value,
            None => {
                self.failure
                    .get_or_insert(ParameterSourceError::CountOverflow);
            }
        }
    }
    fn retained(&mut self, _: &'a Array) {}
}

struct Occurrences<'k> {
    key: &'k str,
    count: usize,
}
impl<'a> NativeParameterSourceVisitor<'a> for Occurrences<'_> {
    fn parameter(&mut self, local: &'static str, _: &'a Array, _: bool) {
        if local == self.key {
            self.count = self.count.saturating_add(1);
        }
    }
    fn retained(&mut self, _: &'a Array) {}
}

struct SourceEmit<'t, 'v, V> {
    topology: &'t dyn NativeParameterTopology,
    visitor: &'v mut V,
}
impl<'a, V: ParameterSourceVisitor<'a, MlxTensor>> NativeParameterSourceVisitor<'a>
    for SourceEmit<'a, '_, V>
{
    fn parameter(&mut self, local: &'static str, value: &'a Array, trainable: bool) {
        let spec = self
            .topology
            .get(local)
            .expect("immutable validated physical topology");
        self.visitor.parameter(
            ParameterMetadataView::from_spec(spec, trainable),
            MlxTensor::ref_cast(value),
        );
    }
    fn retained(&mut self, value: &'a Array) {
        self.visitor.retained(MlxTensor::ref_cast(value));
    }
}

struct MutableEmit<'t, 'v, V> {
    topology: &'t dyn NativeParameterTopology,
    visitor: &'v mut V,
}
impl<'a, V: ParameterVisitorMut<'a, MlxTensor>> NativeParameterSourceVisitorMut<'a>
    for MutableEmit<'_, '_, V>
{
    fn parameter(&mut self, local: &'static str, value: &'a mut Array, trainable: bool) {
        let Some(spec) = self.topology.get(local) else {
            self.visitor.borrowed_metadata_unavailable();
            return;
        };
        self.visitor.visit_mut_borrowed(
            ParameterMetadataView::from_spec(spec, trainable),
            MlxTensor::ref_cast_mut(value),
        );
    }
}

pub(in crate::backend::nn::shared) fn visit_module_parameter_sources<
    'a,
    M: NativeRetainedValues,
    V: ParameterSourceVisitor<'a, MlxTensor>,
>(
    module: &'a M,
    topology: &'a dyn NativeParameterTopology,
    visitor: &mut V,
) -> Result<(), ParameterSourceError> {
    let expected = module
        .native_parameter_source_count()
        .ok_or(ParameterSourceError::Unavailable)?;
    let mut check = Check {
        topology,
        count: 0,
        failure: None,
    };
    module.visit_native_parameter_sources(&mut check)?;
    if let Some(error) = check.failure {
        return Err(error);
    }
    if check.count != expected || check.count != topology.len() {
        return Err(ParameterSourceError::TopologyMismatch { slot: check.count });
    }
    // Validate every retained key exactly once before exposing any row. This
    // also rejects a repeated physical key hiding a missing topology member.
    for (slot, key) in topology.keys().enumerate() {
        let mut occurrences = Occurrences { key, count: 0 };
        module.visit_native_parameter_sources(&mut occurrences)?;
        if occurrences.count != 1 {
            return Err(ParameterSourceError::TopologyMismatch { slot });
        }
    }
    module.visit_native_parameter_sources(&mut SourceEmit { topology, visitor })
}

// Mutable fields are supplied by the same audited leaf declarations as the
// immutable source. Validate topology first; no maps or names are reconstructed.
pub(super) fn visit_module_parameters_mut_borrowed<'a, M, V>(
    module: &'a mut M,
    topology: &dyn NativeParameterTopology,
    visitor: &mut V,
) where
    M: NativeRetainedValues,
    V: ParameterVisitorMut<'a, MlxTensor>,
{
    struct Check;
    impl<'a> ParameterSourceVisitor<'a, MlxTensor> for Check {
        fn parameter(&mut self, _: ParameterMetadataView<'a>, _: &'a MlxTensor) {}
        fn retained(&mut self, _: &'a MlxTensor) {}
    }
    if visit_module_parameter_sources(&*module, topology, &mut Check).is_err() {
        visitor.borrowed_metadata_unavailable();
        return;
    }
    let mut emit = MutableEmit { topology, visitor };
    if module
        .visit_native_parameter_sources_mut(&mut emit)
        .is_err()
    {
        emit.visitor.borrowed_metadata_unavailable();
    }
}

// Layout-only visitor; real generic adapters have the identical borrowed
// pointer fields and never store V by value.
struct VisitLayout;
impl<'a> ParameterSourceVisitor<'a, MlxTensor> for VisitLayout {
    fn parameter(&mut self, _: ParameterMetadataView<'a>, _: &'a MlxTensor) {}
    fn retained(&mut self, _: &'a MlxTensor) {}
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for VisitLayout {
    fn visit_mut(&mut self, _: ParameterMetadata, _: &'a mut MlxTensor) {}
}
pub(super) fn binding_visit_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<Check<'static>>(),
        size_of::<super::topology::TopologyKeys<'static>>(),
        size_of::<Occurrences<'static>>(),
        size_of::<SourceEmit<'static, 'static, VisitLayout>>(),
        size_of::<MutableEmit<'static, 'static, VisitLayout>>(),
        size_of::<&VisitLayout>(),
        size_of::<&mut dyn NativeParameterSourceVisitor<'static>>(),
        size_of::<&mut dyn NativeParameterSourceVisitorMut<'static>>(),
        size_of::<ParameterMetadataView<'static>>(),
        size_of::<Option<&ParameterSpec>>(),
        size_of::<Result<(), ParameterSourceError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod tests;
