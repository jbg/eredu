use super::*;
use std::{alloc::Layout, collections::TryReserveError, fmt, ops::Range};
#[derive(Debug)]
struct PartRecord {
    modality: InputModality,
    kind: InputPayloadKind,
    slots: Range<usize>,
    extents: Range<usize>,
}
#[derive(Debug)]
struct SlotRecord {
    key: Option<InputMetadataKey>,
    shape: Range<usize>,
    values: Range<usize>,
    kind: u8,
}
#[derive(Debug, Default)]
pub(crate) struct Storage {
    parts: Vec<PartRecord>,
    slots: Vec<SlotRecord>,
    shapes: Vec<usize>,
    extents: Vec<InputExtent>,
    u32s: Vec<u32>,
    i32s: Vec<i32>,
    f32s: Vec<f32>,
    bools: Vec<bool>,
}
pub(crate) fn allocation_bytes(counts: Counts) -> Result<usize, HostInputPlanError> {
    let c = counts.0;
    let layouts = [
        Layout::array::<PartRecord>(c[0]),
        Layout::array::<SlotRecord>(c[1]),
        Layout::array::<usize>(c[2]),
        Layout::array::<InputExtent>(c[3]),
        Layout::array::<u32>(c[4]),
        Layout::array::<i32>(c[5]),
        Layout::array::<f32>(c[6]),
        Layout::array::<bool>(c[7]),
    ];
    layouts.into_iter().try_fold(0usize, |sum, layout| {
        sum.checked_add(layout.map_err(|_| HostInputPlanError::Overflow)?.size())
            .ok_or(HostInputPlanError::Overflow)
    })
}
pub(crate) struct CompileFailure {
    pub(crate) buffer: usize,
    pub(crate) cause: TryReserveError,
    storage: Storage,
}
impl CompileFailure {
    pub(crate) fn retained_heap_bytes(&self) -> usize {
        self.storage.heap_bytes()
    }
}
impl fmt::Debug for CompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostInputCompileFailure")
            .field("buffer", &self.buffer)
            .field("cause", &self.cause)
            .field("retained_heap_bytes", &self.retained_heap_bytes())
            .finish()
    }
}
impl fmt::Display for CompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prepared host input buffer {}: {}",
            self.buffer, self.cause
        )
    }
}
impl std::error::Error for CompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl Storage {
    pub(crate) fn compile(plan: PreparedHostInputPlan<'_>) -> Result<Self, CompileFailure> {
        Self::compile_inner(plan, None)
    }
    #[cfg(test)]
    pub(crate) fn compile_failing(
        plan: PreparedHostInputPlan<'_>,
        at: usize,
    ) -> Result<Self, CompileFailure> {
        Self::compile_inner(plan, Some(at))
    }
    fn compile_inner(
        plan: PreparedHostInputPlan<'_>,
        fail: Option<usize>,
    ) -> Result<Self, CompileFailure> {
        let mut s = Self::default();
        // Each buffer issues one complete requested-capacity reservation. No
        // grow/shrink/boxed-slice conversion follows. `fail` is private test-only
        // input; production always passes None and no callback enters this loop.
        macro_rules! reserve {
            ($field:ident,$i:expr) => {
                let requested = if fail == Some($i) {
                    usize::MAX
                } else {
                    plan.counts.0[$i]
                };
                let result = s.$field.try_reserve_exact(requested);
                if let Err(cause) = result {
                    return Err(CompileFailure {
                        buffer: $i,
                        cause,
                        storage: s,
                    });
                }
            };
        }
        reserve!(parts, 0);
        reserve!(slots, 1);
        reserve!(shapes, 2);
        reserve!(extents, 3);
        reserve!(u32s, 4);
        reserve!(i32s, 5);
        reserve!(f32s, 6);
        reserve!(bools, 7);
        for part in plan.parts {
            let first = s.slots.len();
            s.push_slot(None, part.payload);
            for key in KEYS {
                if let Some((_, value)) = part.metadata.iter().find(|(actual, _)| *actual == key) {
                    s.push_slot(Some(key), *value);
                }
            }
            let extents = s.extents.len();
            s.extents.extend_from_slice(part.extents);
            s.parts.push(PartRecord {
                modality: part.modality,
                kind: part.kind,
                slots: first..s.slots.len(),
                extents: extents..s.extents.len(),
            });
        }
        debug_assert_eq!(
            [
                s.parts.len(),
                s.slots.len(),
                s.shapes.len(),
                s.extents.len(),
                s.u32s.len(),
                s.i32s.len(),
                s.f32s.len(),
                s.bools.len()
            ],
            plan.counts.0
        );
        Ok(s)
    }
    fn push_slot(&mut self, key: Option<InputMetadataKey>, value: HostTensorView<'_>) {
        let shape = self.shapes.len();
        self.shapes.extend_from_slice(value.shape);
        let (kind, values) = match value.values {
            HostTensorValues::U32(v) => {
                let start = self.u32s.len();
                self.u32s.extend_from_slice(v);
                (0, start..self.u32s.len())
            }
            HostTensorValues::I32(v) => {
                let start = self.i32s.len();
                self.i32s.extend_from_slice(v);
                (1, start..self.i32s.len())
            }
            HostTensorValues::F32(v) => {
                let start = self.f32s.len();
                self.f32s.extend_from_slice(v);
                (2, start..self.f32s.len())
            }
            HostTensorValues::Bool(v) => {
                let start = self.bools.len();
                self.bools.extend_from_slice(v);
                (3, start..self.bools.len())
            }
        };
        self.slots.push(SlotRecord {
            key,
            shape: shape..self.shapes.len(),
            values,
            kind,
        });
    }
    fn view(&self, slot: &SlotRecord) -> HostTensorView<'_> {
        let values = match slot.kind {
            0 => HostTensorValues::U32(&self.u32s[slot.values.clone()]),
            1 => HostTensorValues::I32(&self.i32s[slot.values.clone()]),
            2 => HostTensorValues::F32(&self.f32s[slot.values.clone()]),
            3 => HostTensorValues::Bool(&self.bools[slot.values.clone()]),
            _ => unreachable!("closed compiled slot"),
        };
        HostTensorView {
            shape: &self.shapes[slot.shape.clone()],
            values,
        }
    }
    pub(crate) fn parts(&self) -> impl ExactSizeIterator<Item = PreparedHostPart<'_>> + Clone {
        self.parts.iter().map(|record| PreparedHostPart {
            storage: self,
            record,
        })
    }
    pub(crate) fn part(&self, index: usize) -> Option<PreparedHostPart<'_>> {
        self.parts.get(index).map(|record| PreparedHostPart {
            storage: self,
            record,
        })
    }
    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }
    pub(crate) fn slot(&self, index: usize) -> Option<HostTensorView<'_>> {
        self.slots.get(index).map(|slot| self.view(slot))
    }
    fn heap_bytes(&self) -> usize {
        self.parts.capacity() * std::mem::size_of::<PartRecord>()
            + self.slots.capacity() * std::mem::size_of::<SlotRecord>()
            + self.shapes.capacity() * std::mem::size_of::<usize>()
            + self.extents.capacity() * std::mem::size_of::<InputExtent>()
            + 4 * (self.u32s.capacity() + self.i32s.capacity() + self.f32s.capacity())
            + self.bools.capacity() * std::mem::size_of::<bool>()
    }
}
/// Borrowed compiled part; the original owner remains necessary for every view.
#[derive(Debug)]
pub struct PreparedHostPart<'a> {
    storage: &'a Storage,
    record: &'a PartRecord,
}
impl<'a> PreparedHostPart<'a> {
    /// Borrow the actual source payload for the source lifetime, independent of
    /// this small temporary view's stack lifetime.
    pub fn payload_view(&self) -> HostTensorView<'a> {
        self.storage
            .view(&self.storage.slots[self.record.slots.start])
    }
    /// Actual stable slot ordinal and metadata view in the immutable source.
    pub fn metadata_view(&self, key: InputMetadataKey) -> Option<(usize, HostTensorView<'a>)> {
        (self.record.slots.start + 1..self.record.slots.end)
            .find(|index| self.storage.slots[*index].key == Some(key))
            .map(|index| (index, self.storage.view(&self.storage.slots[index])))
    }
}
impl HostInputPartView for PreparedHostPart<'_> {
    fn modality(&self) -> InputModality {
        self.record.modality
    }
    fn kind(&self) -> InputPayloadKind {
        self.record.kind
    }
    fn payload(&self) -> HostTensorView<'_> {
        self.storage
            .view(&self.storage.slots[self.record.slots.start])
    }
    fn metadata(&self) -> impl ExactSizeIterator<Item = (InputMetadataKey, HostTensorView<'_>)> {
        self.storage.slots[self.record.slots.start + 1..self.record.slots.end]
            .iter()
            .map(|slot| {
                (
                    slot.key.expect("closed metadata slot"),
                    self.storage.view(slot),
                )
            })
    }
    fn extents(&self) -> &[InputExtent] {
        &self.storage.extents[self.record.extents.clone()]
    }
}
