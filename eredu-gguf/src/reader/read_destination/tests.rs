use super::*;
use std::{
    cell::RefCell,
    io::{self, Cursor},
    rc::Rc,
};

#[derive(Clone, Debug, PartialEq)]
enum Call {
    Seek(u64),
    Read(u64, usize),
}
struct TraceIo {
    input: Cursor<Vec<u8>>,
    calls: Rc<RefCell<Vec<Call>>>,
    failure: Option<u64>,
    seek_failure: Option<u64>,
}
impl Read for TraceIo {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let pos = self.input.position();
        self.calls.borrow_mut().push(Call::Read(pos, out.len()));
        if self.failure == Some(pos) {
            return Err(io::Error::other("injected payload read"));
        }
        let limit = self
            .failure
            .filter(|offset| *offset > pos)
            .map(|offset| usize::try_from(offset - pos).unwrap().min(out.len()))
            .unwrap_or(out.len());
        self.input.read(&mut out[..limit])
    }
}
impl Seek for TraceIo {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if let SeekFrom::Start(offset) = from {
            self.calls.borrow_mut().push(Call::Seek(offset));
            if self.seek_failure == Some(offset) {
                return Err(io::Error::other("injected payload seek"));
            }
        }
        self.input.seek(from)
    }
}
fn file(endian: Endian) -> Vec<u8> {
    let values = [1.5_f32, -2.5, 3.5, 4.5, -5.5, 6.5];
    let raw: Vec<u8> = values
        .into_iter()
        .flat_map(|x| match endian {
            Endian::Little => x.to_le_bytes(),
            Endian::Big => x.to_be_bytes(),
        })
        .collect();
    let mut bytes = Cursor::new(Vec::new());
    crate::Writer::new(crate::WriterOptions {
        version: 3,
        endian,
        alignment: 32,
    })
    .unwrap()
    .write(
        &mut bytes,
        &BTreeMap::new(),
        &[crate::TensorInput {
            name: "matrix.weight",
            dimensions: &[3, 2],
            ggml_type: GgmlType::F32,
            data: &raw,
        }],
    )
    .unwrap();
    bytes.into_inner()
}
fn reader(bytes: &[u8]) -> Reader<TraceIo> {
    let io = TraceIo {
        input: Cursor::new(bytes.to_vec()),
        calls: Rc::new(RefCell::new(Vec::new())),
        failure: None,
        seek_failure: None,
    };
    let reader = Reader::new(io).unwrap();
    reader.inner.calls.borrow_mut().clear();
    reader
}
fn values(t: &ConvertedTensor) -> Vec<f32> {
    let ConvertedTensor::Dense(t) = t else {
        panic!("expected dense")
    };
    t.data
        .chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
}

#[test]
fn raw_destinations_match_independent_old_reads_and_nonzero_span_order() {
    for endian in [Endian::Little, Endian::Big] {
        let bytes = file(endian);
        let mut old = reader(&bytes);
        let mut ordinary = reader(&bytes);
        let mut fixed = reader(&bytes);
        let descriptor = old.tensors()[0].clone();
        let mut raw = vec![99; descriptor.byte_len as usize];
        let ptr = raw.as_ptr();
        let expected = old.old_read_tensor(&descriptor).unwrap();
        assert_eq!(ordinary.read_tensor(&descriptor).unwrap(), expected);
        assert_eq!(
            fixed
                .read_tensor_with_raw_destination(&descriptor, &mut raw)
                .unwrap(),
            expected
        );
        assert_eq!(raw.as_ptr(), ptr);
        assert_eq!(values(&expected), [1.5, -2.5, 3.5, 4.5, -5.5, 6.5]);
        assert_eq!(*old.inner.calls.borrow(), *fixed.inner.calls.borrow());
        assert_eq!(*old.inner.calls.borrow(), *ordinary.inner.calls.borrow());
        for (selection, expected_values) in [
            (
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
                vec![4.5, -5.5, 6.5],
            ),
            (
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![1, 0, 1],
                },
                vec![4.5, -5.5, 6.5, 1.5, -2.5, 3.5, 4.5, -5.5, 6.5],
            ),
            (
                TensorSelection::Indices {
                    axis: 1,
                    indices: vec![2, 0],
                },
                vec![3.5, 1.5, 6.5, 4.5],
            ),
        ] {
            let plan = TensorSelectionPlan::new(&descriptor, selection).unwrap();
            let mut raw = vec![99; plan.encoded_byte_len() as usize];
            let mut old = reader(&bytes);
            let mut ordinary = reader(&bytes);
            let mut fixed = reader(&bytes);
            let expected = old.old_read_tensor_plan(&plan).unwrap();
            assert_eq!(ordinary.read_tensor_plan(&plan).unwrap(), expected);
            assert_eq!(
                fixed
                    .read_tensor_plan_with_raw_destination(&plan, &mut raw)
                    .unwrap(),
                expected
            );
            assert_eq!(values(&expected), expected_values);
            assert_eq!(*old.inner.calls.borrow(), *fixed.inner.calls.borrow());
            assert_eq!(*old.inner.calls.borrow(), *ordinary.inner.calls.borrow());
        }
        let plan =
            DenseTensorSpanPlan::new(&descriptor, DenseTensorSpan::new(1, vec![2, 2]).unwrap())
                .unwrap();
        let mut old = reader(&bytes);
        let mut ordinary = reader(&bytes);
        let mut fixed = reader(&bytes);
        let mut raw = vec![99; plan.encoded_byte_len() as usize];
        let expected = old.old_read_dense_tensor_span(&plan).unwrap();
        assert_eq!(ordinary.read_dense_tensor_span(&plan).unwrap(), expected);
        assert_eq!(
            fixed
                .read_dense_tensor_span_with_raw_destination(&plan, &mut raw)
                .unwrap(),
            expected
        );
        assert_eq!(values(&expected), [-2.5, 3.5, 4.5, -5.5]);
        assert_eq!(*old.inner.calls.borrow(), *fixed.inner.calls.borrow());
        assert_eq!(*old.inner.calls.borrow(), *ordinary.inner.calls.borrow());
    }
}

#[test]
fn exact_destination_refuses_short_and_long_without_payload_io_or_writes() {
    let bytes = file(Endian::Little);
    let mut r = reader(&bytes);
    let descriptor = r.tensors()[0].clone();
    let axis = TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
    )
    .unwrap();
    let span = DenseTensorSpanPlan::new(&descriptor, DenseTensorSpan::new(1, vec![2, 2]).unwrap())
        .unwrap();
    for route in 0..3 {
        let n = match route {
            0 => descriptor.byte_len,
            1 => axis.encoded_byte_len(),
            _ => span.encoded_byte_len(),
        } as usize;
        for size in [n - 1, n + 1] {
            let mut raw = vec![91; size];
            let result = match route {
                0 => r.read_tensor_with_raw_destination(&descriptor, &mut raw),
                1 => r.read_tensor_plan_with_raw_destination(&axis, &mut raw),
                _ => r.read_dense_tensor_span_with_raw_destination(&span, &mut raw),
            };
            assert!(
                matches!(result, Err(ReadDestinationError::Length { expected, actual }) if expected == n as u64 && actual == size)
            );
            assert_eq!(raw, vec![91; size]);
            assert!(r.inner.calls.borrow().is_empty());
        }
    }
    let mut zero = descriptor.clone();
    zero.dimensions = vec![0];
    zero.byte_len = 0;
    let expected = r.old_read_tensor(&zero).unwrap();
    assert_eq!(
        r.read_tensor_with_raw_destination(&zero, &mut []).unwrap(),
        expected
    );
}

#[test]
fn read_and_seek_failures_preserve_old_causes_and_partial_selected_contents() {
    let bytes = file(Endian::Little);
    let r = reader(&bytes);
    let descriptor = r.tensors()[0].clone();
    let plan = TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0],
        },
    )
    .unwrap();
    for offset in [descriptor.data_offset + 14, descriptor.data_offset + 2] {
        let mut old = reader(&bytes);
        let mut fixed = reader(&bytes);
        old.inner.failure = Some(offset);
        fixed.inner.failure = Some(offset);
        let mut raw = vec![91; 24];
        let ptr = raw.as_ptr();
        let expected = old.old_read_tensor_plan(&plan).unwrap_err();
        let actual = fixed
            .read_tensor_plan_with_raw_destination(&plan, &mut raw)
            .unwrap_err();
        let ReadDestinationError::Gguf(actual) = actual else {
            panic!("expected actual I/O")
        };
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(*old.inner.calls.borrow(), *fixed.inner.calls.borrow());
        assert_eq!(raw.as_ptr(), ptr);
        if offset == descriptor.data_offset + 2 {
            assert_eq!(
                &raw[..12],
                &bytes[descriptor.data_offset as usize + 12..descriptor.data_offset as usize + 24]
            );
            assert_eq!(
                &raw[12..14],
                &bytes[descriptor.data_offset as usize..descriptor.data_offset as usize + 2]
            );
            assert_eq!(&raw[14..], &[0; 10]);
        } else {
            assert_eq!(&raw[2..12], &[0; 10]);
            assert_eq!(&raw[12..], &[91; 12]);
        }
    }
    let mut old = reader(&bytes);
    let mut fixed = reader(&bytes);
    old.inner.seek_failure = Some(descriptor.data_offset);
    fixed.inner.seek_failure = Some(descriptor.data_offset);
    let mut raw = vec![91; 24];
    let expected = old.old_read_tensor(&descriptor).unwrap_err();
    let actual = fixed
        .read_tensor_with_raw_destination(&descriptor, &mut raw)
        .unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(raw, vec![91; 24]);
    assert_eq!(*old.inner.calls.borrow(), *fixed.inner.calls.borrow());
    old.limits.max_allocation_bytes = 0;
    fixed.limits.max_allocation_bytes = 0;
    let expected = old.old_read_tensor(&descriptor).unwrap_err();
    let actual = fixed
        .read_tensor_with_raw_destination(&descriptor, &mut raw)
        .unwrap_err();
    assert!(matches!(expected, Error::Limit { .. }));
    assert_eq!(actual.to_string(), expected.to_string());
}

impl<R: Read + Seek> Reader<R> {
    fn old_read_raw(&mut self, tensor: &TensorDescriptor) -> Result<Vec<u8>> {
        check_limit(
            "tensor allocation",
            tensor.byte_len,
            self.limits.max_allocation_bytes,
        )?;
        self.inner
            .seek(SeekFrom::Start(tensor.data_offset))
            .map_err(|source| Error::Io {
                offset: tensor.data_offset,
                source,
            })?;
        let len =
            usize::try_from(tensor.byte_len).map_err(|_| Error::Overflow("tensor allocation"))?;
        let mut data = vec![0; len];
        self.inner
            .read_exact(&mut data)
            .map_err(|source| Error::Io {
                offset: tensor.data_offset,
                source,
            })?;
        Ok(data)
    }

    fn old_read_tensor(&mut self, tensor: &TensorDescriptor) -> Result<ConvertedTensor> {
        let raw = self.old_read_raw(tensor)?;
        crate::convert::convert(tensor, &raw, self.endian)
    }

    /// Execute a validated metadata-only physical selection plan.
    fn old_read_tensor_plan(&mut self, plan: &TensorSelectionPlan) -> Result<ConvertedTensor> {
        check_limit(
            "tensor allocation",
            plan.encoded_byte_len(),
            self.limits.max_allocation_bytes,
        )?;
        let selected_len = usize::try_from(plan.encoded_byte_len())
            .map_err(|_| Error::Overflow("selected tensor allocation"))?;
        let mut raw = Vec::with_capacity(selected_len);
        for span in plan.encoded_spans() {
            let span_len = usize::try_from(span.byte_len())
                .map_err(|_| Error::Overflow("selection span allocation"))?;
            self.inner
                .seek(SeekFrom::Start(span.offset()))
                .map_err(|source| Error::Io {
                    offset: span.offset(),
                    source,
                })?;
            let start = raw.len();
            let end = start
                .checked_add(span_len)
                .ok_or(Error::Overflow("selected tensor allocation"))?;
            raw.resize(end, 0);
            self.inner
                .read_exact(&mut raw[start..end])
                .map_err(|source| Error::Io {
                    offset: span.offset(),
                    source,
                })?;
        }
        crate::convert::convert(plan.selected_descriptor(), &raw, self.endian)
    }

    /// Execute a validated block-aligned contiguous-span plan.
    fn old_read_dense_tensor_span(
        &mut self,
        plan: &DenseTensorSpanPlan,
    ) -> Result<ConvertedTensor> {
        check_limit(
            "tensor allocation",
            plan.encoded_byte_len(),
            self.limits.max_allocation_bytes,
        )?;
        let span = plan.encoded_span();
        self.inner
            .seek(SeekFrom::Start(span.offset()))
            .map_err(|source| Error::Io {
                offset: span.offset(),
                source,
            })?;
        let len = usize::try_from(span.byte_len())
            .map_err(|_| Error::Overflow("dense tensor span allocation"))?;
        let mut raw = vec![0; len];
        self.inner
            .read_exact(&mut raw)
            .map_err(|source| Error::Io {
                offset: span.offset(),
                source,
            })?;
        crate::convert::convert(plan.selected_descriptor(), &raw, self.endian)
    }
}
