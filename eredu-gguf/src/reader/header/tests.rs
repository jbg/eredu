use super::*;
use crate::reader::*;
use std::{
    cell::RefCell,
    io::{self, Cursor, Read, Seek, SeekFrom},
    rc::Rc,
};
mod reference;

#[derive(Debug)]
struct Trace {
    cursor: Cursor<Vec<u8>>,
    log: Rc<RefCell<Vec<String>>>,
    fail_at: Option<u64>,
}
impl Trace {
    fn new(bytes: Vec<u8>) -> (Self, Rc<RefCell<Vec<String>>>) {
        let log = Rc::new(RefCell::new(Vec::new()));
        (
            Self {
                cursor: Cursor::new(bytes),
                log: log.clone(),
                fail_at: None,
            },
            log,
        )
    }
}
impl Read for Trace {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let pos = self.cursor.position();
        self.log
            .borrow_mut()
            .push(format!("read {pos} {}", out.len()));
        if self.fail_at.is_some_and(|at| pos >= at) {
            return Err(io::Error::other("scripted failure"));
        }
        let limit = self
            .fail_at
            .map_or(out.len(), |at| out.len().min((at - pos) as usize));
        self.cursor.read(&mut out[..limit])
    }
}
impl Seek for Trace {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.log.borrow_mut().push(format!("seek {pos:?}"));
        self.cursor.seek(pos)
    }
}
fn u32b(bytes: &mut Vec<u8>, x: u32, big: bool) {
    bytes.extend(if big {
        x.to_be_bytes()
    } else {
        x.to_le_bytes()
    });
}
fn u64b(bytes: &mut Vec<u8>, x: u64, big: bool) {
    bytes.extend(if big {
        x.to_be_bytes()
    } else {
        x.to_le_bytes()
    });
}
fn count(bytes: &mut Vec<u8>, x: u64, version: u32, big: bool) {
    if version == 1 {
        u32b(bytes, x as u32, big);
    } else {
        u64b(bytes, x, big);
    }
}
fn string(bytes: &mut Vec<u8>, s: &[u8], version: u32, big: bool) {
    count(bytes, s.len() as u64, version, big);
    bytes.extend(s);
}
fn fixture(version: u32, big: bool) -> (Vec<u8>, usize) {
    let mut b = Vec::from(if big { &b"FUGG"[..] } else { &b"GGUF"[..] });
    u32b(&mut b, version, big);
    count(&mut b, 1, version, big);
    count(&mut b, 2, version, big);
    string(&mut b, b"label", version, big);
    u32b(&mut b, 8, big);
    string(&mut b, b"old", version, big);
    string(&mut b, b"nested", version, big);
    u32b(&mut b, 9, big);
    u32b(&mut b, 9, big);
    count(&mut b, 2, version, big);
    u32b(&mut b, 8, big);
    count(&mut b, 2, version, big);
    string(&mut b, b"a", version, big);
    string(&mut b, b"bc", version, big);
    u32b(&mut b, 4, big);
    count(&mut b, 2, version, big);
    u32b(&mut b, 7, big);
    u32b(&mut b, 11, big);
    string(&mut b, b"weight", version, big);
    u32b(&mut b, 1, big);
    count(&mut b, 4, version, big);
    u32b(&mut b, 0, big);
    u64b(&mut b, 0, big);
    let end = b.len();
    b.resize((end + 31) & !31, 0);
    for f in [1.25f32, -2.5, 3.0, 7.0] {
        u32b(&mut b, f.to_bits(), big);
    }
    (b, end)
}
fn captured(bytes: Vec<u8>) -> Arc<PreparedHeader> {
    let (source, _) = Trace::new(bytes);
    Reader::with_header_policy(source, Limits::default(), HeaderPolicy::capture())
        .unwrap()
        .prepared_header
        .unwrap()
}

#[test]
fn prepared_headers_match_independent_parser_reads_and_nonzero_payloads() {
    for version in [1, 2, 3] {
        for big in [false, true] {
            let (bytes, end) = fixture(version, big);
            let (source, old_log) = Trace::new(bytes.clone());
            let mut old = reference::read(source, Limits::default()).unwrap();
            let (source, ordinary_log) = Trace::new(bytes.clone());
            let mut ordinary = Reader::with_limits(source, Limits::default()).unwrap();
            let (source, capture_log) = Trace::new(bytes.clone());
            let capture =
                Reader::with_header_policy(source, Limits::default(), HeaderPolicy::capture())
                    .unwrap();
            let header = capture.prepared_header.unwrap();
            assert_eq!(header.byte_len(), end);
            assert_eq!(header.image, &bytes[..end]);
            let (source, prepared_log) = Trace::new(bytes);
            let mut prepared = Reader::with_header_policy(
                source,
                Limits::default(),
                HeaderPolicy::compare(&header),
            )
            .unwrap();
            assert_eq!(*old_log.borrow(), *capture_log.borrow());
            assert_eq!(*old_log.borrow(), *prepared_log.borrow());
            assert_eq!(old.metadata(), prepared.metadata());
            assert_eq!(old.tensors(), prepared.tensors());
            let descriptor = old.tensors()[0].clone();
            let expected = old.read_tensor(&descriptor).unwrap();
            assert_eq!(ordinary.read_tensor(&descriptor).unwrap(), expected);
            assert_eq!(prepared.read_tensor(&descriptor).unwrap(), expected);
            assert_eq!(*old_log.borrow(), *ordinary_log.borrow());
            assert_eq!(*old_log.borrow(), *prepared_log.borrow());
            assert!(Arc::ptr_eq(
                prepared.prepared_header.as_ref().unwrap(),
                &header
            ));
            let steps = header.storage_steps();
            assert!(steps
                .iter()
                .any(|s| s.kind() == HeaderStorageKind::Array && s.elements() == 2));
            assert!(steps
                .iter()
                .filter(|s| s.kind() == HeaderStorageKind::MetadataEntries)
                .all(|s| s.observed_capacity().is_none()));
            let dims = steps
                .iter()
                .find(|s| s.kind() == HeaderStorageKind::Dimensions)
                .unwrap();
            assert_eq!(
                dims.requested_layout().unwrap(),
                Layout::array::<u64>(1).unwrap()
            );
            let ranges = steps.last().unwrap();
            assert_eq!(ranges.kind(), HeaderStorageKind::Ranges);
            assert!(ranges.requested_layout().is_none());
            assert!(header.retained_layouts().unwrap()[0].size() >= header.byte_len());
        }
    }
}

#[test]
fn prepared_header_changed_count_refuses_before_unplanned_string_read() {
    let (mut bytes, _) = fixture(3, false);
    let header = captured(bytes.clone());
    let length_offset = 24;
    bytes[length_offset..length_offset + 8].copy_from_slice(&4096u64.to_le_bytes());
    let (source, log) = Trace::new(bytes);
    let error =
        Reader::with_header_policy(source, Limits::default(), HeaderPolicy::compare(&header))
            .err()
            .unwrap();
    assert_eq!(error.prepared_header_change().unwrap().offset(), 24);
    assert_eq!(log.borrow().last().unwrap(), "read 24 8");
    assert!(!log.borrow().iter().any(|x| x.ends_with("4096")));
}

#[test]
fn prepared_header_partial_read_error_precedes_byte_comparison() {
    let (mut bytes, _) = fixture(3, false);
    let header = captured(bytes.clone());
    bytes[32] = b'X';
    let (mut source, actual_log) = Trace::new(bytes.clone());
    source.fail_at = Some(34);
    let error =
        Reader::with_header_policy(source, Limits::default(), HeaderPolicy::compare(&header))
            .err()
            .unwrap();
    let (mut source, old_log) = Trace::new(bytes);
    source.fail_at = Some(34);
    let old = reference::read(source, Limits::default()).err().unwrap();
    assert_eq!(error.to_string(), old.to_string());
    assert!(error.prepared_header_change().is_none());
    assert!(matches!(error, Error::Io { offset: 32, .. }));
    assert_eq!(*actual_log.borrow(), *old_log.borrow());
}

#[test]
fn prepared_header_identity_is_not_payload_or_file_extent_identity() {
    let (mut bytes, _) = fixture(3, false);
    let header = captured(bytes.clone());
    let data = bytes.len() - 16;
    bytes[data..data + 4].copy_from_slice(&19.0f32.to_le_bytes());
    let (source, _) = Trace::new(bytes.clone());
    let mut actual =
        Reader::with_header_policy(source, Limits::default(), HeaderPolicy::compare(&header))
            .unwrap();
    let (source, _) = Trace::new(bytes.clone());
    let mut expected = reference::read(source, Limits::default()).unwrap();
    let d = expected.tensors()[0].clone();
    assert_eq!(
        actual.read_tensor(&d).unwrap(),
        expected.read_tensor(&d).unwrap()
    );
    bytes.pop();
    let (source, log) = Trace::new(bytes.clone());
    let error =
        Reader::with_header_policy(source, Limits::default(), HeaderPolicy::compare(&header))
            .err()
            .unwrap();
    let (source, old_log) = Trace::new(bytes);
    let old = reference::read(source, Limits::default()).err().unwrap();
    assert_eq!(error.to_string(), old.to_string());
    assert!(error.prepared_header_change().is_none());
    assert_eq!(*log.borrow(), *old_log.borrow());
}

#[test]
fn ordinary_parser_malformed_prefixes_keep_old_errors_and_reads() {
    let (bytes, _) = fixture(3, false);
    let mut cases = Vec::new();
    for length in [0, 3, 9, 25, 34, 47, bytes.len() - 1] {
        cases.push(bytes[..length].to_vec());
    }
    let mut bad = bytes.clone();
    bad[32] = 0xff;
    cases.push(bad);
    let mut bad = bytes.clone();
    bad[37..41].copy_from_slice(&99u32.to_le_bytes());
    cases.push(bad);
    for bytes in cases {
        let (source, log) = Trace::new(bytes.clone());
        let actual = Reader::with_limits(source, Limits::default())
            .err()
            .unwrap();
        let (source, old_log) = Trace::new(bytes);
        let old = reference::read(source, Limits::default()).err().unwrap();
        assert_eq!(actual.to_string(), old.to_string());
        assert_eq!(*log.borrow(), *old_log.borrow());
    }
}

#[test]
fn prepared_header_sharded_direct_replacement_failure_keeps_the_old_reader() {
    use crate::{Checkpoint, MetadataValue, TensorInput, Writer};
    use std::{collections::BTreeMap, fs::File};
    let dir = tempfile::tempdir().unwrap();
    let paths = [
        dir.path().join("model-00001-of-00002.gguf"),
        dir.path().join("model-00002-of-00002.gguf"),
    ];
    let write = |i: usize, label: &str| {
        let m = BTreeMap::from([
            ("label".into(), MetadataValue::String(label.into())),
            ("split.no".into(), MetadataValue::Uint16(i as u16)),
            ("split.count".into(), MetadataValue::Uint16(2)),
            ("split.tensors.count".into(), MetadataValue::Uint64(2)),
        ]);
        Writer::default()
            .write(
                File::create(&paths[i]).unwrap(),
                &m,
                &[TensorInput {
                    name: if i == 0 { "a" } else { "b" },
                    dimensions: &[2],
                    ggml_type: GgmlType::F32,
                    data: &[1.25f32.to_le_bytes(), (-3.5f32).to_le_bytes()].concat(),
                }],
            )
            .unwrap();
    };
    write(0, "old");
    write(1, "old");
    let c = Checkpoint::open_with_prepared_headers(&paths[0]).unwrap();
    assert!(c.shards().iter().all(|s| s.prepared_header().is_some()));
    let mut m = c.into_materializer();
    let first = m.converted_tensor("a").unwrap();
    write(1, "new");
    assert!(m
        .converted_tensor("b")
        .err()
        .unwrap()
        .prepared_header_change()
        .is_some());
    assert_eq!(m.open_shard_path(), Some(paths[0].as_path()));
    assert_eq!(m.converted_tensor("a").unwrap(), first);
    write(1, "old");
    m.converted_tensor("b").unwrap();
    assert_eq!(m.open_shard_path(), Some(paths[1].as_path()));
    assert!(m.close_reader_without_path());
    // Iterator uses the same explicit source contract and terminates on its error.
    let c = Checkpoint::open_with_prepared_headers(&paths[0]).unwrap();
    write(1, "new");
    let mut iter = c.converted_tensors();
    assert!(iter.next().unwrap().is_ok());
    assert!(iter
        .next()
        .unwrap()
        .err()
        .unwrap()
        .prepared_header_change()
        .is_some());
    assert!(iter.next().is_none());
}

#[test]
fn prepared_header_shared_worker_keeps_duplicate_and_stable_overlap_precedence() {
    let prefix = |tensors: u64, metadata: u64| {
        let mut b = b"GGUF".to_vec();
        u32b(&mut b, 3, false);
        count(&mut b, tensors, 3, false);
        count(&mut b, metadata, 3, false);
        b
    };
    let descriptor = |b: &mut Vec<u8>, name: &[u8]| {
        string(b, name, 3, false);
        u32b(b, 1, false);
        count(b, 4, 3, false);
        u32b(b, 0, false);
        u64b(b, 0, false);
    };
    let mut cases = Vec::new();
    for boolean in [0, 2] {
        let mut b = prefix(0, 2);
        string(&mut b, b"same", 3, false);
        u32b(&mut b, 4, false);
        u32b(&mut b, 7, false);
        string(&mut b, b"same", 3, false);
        u32b(&mut b, 7, false);
        b.push(boolean);
        cases.push((
            b,
            if boolean == 0 {
                "duplicate GGUF metadata"
            } else {
                "invalid boolean"
            },
        ));
    }
    let mut b = prefix(2, 0);
    descriptor(&mut b, b"same");
    string(&mut b, b"same", 3, false);
    u32b(&mut b, u32::MAX, false);
    cases.push((b, "duplicate GGUF tensor"));
    let mut b = prefix(3, 0);
    for n in [b"z", b"a", b"m"] {
        descriptor(&mut b, n);
    }
    b.resize((b.len() + 31) & !31, 0);
    b.extend([0; 16]);
    cases.push((b, "data overlaps tensor \"z\""));
    for (bytes, pattern) in cases {
        let (source, old_log) = Trace::new(bytes.clone());
        let old = reference::read(source, Limits::default()).err().unwrap();
        assert!(old.to_string().contains(pattern), "{old}");
        for policy in [HeaderPolicy::Ordinary, HeaderPolicy::capture()] {
            let (source, log) = Trace::new(bytes.clone());
            let error = Reader::with_header_policy(source, Limits::default(), policy)
                .err()
                .unwrap();
            assert_eq!(error.to_string(), old.to_string());
            assert_eq!(*log.borrow(), *old_log.borrow());
        }
    }
}

mod reuse;
