use super::*;
use std::{
    cell::RefCell,
    io::{self, Cursor},
    rc::Rc,
};

#[derive(Clone)]
struct Script {
    cursor: Cursor<Vec<u8>>,
    trace: Rc<RefCell<Vec<String>>>,
    interrupt: bool,
    fail_seek: Option<SeekFrom>,
}
impl Script {
    fn new() -> Self {
        Self {
            cursor: Cursor::new((0..24000).map(|i| ((i * 19 + 7) % 251) as u8).collect()),
            trace: Rc::default(),
            interrupt: true,
            fail_seek: None,
        }
    }
}
impl Read for Script {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.trace.borrow_mut().push(format!(
            "read {} at {}",
            bytes.len(),
            self.cursor.position()
        ));
        if std::mem::take(&mut self.interrupt) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        self.cursor.read(bytes)
    }
}
impl Seek for Script {
    fn seek(&mut self, target: SeekFrom) -> io::Result<u64> {
        self.trace.borrow_mut().push(format!("seek {target:?}"));
        if self.fail_seek == Some(target) {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        self.cursor.seek(target)
    }
}
fn outcome<T: std::fmt::Debug>(result: io::Result<T>) -> String {
    match result {
        Ok(value) => format!("{value:?}"),
        Err(error) => format!("{:?}: {error}", error.kind()),
    }
}
fn exercise(reader: &mut (impl Read + Seek)) -> Vec<String> {
    let mut trace = Vec::new();
    for length in [3, 20, 8193, 16000, 0] {
        let mut bytes = vec![0xaa; length];
        trace.push(outcome(reader.read_exact(&mut bytes)));
        trace.push(format!("{bytes:?}"));
        trace.push(outcome(reader.stream_position()));
    }
    for target in [
        SeekFrom::Start(4),
        SeekFrom::Current(9),
        SeekFrom::End(-33),
        SeekFrom::Start(u64::MAX),
        SeekFrom::Start(0),
    ] {
        trace.push(outcome(reader.seek(target)));
        let mut bytes = [0xab; 13];
        trace.push(outcome(reader.read(&mut bytes)));
        trace.push(format!("{bytes:?}"));
    }
    trace
}
#[test]
fn prepared_buffer_matches_real_std_read_seek_and_exact_error_behavior() {
    let mut a = Script::new();
    a.fail_seek = Some(SeekFrom::Start(u64::MAX));
    let mut b = a.clone();
    b.trace = Rc::default();
    let a_trace = a.trace.clone();
    let b_trace = b.trace.clone();
    let mut ordinary = BufReader::with_capacity(8192, a);
    let mut prepared = Buffered::new(b, ReaderBuffer::prepare().unwrap());
    assert_eq!(exercise(&mut prepared), exercise(&mut ordinary));
    assert_eq!(*b_trace.borrow(), *a_trace.borrow());
}
#[test]
fn prepared_buffer_preserves_failed_current_seek_and_underflow_discard_behavior() {
    for underflow in [false, true] {
        let mut a = Script::new();
        a.interrupt = false;
        a.fail_seek = Some(SeekFrom::Current(if underflow { i64::MIN } else { -8190 }));
        let mut b = a.clone();
        b.trace = Rc::default();
        let a_trace = a.trace.clone();
        let b_trace = b.trace.clone();
        let mut ordinary = BufReader::with_capacity(8192, a);
        let mut prepared = Buffered::new(b, ReaderBuffer::prepare().unwrap());
        let mut x = [0; 2];
        let mut y = [0; 2];
        ordinary.read_exact(&mut x).unwrap();
        prepared.read_exact(&mut y).unwrap();
        assert_eq!(x, y);
        let target = SeekFrom::Current(if underflow { i64::MIN } else { 0 });
        assert_eq!(
            outcome(prepared.seek(target)),
            outcome(ordinary.seek(target))
        );
        assert_eq!(
            outcome(prepared.read(&mut x)),
            outcome(ordinary.read(&mut y))
        );
        assert_eq!(x, y);
        assert_eq!(
            outcome(prepared.stream_position()),
            outcome(ordinary.stream_position())
        );
        assert_eq!(*b_trace.borrow(), *a_trace.borrow());
    }
}
#[test]
fn incomplete_direct_and_iterator_buffers_are_returned_without_opening() {
    let checkpoint =
        crate::Checkpoint::open(std::path::Path::new("absent-reader-buffer-fixture.gguf"));
    assert!(checkpoint.is_err());
    let buffer = ReaderBuffer {
        bytes: vec![1, 2, 3],
    };
    let pointer = buffer.address();
    let capacity = buffer.capacity();
    let failure = CatalogReader::open_with_buffer(
        std::path::Path::new("absent-reader-buffer-fixture.gguf"),
        Limits::default(),
        None,
        &mut [],
        buffer,
    )
    .err()
    .unwrap();
    assert!(matches!(failure.0, Error::PreparedReaderStorage { .. }));
    assert_eq!(failure.1.address(), pointer);
    assert_eq!(failure.1.capacity(), capacity);
    let buffer = ReaderBuffer::prepare().unwrap();
    let pointer = buffer.address();
    let failure = CatalogReader::open_with_buffer(
        std::path::Path::new("absent-reader-buffer-fixture.gguf"),
        Limits::default(),
        None,
        &mut [],
        buffer,
    )
    .err()
    .unwrap();
    assert!(matches!(failure.0, Error::Io { offset: 0, .. }));
    assert_eq!(failure.1.address(), pointer);
}
