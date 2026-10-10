//! Private capacity backing with immutable, independently advanceable prefixes.
use crate::{
    error::{Exception, Result},
    ops,
    utils::guard::Guarded,
    Array, Dtype, Stream,
};
use std::{cell::Cell, rc::Rc};

#[derive(Debug)]
struct Geometry {
    shape: Vec<i32>,
    dtype: Dtype,
    bytes: usize,
}

/// An append-only native buffer. Clones share a published immutable prefix.
/// A suffix is reserved once before any native work; divergent, truncated or
/// discarded descendants copy their visible prefix into independent capacity.
/// Physical spare storage is private. No safe caller can obtain its array.
/// Native nodes are one-use and reject graph tracing/replay. Ordinary completion
/// events retain their inputs/backing, including after the buffer is dropped.
#[derive(Debug, Clone)]
pub struct PrefixAppendBuffer {
    backing: Array,
    geometry: Rc<Geometry>,
    frontier: Rc<Cell<i32>>,
    length: i32,
    axis: usize,
}
impl PrefixAppendBuffer {
    /// Creates private capacity whose initial values remain immutable.
    /// Spare capacity allocates independent storage; a full buffer may share
    /// the input because it contains no writable suffix.
    pub fn new(initial: &Array, capacity: i32, axis: i32, stream: &Stream) -> Result<Self> {
        let shape = initial.shape();
        let axis = if axis < 0 {
            shape.len() as i32 + axis
        } else {
            axis
        };
        if axis < 0 || axis >= shape.len() as i32 || capacity < shape[axis as usize] {
            return Err(Exception::custom("invalid prefix append capacity/axis"));
        }
        let length = shape[axis as usize];
        let dtype = initial.dtype();
        let mut padding = shape.to_vec();
        padding[axis as usize] = capacity - length;
        // MLX copy may share immutable storage. A full buffer has no writable
        // suffix; any future growth allocates independent spare capacity.
        let backing = if capacity == length {
            ops::copy(initial, stream)?
        } else {
            let zeros = ops::zeros_dtype(&padding, dtype, stream)?;
            ops::concatenate_axis(&[initial, &zeros], axis, stream)?
        };
        let geometry = Rc::new(Geometry {
            shape: backing.shape().to_vec(),
            dtype,
            bytes: backing.nbytes(),
        });
        Ok(Self {
            backing,
            geometry,
            frontier: Rc::new(Cell::new(length)),
            length,
            axis: axis as usize,
        })
    }
    /// Current visible extent along the append axis.
    pub fn length(&self) -> i32 {
        self.length
    }
    /// Physical reserved extent along the append axis.
    pub fn capacity(&self) -> i32 {
        self.geometry.shape[self.axis]
    }
    /// Complete private logical backing payload, including spare capacity.
    pub fn capacity_bytes(&self) -> usize {
        self.geometry.bytes
    }
    /// Unpublished payload beyond the immutable prefix, without native queries.
    pub fn spare_capacity_bytes(&self) -> usize {
        if self.capacity() == 0 {
            0
        } else {
            self.geometry.bytes / self.capacity() as usize
                * (self.capacity() - self.length) as usize
        }
    }
    /// Whether this append can reserve unused slots in the current allocation.
    pub fn can_append(&self, count: i32) -> bool {
        count >= 0
            && self.length == self.frontier.get()
            && self
                .length
                .checked_add(count)
                .is_some_and(|end| end <= self.capacity())
    }
    /// Whether the native implementation reuses existing storage for this append.
    /// CUDA retains an immutable-copy fallback even when capacity is sufficient.
    pub fn can_append_without_copy(&self, count: i32) -> bool {
        cfg!(not(feature = "cuda")) && self.can_append(count)
    }
    /// Immutable logical view; spare capacity never escapes through this API.
    pub fn prefix(&self, stream: &Stream) -> Result<Array> {
        self.view(self.length, stream)
    }
    fn view(&self, length: i32, stream: &Stream) -> Result<Array> {
        let starts = vec![0; self.geometry.shape.len()];
        let mut ends = self.geometry.shape.clone();
        ends[self.axis] = length;
        let strides = vec![1; self.geometry.shape.len()];
        // SAFETY: validated geometry and initialized vectors live for the FFI call.
        Array::try_from_op(|out| unsafe {
            safemlx_sys::mlx_slice(
                out,
                self.backing.as_ptr(),
                starts.as_ptr(),
                starts.len(),
                ends.as_ptr(),
                ends.len(),
                strides.as_ptr(),
                strides.len(),
                stream.as_ptr(),
            )
        })
    }
    /// Creates a candidate by writing only unreserved slots. Growth/divergence
    /// copies the visible prefix. `capacity` is the caller's admitted growth bound.
    pub fn appended(&self, update: &Array, capacity: i32, stream: &Stream) -> Result<Self> {
        let shape = update.shape();
        if shape.len() != self.geometry.shape.len()
            || update.dtype() != self.geometry.dtype
            || shape
                .iter()
                .zip(&self.geometry.shape)
                .enumerate()
                .any(|(i, (a, b))| i != self.axis && a != b)
        {
            return Err(Exception::custom("prefix append geometry/dtype differs"));
        }
        let count = shape[self.axis];
        let end = self
            .length
            .checked_add(count)
            .ok_or_else(|| Exception::custom("prefix append overflow"))?;
        if count == 0 {
            return Ok(self.clone());
        }
        if !self.can_append(count) {
            if capacity < end {
                return Err(Exception::custom(
                    "prefix append capacity below required extent",
                ));
            }
            return Self::new(&self.prefix(stream)?, capacity, self.axis as i32, stream)?
                .appended(update, capacity, stream);
        }
        #[cfg(feature = "cuda")]
        {
            // CUDA device-to-host access may copy complete allocation capacity.
            // Keep CUDA backing entirely immutable rather than racing that copy.
            let prefix = self.prefix(stream)?;
            let combined = ops::concatenate_axis(&[&prefix, update], self.axis as i32, stream)?;
            return Self::new(&combined, self.capacity(), self.axis as i32, stream);
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Never refund a reserved suffix, even if construction/submission fails
            // or the lazy node is discarded. Another descendant must use new backing.
            self.frontier.set(end);
            // SAFETY: capacity is private and this suffix is reserved exactly once.
            let backing = Array::try_from_op(|out| unsafe {
                safemlx_sys::_mlx_prefix_append(
                    out,
                    self.backing.as_ptr(),
                    update.as_ptr(),
                    self.axis as i32,
                    self.length,
                    stream.as_ptr(),
                )
            })?;
            Ok(Self {
                backing,
                geometry: self.geometry.clone(),
                frontier: self.frontier.clone(),
                length: end,
                axis: self.axis,
            })
        }
    }
    /// Produces an older immutable prefix. Its next append diverges and copies.
    pub fn truncated(&self, length: i32) -> Result<Self> {
        if length < 0 || length > self.length {
            return Err(Exception::custom("invalid prefix truncation"));
        }
        Ok(Self {
            length,
            ..self.clone()
        })
    }
}
