//! Stateful random-key ownership for MLX backend sessions.

use safemlx::{error::Result, ops::indexing::TryIndexOp, random, Array, Stream};

pub(crate) fn split_key_at(key: &Array, index: usize, stream: &Stream) -> Result<Array> {
    let count = index
        .checked_add(1)
        .and_then(|count| i32::try_from(count).ok())
        .ok_or_else(|| safemlx::error::Exception::custom("random subkey index exceeds i32"))?;
    let keys = safemlx::random::split_n(key, count, stream)?;
    keys.try_index_device((index as i32,), stream)
}

/// A backend-owned MLX PRNG key that advances through native key splitting.
#[derive(Debug, Clone)]
pub struct RandomState {
    state: Array,
}
#[derive(Debug,thiserror::Error)]
#[error("random key descriptor clone: {cause}")]
struct RandomCloneFailure {
    #[source]
    cause:safemlx::PreparedArrayCloneCause,
    funding:eredu_core::HostMetadataFunding,
}

impl RandomState {
    /// Creates reproducible state from a seed.
    pub fn with_seed(seed: u64) -> Result<Self> {
        Ok(Self {
            state: random::key(seed)?,
        })
    }

    /// Takes ownership of an existing native PRNG key.
    pub fn from_key(key: Array) -> Self {
        Self { state: key }
    }

    /// Exact immutable-key clone census. Advancing a random state replaces its
    /// key, so this shares the actual descriptor and never copies its data.
    pub(crate) fn host_clone_bytes()->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let parts=[safemlx::PreparedArrayClone::control_bytes()?,
            Array::inspection_clone_handle_bytes(),size_of::<Self>(),size_of::<&Self>(),
            size_of::<&eredu_core::HostMetadataFunding>(),size_of::<RandomCloneFailure>(),
            size_of::<Option<eredu_core::HostMetadataFunding>>(),
            size_of::<std::result::Result<Self,eredu_core::BackendFailure>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<RandomCloneFailure>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Same key sharing as ordinary Clone, with its actual handle constructor
    /// and controls paid before allocation. The caller retains funding after
    /// the returned key, including any extraction through into_key.
    pub(crate) fn clone_with_host_source(&self,funding:&eredu_core::HostMetadataFunding)
        ->std::result::Result<Self,eredu_core::BackendFailure> {
        let bytes=Self::host_clone_bytes().ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        let fail=|cause|eredu_core::BackendFailure::from_error(RandomCloneFailure{cause,funding:funding.clone()});
        let mut slot=safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(fail)?;
        let state=slot.fill_for_inspection(&self.state).map_err(fail)?;
        Ok(Self { state })
    }
    /// Advances the state and returns the next key.
    pub fn next_key(&mut self, stream: &Stream) -> Result<Array> {
        let next = random::split_n(&self.state, 2, stream)?;
        // Tuple indexing selects a static Slice/Reshape row, matching the
        // shared workspace trace. Scalar indexing instead creates an index
        // tensor and Gather, with different storage and graph constructors.
        self.state = next.try_index_device((0,), stream)?;
        next.try_index_device((1,), stream)
    }

    /// Advances the actual sequential key and returns one F32 draw in [0, 1).
    /// Scalar construction stays fallible inside an admitted original domain.
    pub(crate) fn uniform_unit_interval(&mut self, stream: &Stream) -> Result<Array> {
        let key = self.next_key(stream)?;
        let low = Array::try_from_scalar(0.0f32)?;
        let high = Array::try_from_scalar(1.0f32)?;
        random::uniform::<_, f32>(low, high, &[1], &key, stream)
    }

    /// Replaces the current state from a seed.
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.state = random::key(seed)?;
        Ok(())
    }

    /// Move the current key without cloning its native handle.
    pub fn into_key(self) -> Array { self.state }

    /// Borrows the current native key.
    pub fn as_array(&self) -> &Array {
        &self.state
    }

    /// Mutably borrows the current native key.
    pub fn as_array_mut(&mut self) -> &mut Array {
        &mut self.state
    }
}

/// Actual backend key/state/result controls of the shared Standard worker.
pub(crate) fn standard_sampling_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<RandomState>(),
        size_of::<Option<RandomState>>(),
        size_of::<[Array; 3]>(),
        size_of::<Result<Array>>(),
        size_of::<Option<&mut RandomState>>(),
        size_of::<&Stream>(),
        size_of::<f32>(),
        size_of::<u64>(),
    ]
    .into_iter()
    .try_fold(
        safemlx::random::standard_sampling_control_bytes()?,
        usize::checked_add,
    )
}


/// Actual fixed Rust frame for split(position+1)-then-select. The split's
/// native graph/population and the static index have separate real receipts.
pub(crate) fn split_key_at_control_bytes() -> Option<usize> {
    use std::mem::{size_of,size_of_val};
    let parts=[size_of::<&Array>(),size_of::<usize>(),size_of::<Option<i32>>(),
        size_of::<i32>(),size_of::<&Stream>(),size_of::<Array>(),size_of::<Result<Array>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

/// Fixed Rust caller frame for the sequential uniform draw. The split and
/// uniform native call controls are priced separately by their actual receipts.
pub(crate) fn uniform_unit_interval_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [size_of::<&mut RandomState>(), size_of::<&Stream>(),
        size_of::<[Array; 3]>(), size_of::<Result<Array>>(), size_of::<[f32; 2]>(),
        size_of::<[i32; 1]>()];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
