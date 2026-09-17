//! Shared sparse receipt validation with explicitly supplied duplicate scratch.
use super::*;

/// One fixed duplicate-detection slot. Values describe row coordinates only;
/// they grant neither capture authority nor a source/ownership witness.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct RoutedUnitRowIdentity(Option<u64>, u64, u64);

/// Allocation-free failure from the ordinary sparse receipt validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RoutedUnitValidationError {
    #[error("invalid routed-unit invocation input shape")]
    InvocationShape,
    #[error("empty routed-unit geometry")]
    Empty,
    #[error("routed-unit geometry overflow")]
    Overflow,
    #[error("invalid routed-unit slice rank or extent")]
    SliceExtent,
    #[error("invalid routed-unit slice geometry")]
    SliceGeometry,
    #[error("routed-unit validation scratch is shorter than the receipt")]
    Scratch,
    #[error("invalid or duplicate routed-unit receipt")]
    Row,
    /// The actual row or bank differs from its retained producer placement.
    #[error("routed-unit receipt differs from producer ownership")]
    Ownership,
    #[error("routed-unit receipt differs from selection")]
    Selection,
    #[error("duplicate or incomplete routed-unit chunks")]
    Chunks,
    #[error("incomplete ordinary routed-unit capture")]
    Incomplete,
}
impl From<RoutedUnitValidationError> for CaptureError {
    fn from(cause: RoutedUnitValidationError) -> Self {
        match cause {
            RoutedUnitValidationError::Overflow => Self::Overflow,
            _ => Self::Invalid(cause.to_string()),
        }
    }
}
pub(super) fn validate_slice(
    geometry: RoutedUnitGeometry,
    slice: &ResolvedCaptureSlice,
) -> Result<(), RoutedUnitValidationError> {
    validate_axes(geometry, &slice.starts, &slice.ends, &slice.strides, &slice.shape)
}
pub(super) fn validate_axes(
    geometry: RoutedUnitGeometry, starts: &[u64], ends: &[u64],
    strides: &[u64], shape: &[u64],
) -> Result<(), RoutedUnitValidationError> {
    use RoutedUnitValidationError as E;
    if geometry.experts == 0 || geometry.units_per_expert == 0 || geometry.routes_per_token == 0 {
        return Err(E::Empty);
    }
    geometry
        .experts
        .checked_mul(geometry.units_per_expert)
        .ok_or(E::Overflow)?;
    if starts.len() != 3
        || ends.len() != 3
        || strides.len() != 3
        || shape.len() != 3
        || ends[1] > geometry.routes_per_token
        || ends[2] > geometry.units_per_expert
    {
        return Err(E::SliceExtent);
    }
    for axis in 0..3 {
        if strides[axis] == 0
            || starts[axis] > ends[axis]
            || shape[axis]
                != (ends[axis] - starts[axis]).div_ceil(strides[axis])
        {
            return Err(E::SliceGeometry);
        }
    }
    Ok(())
}
impl RoutedUnitCapture {
    /// Checks selected-unit geometry and route uniqueness. Ordinary callers
    /// provide the scratch allocation here; paid callers lend their destination.
    pub fn validate_rows(&self, slice: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        validate_slice(self.geometry, slice).map_err(CaptureError::from)?;
        let mut scratch = vec![RoutedUnitRowIdentity::default(); self.rows.len()];
        self.validate_rows_with_scratch(slice, &mut scratch)
            .map_err(Into::into)
    }

    /// Performs the ordinary checks using one caller-owned identity slot per
    /// row. Sorting is in place, requires no heap allocation and leaves receipt
    /// rows unchanged. A refusal retains all payloads with their caller.
    pub fn validate_rows_with_scratch(
        &self,
        slice: &ResolvedCaptureSlice,
        scratch: &mut [RoutedUnitRowIdentity],
    ) -> Result<(), RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        validate_slice(self.geometry, slice)?;
        let scratch = scratch.get_mut(..self.rows.len()).ok_or(E::Scratch)?;
        let units = usize::try_from(slice.shape[2]).map_err(|_| E::Overflow)?;
        for (row, key) in self.rows.iter().zip(scratch.iter_mut()) {
            if !row.coefficient.is_finite()
                || row.token < slice.starts[0]
                || row.token >= slice.ends[0]
                || !(row.token - slice.starts[0]).is_multiple_of(slice.strides[0])
                || row.expert >= self.geometry.experts
                || row.slot < slice.starts[1]
                || row.slot >= slice.ends[1]
                || !(row.slot - slice.starts[1]).is_multiple_of(slice.strides[1])
            {
                return Err(E::Row);
            }
            if row.unit_start != slice.starts[2]
                || row.unit_stride != slice.strides[2]
                || row.values.shape() != [units]
                || !matches!(row.values.data(), crate::TensorObservationData::F32(_))
            {
                return Err(E::Selection);
            }
            *key = RoutedUnitRowIdentity(row.source_peer, row.token, row.slot);
        }
        scratch.sort_unstable();
        if scratch.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(E::Row);
        }
        Ok(())
    }

    /// Validate the ordinary partition payload using prepaid duplicate scratch.
    /// Native chunk rows remain distinct from original token/route identities.
    /// Exact invocation completion is established separately by its source owner.
    pub fn validate_partition_with_scratch(
        &self, slice: &ResolvedCaptureSlice, ownership: &RoutedUnitCaptureOwnership,
        source_tokens: u64, scratch: &mut [RoutedUnitRowIdentity],
    ) -> Result<(), RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        ownership.validate_geometry(self.geometry)?;
        self.validate_rows_with_scratch(slice, scratch)?;
        if self.rows.len() as u64 > slice.shape[0].checked_mul(slice.shape[1]).ok_or(E::Overflow)?
            || self.rows.iter().any(|row| row.source_peer != ownership.source_peer
                || usize::try_from(row.expert).ok().and_then(|expert|
                    ownership.coordinates.experts().global_to_local(expert)).is_none()) {
            return Err(E::Ownership);
        }
        let bound = ownership.maximum_source_rows_checked(source_tokens, self.geometry.routes_per_token)?;
        let mut end = 0;
        for &[start, next] in &self.source_token_ranges {
            if start != end || next <= start || next > bound { return Err(E::Chunks); }
            end = next;
        }
        if !self.rows.is_empty()
            && (end == 0 || (ownership.source_peer.is_none() && end != source_tokens)) {
            return Err(E::Incomplete);
        }
        Ok(())
    }

    /// Finishes the ordinary invocation using the shared receipt validator.
    pub fn finish_ordinary(
        &mut self,
        slice: &ResolvedCaptureSlice,
        source_tokens: u64,
    ) -> Result<(), CaptureError> {
        validate_slice(self.geometry, slice).map_err(CaptureError::from)?;
        let mut scratch = vec![RoutedUnitRowIdentity::default(); self.rows.len()];
        self.finish_ordinary_with_scratch(slice, source_tokens, &mut scratch)
            .map_err(Into::into)
    }

    /// Validates all source chunks and selected routes using prepaid scratch,
    /// then orders unique rows without allocating stable-sort storage.
    /// Exchanged rows still need their separate distributed ownership proof.
    pub fn finish_ordinary_with_scratch(
        &mut self,
        slice: &ResolvedCaptureSlice,
        source_tokens: u64,
        scratch: &mut [RoutedUnitRowIdentity],
    ) -> Result<(), RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        self.validate_rows_with_scratch(slice, scratch)?;
        self.source_token_ranges.sort_unstable();
        let mut end = 0;
        for range in &self.source_token_ranges {
            if range[0] != end || range[1] <= range[0] || range[1] > source_tokens {
                return Err(E::Chunks);
            }
            end = range[1];
        }
        if self.rows.iter().any(|row| row.source_peer.is_some())
            || end != source_tokens
            || self.rows.len() as u64
                != slice.shape[0]
                    .checked_mul(slice.shape[1])
                    .ok_or(E::Overflow)?
        {
            return Err(E::Incomplete);
        }
        // Uniqueness was established above, so stable ordering carries no
        // additional semantics for these original token/slot coordinates.
        self.rows.sort_unstable_by_key(|row| (row.token, row.slot));
        Ok(())
    }
}

#[cfg(test)]
mod tests;

/// Typed sparse unit merger shared by ordinary and original-account assembly.
#[derive(Debug,Clone,Copy,PartialEq,Eq,thiserror::Error)]
pub enum RoutedUnitAssemblyError {
    #[error("sparse unit fragments disagree on expert or coefficient")] Identity,
    #[error("sparse receipt has non-floating or mismatched values")] Values,
    #[error("sparse unit destination exceeds selection")] Destination,
    #[error("sparse unit fragments overlap")] Overlap,
    #[error("sparse unit destination overflow")] Overflow,
}
impl RoutedUnitCaptureRow {
    /// Copy exactly one participating route's selected columns into caller-paid
    /// scalar/coverage buffers. Coefficient identity compares original bits.
    pub fn merge_partition_values(&self,expert:u64,coefficient:f32,destination:&ResolvedCaptureSlice,
        output:&mut [f32],seen:&mut [bool])->Result<u64,RoutedUnitAssemblyError> {
        use RoutedUnitAssemblyError as E;
        if self.expert!=expert||self.coefficient.to_bits()!=coefficient.to_bits(){return Err(E::Identity);}
        let crate::TensorObservationData::F32(values)=self.values.data() else{return Err(E::Values);};
        if destination.starts.len()!=3||destination.strides.len()!=3||destination.shape.len()!=3
            ||destination.strides[2]==0||output.len()!=seen.len(){return Err(E::Destination);}
        if values.len() as u64!=destination.shape[2]{return Err(E::Values);}
        let mut copied=0u64;
        for (index,&value) in values.iter().enumerate(){
            let at=(index as u64).checked_mul(destination.strides[2]).and_then(|n|n.checked_add(destination.starts[2]))
                .and_then(|n|usize::try_from(n).ok()).ok_or(E::Overflow)?;
            let flag=seen.get_mut(at).ok_or(E::Destination)?;
            if std::mem::replace(flag,true){return Err(E::Overlap);}
            output[at]=value;copied+=1;
        }
        Ok(copied)
    }
}
impl RoutedUnitCapture {
    /// Final distributed rows retain source ranges in contribution provenance,
    /// not in the assembled payload. Original peer identity remains on each row.
    pub fn finish_partition_with_scratch(&mut self,slice:&ResolvedCaptureSlice,scratch:&mut [RoutedUnitRowIdentity])
        ->Result<(),RoutedUnitValidationError> {
        use RoutedUnitValidationError as E;
        self.validate_rows_with_scratch(slice,scratch)?;
        let expected=if slice.shape.contains(&0){0}else{slice.shape[0].checked_mul(slice.shape[1]).ok_or(E::Overflow)?};
        if !self.source_token_ranges.is_empty()||self.rows.len() as u64!=expected{return Err(E::Incomplete);}
        self.rows.sort_unstable_by_key(|row|(row.source_peer,row.token,row.slot));Ok(())
    }
}
