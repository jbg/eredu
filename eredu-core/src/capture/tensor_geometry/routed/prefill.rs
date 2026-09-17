//! Flattened batch/token coordinates for the existing canonical prefill schedule.
use super::*;
use crate::{InferenceGeometry, OutputDemand};
use std::ops::Range;

/// One source-bound sparse selection; no tensor, allocation or execution authority.
#[derive(Debug)]
pub struct CaptureRoutedPrefillPlan<'a> {
    geometry: CaptureRoutedUnitsGeometry<'a>,
    inference: InferenceGeometry,
}
impl<'a> CaptureRoutedPrefillPlan<'a> {
    /// Validate the actual ordinary prompt, sparse bank and original cached origin.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        index: usize,
        inference: InferenceGeometry,
    ) -> Result<Self, CapturePrefillGeometryError> {
        if source.invocation_bounds().is_some()
            || source.text_origin().map(|o| o.cached_positions) != Some(inference.cached_positions)
            || source.request().batch != inference.batch_size
            || source.request().prompt_tokens != inference.input_positions
            || inference.max_output_tokens > source.request().max_predictions
        {
            return Err(CapturePrefillGeometryError::RequestMismatch);
        }
        if inference.batch_size == 0
            || inference.input_positions == 0
            || inference.max_output_tokens == 0
            || inference.prefill_chunk_positions == 0
            || inference.prefill_chunk_positions > inference.input_positions
        {
            return Err(CapturePrefillGeometryError::Schedule);
        }
        inference
            .cached_positions
            .checked_add(inference.input_positions)
            .and_then(|v| v.checked_add(inference.max_output_tokens))
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let geometry =
            CaptureRoutedUnitsGeometry::prepare(source, index, CapturePhase::Prefill, 0, None)?;
        let axes = source.points()[index]
            .axes
            .as_deref()
            .ok_or(CapturePrefillGeometryError::RequestMismatch)?;
        if axes.len() != 3
            || axes[0].dimension != SymbolicDimension::TokenRows
            || !source.points()[index].prefill
            || (geometry.source_shape()[0] as u64)
                != inference
                    .batch_size
                    .checked_mul(inference.input_positions)
                    .ok_or(CapturePrefillGeometryError::Overflow)?
        {
            return Err(CapturePrefillGeometryError::RequestMismatch);
        }
        Ok(Self {
            geometry,
            inference,
        })
    }
    /// Full logical target, with shared checkpoint parameters kept sparse.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'a> {
        &self.geometry
    }
    /// Original candidate geometry; output demand is never widened here.
    pub fn inference_geometry(&self) -> InferenceGeometry {
        self.inference
    }
    /// Number of canonical nonempty prefill spans.
    pub fn chunk_count(&self) -> u64 {
        self.inference
            .input_positions
            .div_ceil(self.inference.prefill_chunk_positions)
    }
    /// Derive one physical span from the existing shared driver's candidate.
    pub fn fragment(
        &self,
        index: u64,
    ) -> Result<CaptureRoutedPrefillFragment<'_, 'a>, CapturePrefillGeometryError> {
        if index >= self.chunk_count() {
            return Err(CapturePrefillGeometryError::Chunk {
                index,
                chunks: self.chunk_count(),
            });
        }
        let start = index
            .checked_mul(self.inference.prefill_chunk_positions)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let end = start
            .checked_add(self.inference.prefill_chunk_positions)
            .ok_or(CapturePrefillGeometryError::Overflow)?
            .min(self.inference.input_positions);
        Ok(CaptureRoutedPrefillFragment {
            plan: self,
            index,
            input: start..end,
        })
    }
}
/// Fixed batch-aware token coordinates derived only from an admitted canonical
/// prefill fragment. This contains no source owner or execution authority.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct CaptureRoutedTokenWindow { batch:u64, prompt:u64, start:u64, end:u64 }
impl CaptureRoutedTokenWindow {
    /// Physical provider tokens before routing exchange, across this chunk's batch.
    pub fn source_tokens(&self)->u64 {self.batch*(self.end-self.start)}
    /// Translate the same ordinary flattened token without collapsing batch gaps.
    pub fn logical_token(&self,physical:u64)->Option<u64> {
        if physical>=self.source_tokens(){return None;}
        let width=self.end-self.start;
        Some((physical/width)*self.prompt+self.start+physical%width)
    }
}

/// Sparse provider rows use physical flattened tokens; receipt rows use logical tokens.
#[derive(Debug)]
pub struct CaptureRoutedPrefillFragment<'p, 'a> {
    plan: &'p CaptureRoutedPrefillPlan<'a>,
    index: u64,
    input: Range<u64>,
}
impl<'p, 'a> CaptureRoutedPrefillFragment<'p, 'a> {
    /// Exact original source and selection.
    pub fn plan(&self) -> &'p CaptureRoutedPrefillPlan<'a> {
        self.plan
    }
    /// Canonical span ordinal.
    pub fn chunk_index(&self) -> u64 {
        self.index
    }
    /// Source span within every batch member.
    pub fn input(&self) -> &Range<u64> {
        &self.input
    }
    /// Number of actual flattened provider tokens.
    pub fn source_tokens(&self) -> u64 {
        self.token_window().source_tokens()
    }
    /// Retain only the exact immutable coordinates needed by a short native
    /// writer; the same shared mapping remains authoritative for ordinary use.
    pub fn token_window(&self)->CaptureRoutedTokenWindow {
        CaptureRoutedTokenWindow {batch:self.plan.inference.batch_size,prompt:self.plan.inference.input_positions,
            start:self.input.start,end:self.input.end}
    }
    /// Compare the unchanged shared-driver chunk, including output demand.
    pub fn matches_chunk(&self, input: &Range<u64>, position: u64, output: OutputDemand) -> bool {
        input == &self.input
            && self
                .plan
                .inference
                .cached_positions
                .checked_add(self.input.start)
                == Some(position)
            && output
                == self
                    .plan
                    .inference
                    .output
                    .for_chunk(self.input.end == self.plan.inference.input_positions)
    }
    /// Translate one real physical token, preserving batch separation and prompt origin.
    pub fn logical_token(&self, physical: u64) -> Option<u64> {
        self.token_window().logical_token(physical)
    }
    /// Test selection against the original globally anchored token and route strides.
    pub fn selects(&self, physical: u64, slot: u64) -> bool {
        let Some(token) = self.logical_token(physical) else {
            return false;
        };
        let geometry = self.plan.geometry();
        [token, slot].into_iter().enumerate().all(|(axis, value)| {
            value >= geometry.starts()[axis]
                && value < geometry.ends()[axis]
                && (value - geometry.starts()[axis]) % geometry.strides()[axis] == 0
        })
    }
    /// Split a real physical provider span at batch boundaries without allocating.
    /// Each returned interval names only visited original prompt tokens.
    pub fn source_ranges(
        &self,
        start: u64,
        end: u64,
    ) -> Option<impl Iterator<Item = [u64; 2]> + '_> {
        if start >= end || end > self.source_tokens() {
            return None;
        }
        let width = self.input.end - self.input.start;
        Some((start / width..=(end - 1) / width).map(move |batch| {
            let lo = start.max(batch * width) - batch * width;
            let hi = end.min((batch + 1) * width) - batch * width;
            let base = batch * self.plan.inference.input_positions + self.input.start;
            [base + lo, base + hi]
        }))
    }
}
