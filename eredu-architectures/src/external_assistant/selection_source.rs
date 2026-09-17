//! The exact loaded selection and capture have one immutable source owner.
use super::*;

#[derive(Debug)]
struct Selected { selected:eredu_runtime::SelectedSpeculativeRealization, capture:crate::composite_execution::ExternalPredictionCaptureRequest }
/// Retained declaration source created by the actual materialization handoff.
/// Cloning this handle copies no schema, path, parameter, state or native value.
#[derive(Debug,Clone)]
pub struct ExternalSelectionSource(Arc<Selected>);
impl ExternalSelectionSource {
    pub(super) fn new(selected:eredu_runtime::SelectedSpeculativeRealization,capture:crate::composite_execution::ExternalPredictionCaptureRequest)->Self{
        Self(Arc::new(Selected{selected,capture}))
    }
    /// Exact realization whose source was materialized.
    pub fn selected(&self)->&eredu_runtime::SelectedSpeculativeRealization{&self.0.selected}
    /// Exact architecture capture from the same preparation.
    pub fn capture(&self)->&crate::composite_execution::ExternalPredictionCaptureRequest{&self.0.capture}
}
pub(super) enum SelectedOwner { Ordinary(eredu_runtime::SelectedSpeculativeRealization), Source(ExternalSelectionSource) }
impl std::ops::Deref for SelectedOwner {
    type Target=eredu_runtime::SelectedSpeculativeRealization;
    fn deref(&self)->&Self::Target{match self{Self::Ordinary(v)=>v,Self::Source(v)=>v.selected()}}
}

/// Existing executors may own an ordinary declaration or borrow the exact one
/// already retained beside their selected assistant. Neither path clones it.
pub(crate) enum CaptureSource<'a,T>{Owned(T),Borrowed(&'a T)}
impl<T> std::ops::Deref for CaptureSource<'_,T>{type Target=T;fn deref(&self)->&T{match self{Self::Owned(v)=>v,Self::Borrowed(v)=>v}}}

#[derive(Clone)]
pub(super) enum InputIdentity { Ordinary(eredu_runtime::SpeculativeIdentity), Retained(eredu_core::SpeculativeValues<eredu_runtime::SpeculativeIdentity>) }
impl std::ops::Deref for InputIdentity {
    type Target=eredu_runtime::SpeculativeIdentity;
    fn deref(&self)->&Self::Target{match self{Self::Ordinary(v)=>v,Self::Retained(v)=>&v[0]}}
}
