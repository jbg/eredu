//! Closed immutable unit declarations from the actual initial target constructor.
use eredu_nn::{Error, workspace::WorkspaceMetadataError};
#[derive(Debug)]
enum Specifications {
    V3(Vec<crate::deepseek::block::V3BlockSpec>),
    V4(Vec<crate::deepseek::block::V4BlockSpec>),
    Qwen(Vec<crate::qwen::hybrid::block::TargetBlockSpec>),
    Nemotron(Vec<crate::nemotron_h::block::TargetBlockSpec>),
}
#[derive(Clone, Debug)]
pub(crate) struct RetainedRoutedUnits(Option<std::sync::Arc<Specifications>>);
impl Drop for RetainedRoutedUnits {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(std::sync::Arc::into_inner(owner));
        }
    }
}
impl RetainedRoutedUnits {
    pub(crate) fn nemotron_source(specs: Vec<crate::nemotron_h::block::TargetBlockSpec>) -> Self {
        Self(Some(std::sync::Arc::new(Specifications::Nemotron(specs))))
    }
    pub(crate) fn nemotron(&self) -> Result<&[crate::nemotron_h::block::TargetBlockSpec], Error> {
        match self.0.as_deref() {
            Some(Specifications::Nemotron(specs)) => Ok(specs),
            _ => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }

    pub(crate) fn qwen_source(specs: Vec<crate::qwen::hybrid::block::TargetBlockSpec>) -> Self {
        Self(Some(std::sync::Arc::new(Specifications::Qwen(specs))))
    }
    pub(crate) fn qwen(&self) -> Result<&[crate::qwen::hybrid::block::TargetBlockSpec], Error> {
        match self.0.as_deref() {
            Some(Specifications::Qwen(specs)) => Ok(specs),
            _ => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }

    pub(crate) fn v3_source(specs: Vec<crate::deepseek::block::V3BlockSpec>) -> Self {
        Self(Some(std::sync::Arc::new(Specifications::V3(specs))))
    }
    pub(crate) fn v4_source(specs: Vec<crate::deepseek::block::V4BlockSpec>) -> Self {
        Self(Some(std::sync::Arc::new(Specifications::V4(specs))))
    }
    pub(crate) fn v3(&self) -> Result<&[crate::deepseek::block::V3BlockSpec], Error> {
        match self.0.as_deref() {
            Some(Specifications::V3(specs)) => Ok(specs),
            _ => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }
    pub(crate) fn v4(&self) -> Result<&[crate::deepseek::block::V4BlockSpec], Error> {
        match self.0.as_deref() {
            Some(Specifications::V4(specs)) => Ok(specs),
            _ => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }
}
