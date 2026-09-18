// `pub` only for the `__private` re-export consumed by generated code.
#![allow(clippy::must_use_candidate)]
use crate::validator::{
    workspace::{Component, Error},
    ValidationContext,
};
use serde_json::{
    allocation::{Allocation, AllocationError, Unenforced},
    bounded_events::{PlanError, ValueError},
};

pub(crate) type ContentMediaTypeCheckType = fn(&str) -> bool;

fn is_json_with_allocations(
    input: &str,
    allocation: &dyn Allocation,
) -> Result<Result<bool, PlanError>, AllocationError> {
    match serde_json::bounded_events::from_slice_with_allocations(input.as_bytes(), allocation) {
        Ok(_) => Ok(Ok(true)),
        Err(ValueError::Allocation(error)) => Err(error),
        Err(ValueError::Plan(error)) => Ok(Err(error)),
        Err(ValueError::Syntax(_) | ValueError::Empty) => Ok(Ok(false)),
    }
}
pub fn is_json(input: &str) -> bool {
    is_json_with_allocations(input, &Unenforced)
        .expect("ordinary JSON content allocation")
        .expect("ordinary JSON content source profile")
}
pub(crate) fn default_content_media_type(media_type: &str) -> Option<ContentMediaTypeCheckType> {
    match media_type {
        "application/json" => Some(is_json),
        _ => None,
    }
}
#[derive(Clone, Copy)]
pub(crate) enum ContentMediaTypeSource {
    Json,
    Custom(ContentMediaTypeCheckType),
}
impl ContentMediaTypeSource {
    pub(crate) fn qualify(self) -> Result<(), Error> {
        match self {
            Self::Json => Ok(()),
            Self::Custom(_) => Err(Error::Unqualified(Component::Source(
                "custom content media type",
            ))),
        }
    }
    pub(crate) fn check(self, input: &str, context: &mut ValidationContext) -> bool {
        match self {
            Self::Json => {
                match context
                    .workspace
                    .with_json_allocations(|allocation| is_json_with_allocations(input, allocation))
                {
                    Some(Ok(valid)) => valid,
                    Some(Err(error)) => context.workspace.refuse(match error {
                        PlanError::Overflow => Error::Overflow,
                        PlanError::Source => Error::Unqualified(Component::Source(
                            "JSON content parser source profile",
                        )),
                    }),
                    None => false,
                }
            }
            Self::Custom(check) => {
                if context.workspace.original() {
                    return context.workspace.refuse(self.qualify().unwrap_err());
                }
                check(input)
            }
        }
    }
}
