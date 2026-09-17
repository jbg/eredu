//! One allocation-free controller branch/trigger selection for both producers.
use super::{ConstraintError, GenerationRuntimePlan, ToolChoice};
use std::mem::{size_of, size_of_val};
#[derive(Debug, Clone, Copy)]
enum Activation {
    Forbidden,
    Automatic,
}
impl Activation {
    fn name(self) -> &'static str {
        match self {
            Self::Forbidden => "forbidden",
            Self::Automatic => "automatic",
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Error {
    activation: Activation,
    empty: bool,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tool-call sampling requires {} activation trigger",
            self.activation.name(),
            if self.empty {
                "a non-empty"
            } else {
                "an exact"
            }
        )
    }
}
impl std::error::Error for Error {}
impl Error {
    pub(super) fn ordinary(self) -> ConstraintError {
        ConstraintError::new(self.to_string())
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) enum Selection<'a> {
    Active,
    Forbidden(&'a [u8]),
    Auto(&'a [u8]),
}
impl<'a> Selection<'a> {
    pub(super) fn from_plan(plan: &'a GenerationRuntimePlan) -> Result<Self, Error> {
        if !plan.has_tool_surface() || plan.tool_choice() == ToolChoice::Required {
            return Ok(Self::Active);
        }
        let activation = if plan.tool_choice() == ToolChoice::None {
            Activation::Forbidden
        } else {
            Activation::Automatic
        };
        let trigger = match activation {
            Activation::Forbidden => plan.tool_call_trigger(),
            Activation::Automatic => plan.auto_activation_trigger(),
        }
        .ok_or(Error {
            activation,
            empty: false,
        })?;
        if trigger.is_empty() {
            return Err(Error {
                activation,
                empty: true,
            });
        }
        Ok(match activation {
            Activation::Forbidden => Self::Forbidden(trigger.as_bytes()),
            Activation::Automatic => Self::Auto(trigger.as_bytes()),
        })
    }
    pub(super) fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Error>(),
            size_of::<Activation>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(&GenerationRuntimePlan, Option<&str>)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
