//! Source-bound full JSON validation through the existing paid serde event parser.
use eredu_core::{HostPreparationAuthority, SemanticText, SemanticTextAllocationError};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use serde_json::bounded_events::{Event, Plan, PlanError, Sink};
use std::mem::{size_of, size_of_val};

pub use eredu_text::json_fragments::JsonValueKind as OriginalJsonValueKind;
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Syntax(#[from] serde_json::Error),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Text(#[from] SemanticTextAllocationError),
    #[error("original JSON field source has no root value")]
    Source,
    #[error("original JSON field control extent overflow")]
    Overflow,
}
/// A syntax, destination or funding refusal retaining the actual decoded prefix.
/// Syntax failure after a root string does not discard that string's custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalJsonValueError {
    #[source]
    cause: Cause,
    event_failure: Option<Cause>,
    partial: Option<SemanticText>,
    funding: HostMetadataFunding,
}
/// Validated root and exact borrowed input. An object certificate cannot be
/// detached from the JSON bytes consumed by the actual parser.
#[derive(Debug)]
pub struct OriginalJsonValue<'a> {
    input: &'a str,
    kind: OriginalJsonValueKind,
    text: Option<SemanticText>,
    funding: HostMetadataFunding,
}
#[derive(Debug)]
struct Probe {
    kind: Option<OriginalJsonValueKind>,
    text: Option<SemanticText>,
    failure: Option<Cause>,
    funding: HostMetadataFunding,
}
impl Probe {
    fn copy_text(&mut self, value: &str) -> Result<(), Cause> {
        let bytes = SemanticText::retained_control_bytes(value.len())
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .ok_or(Cause::Overflow)?;
        self.funding.reserve_metadata(bytes)?;
        self.text = Some(SemanticText::try_copy_retained(
            value,
            HostPreparationAuthority::retain(self.funding.clone()),
        )?);
        Ok(())
    }
}
impl Sink for Probe {
    fn stopped(&self) -> bool {
        self.failure.is_some()
    }
    fn event(&mut self, event: Event<'_>) {
        if self.kind.is_some() || self.failure.is_some() {
            return;
        }
        self.kind = Some(match event {
            Event::Object => OriginalJsonValueKind::Object,
            Event::Array => OriginalJsonValueKind::Array,
            Event::String(value) => {
                if let Err(cause) = self.copy_text(value) {
                    self.failure = Some(cause);
                }
                OriginalJsonValueKind::String
            }
            Event::I64(_) | Event::U64(_) | Event::F64(_) | Event::Number(_) => {
                OriginalJsonValueKind::Number
            }
            Event::Bool(_) => OriginalJsonValueKind::Bool,
            Event::Null => OriginalJsonValueKind::Null,
            Event::Key(_) | Event::EndObject | Event::EndArray => {
                self.failure = Some(Cause::Source);
                return;
            }
        });
    }
}
impl<'a> OriginalJsonValue<'a> {
    fn controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Probe>(),
            size_of::<OriginalJsonValueKind>(),
            size_of::<OriginalJsonValueError>(),
            size_of::<Cause>(),
            size_of::<Option<Cause>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Option<SemanticText>>(),
            size_of::<Option<OriginalJsonValueKind>>(),
            size_of::<Event<'_>>(),
            size_of::<Result<Self, OriginalJsonValueError>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), serde_json::Error>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<Result<SemanticText, SemanticTextAllocationError>>(),
            size_of::<Result<Plan<'_>, PlanError>>(),
            size_of::<Plan<'_>>(),
            size_of::<serde_json::bounded_events::Requirements>(),
            size_of::<Result<serde_json::bounded_events::Requirements, PlanError>>(),
            size_of::<(&str, &HostMetadataFunding)>(),
            size_of::<(&mut Probe, &str)>(),
            size_of::<HostPreparationAuthority>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Validates all input with the exact existing serde deserializer, reserving
    /// its concrete source-derived parser storage before the first callback.
    /// Root strings additionally pay their own immutable output destination.
    pub fn parse(
        input: &'a str,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalJsonValueError> {
        let retain = |cause| OriginalJsonValueError {
            cause,
            event_failure: None,
            partial: None,
            funding: funding.clone(),
        };
        funding
            .reserve_metadata(Self::controls().ok_or_else(|| retain(Cause::Overflow))?)
            .map_err(|cause| retain(cause.into()))?;
        let plan = Plan::prepare(input.as_bytes()).map_err(|cause| retain(cause.into()))?;
        let requirements = plan
            .requirements::<Probe>()
            .map_err(|cause| retain(cause.into()))?;
        funding
            .reserve_metadata(requirements.required_bytes())
            .map_err(|cause| retain(cause.into()))?;
        let mut probe = Probe {
            kind: None,
            text: None,
            failure: None,
            funding: funding.clone(),
        };
        let allocation = super::original_json_allocation::JsonAllocation::new(funding)
            .map_err(|cause| retain(cause.into()))?;
        let parsed = plan.parse(&mut probe, &allocation);
        if let Some(cause) = allocation.failure() {
            return Err(retain(cause.into()));
        }
        if let Some(cause) = probe.failure {
            return Err(OriginalJsonValueError {
                cause,
                event_failure: None,
                partial: probe.text,
                funding: probe.funding,
            });
        }
        if let Err(cause) = parsed {
            return Err(OriginalJsonValueError {
                cause: cause.into(),
                event_failure: None,
                partial: probe.text,
                funding: probe.funding,
            });
        }
        let Some(kind) = probe.kind else {
            return Err(retain(Cause::Source));
        };
        Ok(Self {
            input,
            kind,
            text: probe.text,
            funding: probe.funding,
        })
    }
    /// Exact bytes whose complete syntax was validated.
    pub fn source(&self) -> &'a str {
        self.input
    }
    /// Root shape from the actual parser.
    pub fn kind(&self) -> OriginalJsonValueKind {
        self.kind
    }
    /// Decoded root string, absent for every other JSON type.
    pub fn string(&self) -> Option<&str> {
        self.text.as_ref().map(SemanticText::as_str)
    }
    /// Transfers the same paid root-string owner to a semantic field or event.
    pub fn into_string(self) -> Option<SemanticText> {
        self.text
    }
}
