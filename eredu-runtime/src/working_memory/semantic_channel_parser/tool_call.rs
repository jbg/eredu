//! Paid call fields and escaped events; completion keeps full-schema authority separate.
use super::super::{OriginalJsonObject, OriginalJsonObjectError, OriginalSemanticChannelSource};
use eredu_core::{HostPreparationAuthority, SemanticEvent, SemanticText, SpeculativeBuffer};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_text::json_fragments::{
    GENERATED_CALL_ID_BYTES, JsonFieldError, JsonFieldRole, generated_call_id_control_bytes,
    write_generated_call_id,
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Fields(#[from] JsonFieldError),
    #[error(transparent)]
    Text(#[from] eredu_core::SemanticTextAllocationError),
    #[error("paid JSON call source is incomplete")]
    Source,
    #[error("paid JSON call frame extent overflow")]
    Overflow,
    #[error("paid JSON call event destination is full")]
    Capacity,
    #[error("paid JSON object worker failed")]
    Object,
}
#[derive(Debug)]
pub(in crate::working_memory) struct Call {
    object: Option<OriginalJsonObject>,
    failed: Option<OriginalJsonObjectError>,
    name: Option<SemanticText>,
    id: Option<SemanticText>,
    emitted: usize,
    index: usize,
    started: bool,
    complete: bool,
    source: OriginalSemanticChannelSource,
    funding: HostMetadataFunding,
}
#[derive(Debug)]
pub(in crate::working_memory) struct Failure {
    cause: Cause,
    prefix: Call,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.cause, &self.prefix.failed) {
            (Cause::Object, Some(failure)) => fmt::Display::fmt(failure, f),
            _ => fmt::Display::fmt(&self.cause, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match (&self.cause, &self.prefix.failed) {
            (Cause::Object, Some(failure)) => Some(failure),
            _ => Some(&self.cause),
        }
    }
}
/// The callback's concrete error can retain its own schema source/funding. The
/// call prefix independently retains all decoded fields and source identity.
#[derive(Debug, thiserror::Error)]
pub(in crate::working_memory) enum CompletionFailure<E: std::error::Error + 'static> {
    #[error(transparent)]
    Control(#[from] Failure),
    #[error("{cause}")]
    Schema {
        #[source]
        cause: E,
        prefix: Call,
    },
}
struct Identifier {
    bytes: [u8; GENERATED_CALL_ID_BYTES],
    used: usize,
}
impl fmt::Write for Identifier {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let end = self.used.checked_add(text.len()).ok_or(fmt::Error)?;
        let target = self.bytes.get_mut(self.used..end).ok_or(fmt::Error)?;
        target.copy_from_slice(text.as_bytes());
        self.used = end;
        Ok(())
    }
}
impl Call {
    fn controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Cause>(),
            size_of::<Failure>(),
            size_of::<Option<OriginalJsonObject>>(),
            size_of::<OriginalJsonObjectError>(),
            size_of::<Option<SemanticText>>(),
            size_of::<SemanticEvent>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<(Self, usize, bool), Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<Result<(), eredu_core::GenerationError>>(),
            size_of::<Identifier>(),
            size_of::<(&Self, &HostMetadataFunding)>(),
            size_of::<(&mut Self, &str, &mut SpeculativeBuffer<SemanticEvent>)>(),
            size_of::<eredu_text::semantic_channels::JsonToolProgram<'_>>(),
            size_of::<Option<&str>>(),
            size_of::<Option<&SemanticText>>(),
            size_of::<(usize, usize, bool, bool)>(),
            eredu_text::json_fragments::JsonFieldNames::control_bytes()?,
            OriginalSemanticChannelSource::control_bytes()?,
            size_of::<std::slice::Iter<'_, super::super::OriginalJsonField>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, super::super::OriginalJsonField>>>(
            ),
            size_of::<(&Self, &str)>(),
            size_of::<Result<&str, std::str::Utf8Error>>(),
            size_of::<fmt::Result>(),
            generated_call_id_control_bytes::<Identifier>()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn failure(self, cause: Cause) -> Failure {
        Failure {
            cause,
            prefix: self,
        }
    }
    pub(in crate::working_memory) fn prepare(
        source: &OriginalSemanticChannelSource,
        bytes: usize,
        index: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Failure> {
        let mut call = Self {
            object: None,
            failed: None,
            name: None,
            id: None,
            emitted: 0,
            index,
            started: false,
            complete: false,
            source: source.clone(),
            funding: funding.clone(),
        };
        let result = (|| {
            funding.reserve_metadata(Self::controls().ok_or(Cause::Overflow)?)?;
            call.source.json_tools().ok_or(Cause::Source)?;
            match OriginalJsonObject::prepare(bytes, funding) {
                Ok(object) => call.object = Some(object),
                Err(failure) => {
                    call.failed = Some(failure);
                    return Err(Cause::Object);
                }
            }
            Ok::<_, Cause>(())
        })();
        match result {
            Ok(()) => Ok(call),
            Err(cause) => Err(call.failure(cause)),
        }
    }
    fn retain_text(&self, text: &str) -> Result<SemanticText, Cause> {
        let bytes = SemanticText::retained_control_bytes(text.len())
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<SemanticText, eredu_core::SemanticTextAllocationError>,
                >())
            })
            .ok_or(Cause::Overflow)?;
        self.funding.reserve_metadata(bytes)?;
        Ok(SemanticText::try_copy_retained(
            text,
            HostPreparationAuthority::retain(self.funding.clone()),
        )?)
    }
    fn publish(&mut self, events: &mut SpeculativeBuffer<SemanticEvent>) -> Result<(), Cause> {
        let program = self.source.json_tools().ok_or(Cause::Source)?;
        let object = self.object.as_ref().ok_or(Cause::Source)?;
        for field in object.fields() {
            let Some(kind) = field.kind() else {
                continue;
            };
            match program.fields.inspect(
                field.key().as_str(),
                kind,
                field.string().map(SemanticText::as_str),
            )? {
                JsonFieldRole::Name => self.name = field.string().cloned(),
                JsonFieldRole::CallId => self.id = field.string().cloned(),
                JsonFieldRole::Arguments | JsonFieldRole::Other => (),
            }
        }
        if !self.started {
            let Some(name) = self.name.as_ref() else {
                return Ok(());
            };
            if program.fields.call_id.is_some() && self.id.is_none() {
                return Ok(());
            }
            let id = if let Some(id) = self.id.as_ref() {
                id.clone()
            } else {
                let mut id = Identifier {
                    bytes: [0; GENERATED_CALL_ID_BYTES],
                    used: 0,
                };
                write_generated_call_id(&mut id, self.index).map_err(|_| Cause::Source)?;
                self.retain_text(
                    std::str::from_utf8(&id.bytes[..id.used]).expect("generated UTF-8 ID"),
                )?
            };
            events
                .try_push(SemanticEvent::ToolCallStart {
                    index: self.index,
                    id,
                    name: name.clone(),
                })
                .map_err(|_| Cause::Capacity)?;
            self.started = true;
        }
        let arguments = match object.active_value() {
            Some((key, value)) if key.as_str() == program.fields.arguments => Some(value),
            _ => object
                .fields()
                .iter()
                .enumerate()
                .find_map(|(index, field)| {
                    (field.key().as_str() == program.fields.arguments)
                        .then(|| object.field_value(index))
                        .flatten()
                }),
        };
        if let Some(arguments) = arguments {
            if arguments.len() > self.emitted {
                let delta = self.retain_text(&arguments[self.emitted..])?;
                events
                    .try_push(SemanticEvent::ToolArgumentsDelta {
                        index: self.index,
                        json_fragment: delta,
                    })
                    .map_err(|_| Cause::Capacity)?;
                self.emitted = arguments.len();
            }
        }
        Ok(())
    }
    pub(in crate::working_memory) fn push(
        mut self,
        input: &str,
        events: &mut SpeculativeBuffer<SemanticEvent>,
    ) -> Result<(Self, usize, bool), Failure> {
        let result = (|| {
            self.funding
                .reserve_metadata(Self::controls().ok_or(Cause::Overflow)?)?;
            let program = self.source.json_tools().ok_or(Cause::Source)?;
            let object = self.object.take().ok_or(Cause::Source)?;
            let (object, used, complete) = match object.push_fields(input, program.fields) {
                Ok(result) => result,
                Err(failure) => {
                    self.failed = Some(failure);
                    return Err(Cause::Object);
                }
            };
            self.object = Some(object);
            self.publish(events)?;
            if complete {
                let object = self.object.take().ok_or(Cause::Source)?;
                self.object = Some(match object.finish_object() {
                    Ok(object) => object,
                    Err(failure) => {
                        self.failed = Some(failure);
                        return Err(Cause::Object);
                    }
                });
                let program = self.source.json_tools().ok_or(Cause::Source)?;
                let object = self.object.as_ref().ok_or(Cause::Source)?;
                let arguments = object.fields().iter().any(|field| {
                    field.key().as_str() == program.fields.arguments
                        && field.kind() == Some(super::super::OriginalJsonValueKind::Object)
                });
                program
                    .fields
                    .complete(self.name.is_some(), arguments, self.id.is_some())?;
                self.complete = true;
            }
            Ok::<_, Cause>((used, complete))
        })();
        match result {
            Ok((used, complete)) => Ok((self, used, complete)),
            Err(cause) => Err(self.failure(cause)),
        }
    }
    pub(in crate::working_memory) fn copy_bytes(&self) -> Option<usize> {
        Self::controls()?.checked_add(self.object.as_ref()?.copy_bytes()?)
    }
    pub(in crate::working_memory) fn try_copy(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Failure> {
        let bytes = self
            .copy_bytes()
            .ok_or_else(|| self.empty_copy(funding).failure(Cause::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| self.empty_copy(funding).failure(cause.into()))?;
        self.copy_prepaid(HostPreparationAuthority::retain(funding.clone()), funding)
    }
    fn empty_copy(&self, funding: &HostMetadataFunding) -> Self {
        Self {
            object: None,
            failed: None,
            name: self.name.clone(),
            id: self.id.clone(),
            emitted: self.emitted,
            index: self.index,
            started: self.started,
            complete: self.complete,
            source: self.source.clone(),
            funding: funding.clone(),
        }
    }
    pub(super) fn rebind_funding(&mut self, funding: &HostMetadataFunding) {
        self.funding = funding.clone();
        if let Some(object) = &mut self.object {
            object.rebind_funding(funding);
        }
    }
    pub(super) fn copy_prepaid(
        &self,
        host: HostPreparationAuthority,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Failure> {
        let mut copy = self.empty_copy(funding);
        let Some(object) = self.object.as_ref() else {
            return Err(copy.failure(Cause::Source));
        };
        match object.copy_prepaid(host, funding) {
            Ok(object) => {
                copy.object = Some(object);
                Ok(copy)
            }
            Err(failure) => {
                copy.failed = Some(failure);
                Err(copy.failure(Cause::Object))
            }
        }
    }
    /// The loan must be the original full-schema producer. This method only
    /// pays its enclosing callback/result controls; the producer pays its own
    /// exact input parse and validation scratch before using its retained source.
    pub(in crate::working_memory) fn complete_with<E, F>(
        self,
        validate: F,
        events: &mut SpeculativeBuffer<SemanticEvent>,
    ) -> Result<Self, CompletionFailure<E>>
    where
        E: std::error::Error + 'static,
        F: FnOnce(&str, &str, &HostMetadataFunding) -> Result<(), E>,
    {
        let parts = [
            Self::controls(),
            Some(size_of::<F>()),
            Some(size_of::<E>()),
            Some(size_of::<Result<(), E>>()),
            Some(size_of::<CompletionFailure<E>>()),
            Some(size_of::<Result<Self, CompletionFailure<E>>>()),
            Some(size_of::<(&str, &str, &HostMetadataFunding)>()),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), |sum, part| sum.checked_add(part?));
        let Some(controls) = controls else {
            return Err(self.failure(Cause::Overflow).into());
        };
        if let Err(cause) = self.funding.reserve_metadata(controls) {
            return Err(self.failure(cause.into()).into());
        }
        if !self.complete || !self.started {
            return Err(self.failure(Cause::Source).into());
        }
        let name = self.name.as_ref().expect("started name");
        let program = self.source.json_tools().expect("prepared tool source");
        let object = self.object.as_ref().expect("complete object");
        let arguments = object
            .fields()
            .iter()
            .enumerate()
            .find_map(|(index, field)| {
                (field.key().as_str() == program.fields.arguments)
                    .then(|| object.field_value(index))
                    .flatten()
            })
            .expect("complete arguments");
        if let Err(cause) = validate(name.as_str(), arguments, &self.funding) {
            return Err(CompletionFailure::Schema {
                cause,
                prefix: self,
            });
        }
        if events.try_push(SemanticEvent::ToolCallEnd).is_err() {
            return Err(self.failure(Cause::Capacity).into());
        }
        Ok(self)
    }
}
