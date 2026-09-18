//! Cold parameter semantics and original validators for the shared tagged worker.
use super::*;
use eredu_text::semantic_channels::tagged::{TaggedValuePolicy, parse_tagged_value_policy};
use serde_json::Value;

pub(super) struct Tagged {
    fields: Vec<Field>,
    required: Vec<String>,
}
struct Field {
    name: String,
    policy: TaggedValuePolicy,
    validator: Option<Source>,
}
struct Allocation<'a>(&'a ParserAllocationFunding);
impl serde_json::allocation::Allocation for Allocation<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), serde_json::allocation::AllocationError> {
        self.0
            .reserve(bytes)
            .map_err(|_| serde_json::allocation::AllocationError::Refused)
    }
    fn is_enforced(&self) -> bool {
        self.0.is_enforced()
    }
}
impl Tagged {
    pub(super) fn compile(
        schema: &Value,
        authority: &HostPreparationAuthority,
        funding: &ParserAllocationFunding,
    ) -> Result<Self, CompilationCause> {
        let controls = [
            size_of::<Self>(),
            size_of::<Field>(),
            size_of::<Allocation<'_>>(),
            size_of::<Result<Self, CompilationCause>>(),
            size_of::<TaggedValuePolicy>(),
            size_of::<Result<TaggedValuePolicy, serde_json::allocation::AllocationError>>(),
            size_of::<crate::runtime::chat::dialect::TaggedProperties<'_>>(),
            size_of::<Option<Source>>(),
            size_of::<Result<Source, original::CompilationFailure>>(),
            size_of::<std::slice::Iter<'_, Value>>(),
            size_of::<(&Value, &HostPreparationAuthority, &ParserAllocationFunding)>(),
        ];
        funding.reserve(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(CompilationCause::Overflow)?,
        )?;
        let allocation = Allocation(funding);
        let mut result = Self {
            fields: Vec::new(),
            required: Vec::new(),
        };
        for field in crate::runtime::chat::dialect::tagged_properties(schema, funding)? {
            let (name, schema) = field?;
            let policy = TaggedValuePolicy::prepare(schema, &allocation).map_err(|error| {
                funding
                    .failure()
                    .map_or(CompilationCause::Json(error), CompilationCause::Funding)
            })?;
            let validator = match Source::compile(schema, authority, funding) {
                Ok(source) => Some(source),
                // The ordinary local-property validator defers unresolved/rooted
                // schema semantics to the complete original object validator.
                Err(error) if error.is_local_property_error() => None,
                Err(error) => return Err(CompilationCause::Parameter(error)),
            };
            funding.try_push(
                &mut result.fields,
                Field {
                    name: funding.try_copy_str(name)?,
                    policy,
                    validator,
                },
            )?;
        }
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for value in required {
                if let Some(name) = value.as_str() {
                    funding.try_push(&mut result.required, funding.try_copy_str(name)?)?;
                }
            }
        }
        result.fields.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }
    pub(super) fn bytes(&self) -> Result<usize, CompilationCause> {
        let mut bytes = Layout::array::<Field>(self.fields.capacity())
            .map_err(|_| CompilationCause::Overflow)?
            .size()
            .checked_add(
                Layout::array::<String>(self.required.capacity())
                    .map_err(|_| CompilationCause::Overflow)?
                    .size(),
            )
            .ok_or(CompilationCause::Overflow)?;
        for field in &self.fields {
            bytes = bytes
                .checked_add(field.name.capacity())
                .and_then(|n| n.checked_add(field.policy.type_name.capacity()))
                .ok_or(CompilationCause::Overflow)?;
            if let Some(source) = &field.validator {
                bytes = bytes
                    .checked_add(
                        source
                            .capacity_bytes()
                            .map_err(CompilationCause::Parameter)?,
                    )
                    .ok_or(CompilationCause::Overflow)?;
            }
        }
        for name in &self.required {
            bytes = bytes
                .checked_add(name.capacity())
                .ok_or(CompilationCause::Overflow)?;
        }
        Ok(bytes)
    }
    pub(super) fn missing(
        &self,
        parameters: &eredu_text::semantic_channels::tagged::TaggedParameters,
    ) -> bool {
        self.required.iter().any(|name| !parameters.contains(name))
    }
    pub(super) fn parse(
        &self,
        parameter: &str,
        declared: Option<&str>,
        raw: &str,
        allocations: &dyn serde_json::allocation::Allocation,
        funding: &HostMetadataFunding,
    ) -> Result<Value, ParseFailure> {
        let field = self
            .fields
            .binary_search_by(|field| field.name.as_str().cmp(parameter))
            .ok()
            .map(|index| &self.fields[index]);
        let any = TaggedValuePolicy::any();
        parse_tagged_value_policy(
            field.map_or(&any, |field| &field.policy),
            declared,
            raw,
            allocations,
            |value| -> Result<bool, ValueValidation> {
                match field.and_then(|field| field.validator.as_ref()) {
                    Some(source) => {
                        let text = value.to_string_with_allocations(allocations)?;
                        Ok(source.matches(&text, funding)?)
                    }
                    None => Ok(true),
                }
            },
        )
        .map_err(ParseFailure::Value)
    }
}
#[derive(Debug, thiserror::Error)]
pub(super) enum ValueValidation {
    #[error(transparent)]
    Serialization(#[from] serde_json::value::ValueWriteError),
    #[error(transparent)]
    Validation(#[from] original::Failure),
}
#[derive(Debug, thiserror::Error)]
pub(super) enum ParseFailure {
    #[error(transparent)]
    Value(eredu_text::semantic_channels::tagged::TaggedValueError<ValueValidation>),
}
