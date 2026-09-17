//! One ordered selector over actual behavior and complete declared token sources.
use super::{gemma, ifm, inkling, muse, remaining};
use crate::api::request::{ChatTemplateRequest, profile::tagged};
use crate::runtime::chat::dialect::{
    DECLARATIVE_DIALECT, DialectParameters, FormatDialect, ProfileDeclaration,
};
#[derive(Clone, Copy, Debug)]
pub(crate) enum Selected {
    Ifm {
        behavior: ifm::Recognition,
        declaration: ProfileDeclaration,
    },
    Muse(muse::Recognition),
    Gemma(gemma::Recognition),
    Inkling(inkling::Recognition),
    Remaining {
        behavior: remaining::Recognition,
        declaration: ProfileDeclaration,
    },
    Unregistered,
}
pub(crate) trait Operations: ifm::RequestOperations + remaining::ProtocolOperations {
    fn history_failure(
        &self,
        cause: tagged::Failure,
        messages: &[serde_json::Value],
        profile: &'static str,
    ) -> Self::Error;
}
fn declared<O: Operations>(
    ops: &mut O,
    dialect: &'static dyn FormatDialect,
    parameters: DialectParameters,
) -> Result<Option<ProfileDeclaration>, O::Error> {
    let Ok(declaration) = dialect.profile_declaration(parameters) else {
        return Ok(None);
    };
    if !ops.structural(declaration.structural)? {
        return Ok(None);
    }
    Ok(Some(declaration))
}
pub(crate) fn select<O: Operations>(
    ops: &mut O,
    request: &ChatTemplateRequest,
) -> Result<Selected, O::Error> {
    if let Some(behavior) = ifm::recognize(ops, request)? {
        let spec = crate::runtime::chat::ifm::spec(
            behavior.format.spelling(),
            behavior.effort.spelling(),
            request.add_generation_prompt,
            behavior.disabled,
        )
        .expect("shared IFM controls were validated");
        if let Some(declaration) = declared(
            ops,
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(spec),
        )? {
            if behavior.format != ifm::Format::Json {
                tagged::validate(&request.messages, "</ifm|arg_value>")
                    .map_err(|cause| ops.history_failure(cause, &request.messages, "IFM"))?;
            }
            return Ok(Selected::Ifm {
                behavior,
                declaration,
            });
        }
    }
    if let Some(behavior) = muse::recognize(ops)? {
        return Ok(Selected::Muse(behavior));
    }
    if let Some(behavior) = gemma::recognize(ops)? {
        return Ok(Selected::Gemma(behavior));
    }
    if let Some(behavior) = inkling::recognize(ops)? {
        return Ok(Selected::Inkling(behavior));
    }
    if let Some(behavior) = remaining::recognize(ops)? {
        let (_, dialect, parameters) = behavior.kind.declaration();
        if let Some(declaration) = declared(ops, dialect, parameters)? {
            return Ok(Selected::Remaining {
                behavior,
                declaration,
            });
        }
    }
    Ok(Selected::Unregistered)
}
pub(crate) fn control_bytes<E>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Selected>(),
        size_of::<Result<Selected, E>>(),
        size_of::<Option<ProfileDeclaration>>(),
        size_of::<Result<Option<ProfileDeclaration>, E>>(),
        size_of::<(
            &ChatTemplateRequest,
            &'static dyn FormatDialect,
            DialectParameters,
        )>(),
        size_of::<Option<ifm::Recognition>>(),
        size_of::<Option<muse::Recognition>>(),
        size_of::<Option<gemma::Recognition>>(),
        size_of::<Option<inkling::Recognition>>(),
        size_of::<Option<remaining::Recognition>>(),
        ProfileDeclaration::control_bytes()?,
        tagged::control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
