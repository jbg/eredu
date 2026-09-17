//! Owning transports for framework-created request metadata.
use super::SpeculativeRequestIdentity;
use crate::{
    GenerationError, HostPreparationAuthority, SpeculativeConfig, generation::SemanticEvent,
};
use std::{alloc::Layout, mem::size_of, ops::Deref};

#[derive(Debug, thiserror::Error)]
enum CopyCause {
    #[error("{0}")]
    Configuration(#[source] GenerationError),
    #[error("speculative EOS allocation failed: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("speculative EOS allocation capacity differs from its request")]
    Capacity,
}
/// An initial configuration-copy refusal retains its actual constructor payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct SpeculativeConfigurationError {
    #[source]
    cause: CopyCause,
    host: HostPreparationAuthority,
}

/// Immutable initial configuration whose managed EOS Vec never escapes its payer.
/// Ordinary inputs retain their previous allocation and ordinary consuming exit.
/// This is host custody only, not source identity or execution authority.
#[derive(Debug)]
pub struct SpeculativeConfiguration {
    config: SpeculativeConfig,
    retained: bool,
    source: Option<SpeculativeRequestIdentity>,
    host: HostPreparationAuthority,
}
impl SpeculativeConfiguration {
    /// Actual fresh EOS destination plus constructor, refusal and handoff controls.
    /// The authority's own producer is priced separately by its caller.
    pub fn retained_control_bytes(eos_count: usize) -> Option<usize> {
        let parts = [
            Layout::array::<u32>(eos_count).ok()?.size(),
            size_of::<Self>(),
            size_of::<SpeculativeRequestIdentity>(),
            size_of::<SpeculativeConfig>(),
            size_of::<Vec<u32>>(),
            size_of::<CopyCause>(),
            size_of::<SpeculativeConfigurationError>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Result<Self, SpeculativeConfigurationError>>(),
            size_of::<Result<SpeculativeConfig, Self>>(),
            size_of::<(usize, usize, f32, &[u32])>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Validates with the existing config worker, then allocates/copies the exact
    /// borrowed EOS rows. This never attaches custody to an earlier owned Vec.
    pub fn try_copy_retained(
        max_tokens: usize,
        max_draft_tokens: usize,
        temperature: f32,
        eos: &[u32],
        host: HostPreparationAuthority,
        source: SpeculativeRequestIdentity,
    ) -> Result<Self, SpeculativeConfigurationError> {
        let mut config = SpeculativeConfig {
            max_tokens,
            max_draft_tokens,
            temperature,
            eos_token_ids: Vec::new(),
        };
        if let Err(cause) = config.validate() {
            return Err(SpeculativeConfigurationError {
                cause: CopyCause::Configuration(cause),
                host,
            });
        }
        if let Err(cause) = config.eos_token_ids.try_reserve_exact(eos.len()) {
            drop(config);
            return Err(SpeculativeConfigurationError {
                cause: CopyCause::Allocation(cause),
                host,
            });
        }
        if config.eos_token_ids.capacity() != eos.len() {
            drop(config);
            return Err(SpeculativeConfigurationError {
                cause: CopyCause::Capacity,
                host,
            });
        }
        config.eos_token_ids.extend_from_slice(eos);
        Ok(Self {
            config,
            retained: true,
            source: Some(source),
            host,
        })
    }
    /// Checks a producer-private preparation identity without exporting it.
    /// Only the selected producer can supply its authentic private identity.
    pub fn matches_preparation(&self, source: &SpeculativeRequestIdentity) -> bool {
        self.source
            .as_ref()
            .is_some_and(|retained| retained.same(source))
    }
    /// Preserves legacy consuming behavior only for an ordinary value.
    pub fn try_into_ordinary(self) -> Result<SpeculativeConfig, Self> {
        if self.retained {
            Err(self)
        } else {
            Ok(self.config)
        }
    }
}
impl From<SpeculativeConfig> for SpeculativeConfiguration {
    fn from(config: SpeculativeConfig) -> Self {
        Self {
            config,
            retained: false,
            source: None,
            host: HostPreparationAuthority::unmanaged(),
        }
    }
}
impl Deref for SpeculativeConfiguration {
    type Target = SpeculativeConfig;
    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

/// Event callback with its framework-created Box paid before allocation.
/// Caller payloads are moved; neither their earlier allocations nor callback
/// work are certified. No raw prepared Box can escape its paying host token.
pub struct SpeculativeEventCallback<'a> {
    callback: Box<dyn FnMut(SemanticEvent) + 'a>,
    source: Option<SpeculativeRequestIdentity>,
    host: HostPreparationAuthority,
}
impl std::fmt::Debug for SpeculativeEventCallback<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeEventCallback")
            .finish_non_exhaustive()
    }
}
impl<'a> SpeculativeEventCallback<'a> {
    /// Exact framework Box destination and call/handoff frames for this F.
    /// Prior allocations reachable from the caller-owned F remain caller-owned.
    pub fn retained_control_bytes<F: FnMut(SemanticEvent) + 'a>() -> Option<usize> {
        let parts = [
            Layout::new::<F>().size(),
            size_of::<Self>(),
            size_of::<SpeculativeRequestIdentity>(),
            size_of::<F>(),
            size_of::<Box<F>>(),
            size_of::<Box<dyn FnMut(SemanticEvent) + 'a>>(),
            size_of::<SemanticEvent>(),
            size_of::<Option<SemanticEvent>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Allocates this concrete callback Box after its exact admission.
    pub fn new_retained<F: FnMut(SemanticEvent) + 'a>(
        callback: F,
        host: HostPreparationAuthority,
        source: SpeculativeRequestIdentity,
    ) -> Self {
        Self {
            callback: Box::new(callback),
            source: Some(source),
            host,
        }
    }
    /// Checks a producer-private preparation identity without exposing an alias.
    pub fn matches_preparation(&self, source: &SpeculativeRequestIdentity) -> bool {
        self.source
            .as_ref()
            .is_some_and(|retained| retained.same(source))
    }
    pub(super) fn callback_mut(&mut self) -> &mut (dyn FnMut(SemanticEvent) + 'a) {
        &mut *self.callback
    }
}
impl<'a, F: FnMut(SemanticEvent) + 'a> From<Box<F>> for SpeculativeEventCallback<'a> {
    fn from(callback: Box<F>) -> Self {
        Self {
            callback,
            source: None,
            host: HostPreparationAuthority::unmanaged(),
        }
    }
}
impl<'a> From<Box<dyn FnMut(SemanticEvent) + 'a>> for SpeculativeEventCallback<'a> {
    fn from(callback: Box<dyn FnMut(SemanticEvent) + 'a>) -> Self {
        Self {
            callback,
            source: None,
            host: HostPreparationAuthority::unmanaged(),
        }
    }
}
