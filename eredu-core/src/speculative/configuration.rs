//! Request-owned settings copied through the actual driver destination.
use super::{
    SpeculativeConfig, SpeculativeConfiguration, SpeculativeDriverError, SpeculativeExecutor,
    SpeculativeRequestGeometry, SpeculativeValues,
};

#[derive(Clone)]
pub(super) struct RequestConfiguration {
    pub(super) max_tokens: usize,
    pub(super) max_draft_tokens: usize,
    pub(super) temperature: f32,
    pub(super) eos_token_ids: SpeculativeValues<u32>,
    // The copied payload and shared row shell retire before request controls.
    _host: crate::HostPreparationAuthority,
}
impl RequestConfiguration {
    pub(super) fn prepare<E: SpeculativeExecutor>(
        source: impl Into<SpeculativeConfiguration>,
        executor: &E,
        context: E::Context<'_>,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
        let source = source.into();
        let parts = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<SpeculativeConfig>(),
            std::mem::size_of::<SpeculativeConfiguration>(),
            std::mem::size_of::<Result<Self, SpeculativeDriverError<E::Error>>>(),
            std::mem::size_of::<(crate::HostPreparationAuthority, Option<usize>)>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add);
        let host = executor.driver_host_metadata(controls, context)?;
        let max_tokens = source.max_tokens;
        let max_draft_tokens = source.max_draft_tokens;
        let temperature = source.temperature;
        let eos_token_ids = match source.try_into_ordinary() {
            Ok(source) if host.is_unmanaged() => source.eos_token_ids.into(),
            Ok(source) => SpeculativeValues::collect_with_metadata(
                source.eos_token_ids.into_iter(),
                executor,
                context,
                0,
            )?,
            Err(source) => SpeculativeValues::collect_with_metadata(
                source.eos_token_ids.iter().copied(),
                executor,
                context,
                0,
            )?,
        };
        Ok(Self {
            max_tokens,
            max_draft_tokens,
            temperature,
            eos_token_ids,
            _host: host,
        })
    }
    pub(super) fn geometry(&self, selected_capacity: usize) -> SpeculativeRequestGeometry {
        SpeculativeRequestGeometry::from_caps(
            self.max_tokens,
            self.max_draft_tokens,
            selected_capacity,
        )
    }
}
