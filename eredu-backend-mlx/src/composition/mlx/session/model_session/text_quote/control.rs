//! Lexical installation of the same finite admitted control source.
use super::*;
impl TextExecutionQuote {
    /// The actual all-rank control exchange uses this fresh admitted request's
    /// communicator owner. The saved decoder never retains a control cursor.
    /// The projection is weak and its activation closes on every exit/unwind.
    pub(in crate::composition::mlx::session) fn with_parallel_control<T, F>(
        &self, runtime: &mut ModelRuntime<MlxBackend<'_>>, run: F,
    ) -> Result<T, Error>
    where F: FnOnce(&mut ModelRuntime<MlxBackend<'_>>) -> Result<T, Error> {
        let Some(owner) = self.parallel_control.as_ref() else {
            return run(runtime);
        };
        let funding = self.planning_metadata.as_ref().ok_or_else(|| unknown())?;
        funding.reserve_metadata(std::mem::size_of::<(
            F, T, Result<T, Error>, Result<(), Error>,
            Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection>,
            crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlInstallation,
        )>()).map_err(Error::WorkspacePlanning)?;
        let (installation, projection) = owner.install()?;
        runtime.session().payload.model.erased().install_parallel_control(Some(projection))?;
        let result = run(runtime);
        let cleared = runtime.session().payload.model.erased().install_parallel_control(None);
        drop(installation);
        match (result, cleared) {
            (Err(cause), _) => Err(cause),
            (Ok(_), Err(cause)) => Err(cause),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

}
