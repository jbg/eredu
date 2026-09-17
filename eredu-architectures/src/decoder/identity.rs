//! One semantic fingerprint producer with caller-owned metadata destinations.
use eredu_core::cache::{PromptCacheArchitectureFingerprint, PromptCacheError};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::fmt::{self, Display};

#[derive(Clone, Copy)]
pub(crate) struct Metadata<'a>(Option<&'a WorkspaceContext>);
impl<'a> Metadata<'a> {
    pub(crate) fn context(self) -> Option<&'a WorkspaceContext> {
        self.0
    }

    pub(crate) fn new(context: Option<&'a WorkspaceContext>) -> Self {
        Self(context.filter(|context| context.uses_checked_metadata()))
    }
    pub(crate) fn controls<T>(self) -> Result<(), Error> {
        if let Some(context) = self.0 {
            let controls = [
                std::mem::size_of::<T>(),
                std::mem::size_of::<Result<T, Error>>(),
                std::mem::size_of::<Self>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?;
            context.charge_metadata(bytes)?;
        }
        Ok(())
    }
    pub(crate) fn borrowed_controls<T: ?Sized>(self, value: &T) -> Result<(), Error> {
        self.controls::<(&T, usize)>()?;
        if let Some(context) = self.0 { context.charge_metadata(std::mem::size_of_val(value))?; }
        Ok(())
    }
    pub(crate) fn format(self, arguments: fmt::Arguments<'_>) -> Result<String, Error> {
        match self.0 {
            Some(context) => context.metadata_string(arguments),
            None => Ok(arguments.to_string()),
        }
    }
    pub(crate) fn configured(self, config: &impl super::Config) -> Result<String, Error> {
        match self.0 {
            Some(context) => config.architecture_fingerprint_with_metadata(context),
            None => Ok(config.architecture_fingerprint()),
        }
    }
    pub(crate) fn text(self, text: &str) -> Result<String, Error> {
        self.format(format_args!("{text}"))
    }
    pub(crate) fn vector<T>(self, count: usize) -> Result<Vec<T>, Error> {
        match self.0 {
            Some(context) => context.metadata_vec(count),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    pub(crate) fn join<T: Display>(self, values: &[T], separator: &str) -> Result<String, Error> {
        struct Joined<'a, T>(&'a [T], &'a str);
        impl<T: Display> Display for Joined<'_, T> {
            fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
                for (index, value) in self.0.iter().enumerate() {
                    if index != 0 {
                        output.write_str(self.1)?;
                    }
                    Display::fmt(value, output)?;
                }
                Ok(())
            }
        }
        self.format(format_args!("{}", Joined(values, separator)))
    }
    pub(crate) fn fingerprint<const N: usize, F>(
        self,
        family: &str,
        fields: F,
    ) -> Result<String, Error>
    where
        F: FnOnce() -> Result<[(&'static str, String); N], Error>,
    {
        if let Some(context) = self.0 {
            let controls = [
                std::mem::size_of::<[(&str, String); N]>(),
                std::mem::size_of::<F>(),
                PromptCacheArchitectureFingerprint::construction_bytes(),
                std::mem::size_of::<Self>(),
                std::mem::size_of::<Result<String, Error>>(),
                std::mem::size_of::<&str>(),
                std::mem::size_of::<&mut [(&str, String)]>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?;
            context.charge_metadata(bytes)?;
        }
        let mut fields = fields()?;
        let digest = PromptCacheArchitectureFingerprint::new(family, &mut fields);
        self.format(format_args!("{digest}"))
    }
    pub(crate) fn error(self, message: fmt::Arguments<'_>) -> Error {
        match self.0 {
            Some(context) => context.metadata_error(message),
            None => Error::backend(message),
        }
    }
    pub(crate) fn source<E: std::error::Error + Send + Sync + 'static>(self, cause: E) -> Error {
        match self.0 {
            Some(context) => context.metadata_source(cause),
            None => Error::backend(cause),
        }
    }
    pub(crate) fn prompt_error(self, message: fmt::Arguments<'_>) -> Error {
        match self.0 {
            Some(context) => match context.metadata_string(message) {
                Ok(message) => context.metadata_source(PromptCacheError::Malformed(message)),
                Err(error) => error,
            },
            None => Error::backend(PromptCacheError::Malformed(message.to_string())),
        }
    }
}
