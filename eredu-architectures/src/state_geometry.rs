
//! Shared destinations for actual architecture state-geometry construction.
use eredu_core::{
    AttentionPolicy, LayerSchedule,
    cache::{
        LayerCachePolicy, StateTensorDimension, StateTensorPolicy, StateTensorRole,
        StateTensorDtype, MutableStateResidency,
    },
};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use eredu_runtime::{StateLayout, StateSegmentSpec, StateSegmentLifetime, StateError};
use std::{
    fmt,
    marker::PhantomData,
    ops::Range,
    mem::{size_of, size_of_val},
};

pub(crate) trait Destination {
    type Error;
    fn error(&self, text: fmt::Arguments<'_>) -> Self::Error;
    fn vector<T>(&self, count: usize) -> Result<Vec<T>, Self::Error>;
    fn push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), Self::Error>;
    fn controls<T>(&self) -> Result<(), Self::Error>;
    fn schedule<P>(&self, count: usize, values: Vec<P>) -> Result<LayerSchedule<P>, Self::Error>;
    fn layout(&self, layers: LayerSchedule<LayerCachePolicy>) -> Result<StateLayout, Self::Error>;
    fn segmented(
        &self,
        layers: LayerSchedule<LayerCachePolicy>,
        segments: Vec<StateSegmentSpec>,
    ) -> Result<StateLayout, Self::Error>;
    fn segment(
        &self,
        id: &str,
        layers: Range<usize>,
        lifetime: StateSegmentLifetime,
        offset: i32,
    ) -> Result<StateSegmentSpec, Self::Error>;
    fn collect_values<T>(&self, values: impl Iterator<Item = T>) -> Result<Vec<T>, Self::Error> {
        let mut out = self.vector(values.size_hint().0)?;
        for value in values {
            self.push(&mut out, value)?;
        }
        Ok(out)
    }
    fn values<T, const N: usize>(&self, values: [T; N]) -> Result<Vec<T>, Self::Error> {
        let mut out = self.vector(N)?;
        out.extend(values);
        Ok(out)
    }
    fn collect<T>(
        &self,
        values: impl ExactSizeIterator<Item = Result<T, Self::Error>>,
    ) -> Result<Vec<T>, Self::Error> {
        let mut out = self.vector(values.len())?;
        for value in values {
            out.push(value?);
        }
        Ok(out)
    }
    fn fixed(&self, value: i32) -> Result<StateTensorDimension, Self::Error> {
        StateTensorDimension::fixed_with_diagnostic(value, |text| self.error(text))
    }
    fn key_value(
        &self,
        attention: AttentionPolicy,
        heads: i32,
        dim: i32,
    ) -> Result<LayerCachePolicy, Self::Error> {
        LayerCachePolicy::key_value_with_diagnostic(attention, heads, dim, |text| self.error(text))
    }
    fn compressed(
        &self,
        attention: AttentionPolicy,
        latent: i32,
        rotary: i32,
    ) -> Result<LayerCachePolicy, Self::Error> {
        LayerCachePolicy::compressed_latent_rotary_with_diagnostic(
            attention,
            latent,
            rotary,
            |text| self.error(text),
        )
    }
    fn fixed_only(&self, tensors: Vec<StateTensorPolicy>) -> Result<LayerCachePolicy, Self::Error> {
        LayerCachePolicy::fixed_only_with_diagnostic(tensors, |text| self.error(text))
    }
    fn tensor(
        &self,
        role: StateTensorRole,
        shape: Vec<StateTensorDimension>,
        dtype: StateTensorDtype,
        residency: MutableStateResidency,
    ) -> Result<StateTensorPolicy, Self::Error> {
        StateTensorPolicy::new_with_diagnostic(role, shape, dtype, residency, |text| {
            self.error(text)
        })
    }
}

pub(crate) struct Ordinary<E>(pub(crate) fn(String) -> E);
impl<E> Destination for Ordinary<E> {
    type Error = E;
    fn error(&self, text: fmt::Arguments<'_>) -> E {
        (self.0)(text.to_string())
    }
    fn vector<T>(&self, count: usize) -> Result<Vec<T>, E> {
        Ok(Vec::with_capacity(count))
    }
    fn push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), E> {
        values.push(value);
        Ok(())
    }
    fn controls<T>(&self) -> Result<(), E> {
        Ok(())
    }
    fn schedule<P>(&self, count: usize, values: Vec<P>) -> Result<LayerSchedule<P>, E> {
        LayerSchedule::new(count, values).map_err(|cause| self.error(format_args!("{cause}")))
    }
    fn layout(&self, layers: LayerSchedule<LayerCachePolicy>) -> Result<StateLayout, E> {
        StateLayout::new(layers).map_err(|cause| self.error(format_args!("{cause}")))
    }
    fn segmented(
        &self,
        layers: LayerSchedule<LayerCachePolicy>,
        segments: Vec<StateSegmentSpec>,
    ) -> Result<StateLayout, E> {
        StateLayout::segmented(layers, segments)
            .map_err(|cause| self.error(format_args!("{cause}")))
    }
    fn segment(
        &self,
        id: &str,
        layers: Range<usize>,
        lifetime: StateSegmentLifetime,
        offset: i32,
    ) -> Result<StateSegmentSpec, E> {
        StateSegmentSpec::new(id, layers, lifetime, offset)
            .map_err(|cause| self.error(format_args!("{cause}")))
    }
}

pub(crate) struct Counted<'a, E> {
    context: &'a WorkspaceContext,
    error: fn(String) -> E,
    kind: PhantomData<E>,
}
impl<'a, E: std::error::Error + Send + Sync + 'static> Counted<'a, E> {
    pub(crate) fn new(context: &'a WorkspaceContext, error: fn(String) -> E) -> Self {
        Self {
            context,
            error,
            kind: PhantomData,
        }
    }
    fn state_error(&self, cause: StateError) -> Error {
        match cause {
            StateError::WorkspaceConstruction(cause) => cause,
            cause => self.error(format_args!("{cause}")),
        }
    }
}
impl<E: std::error::Error + Send + Sync + 'static> Destination for Counted<'_, E> {
    type Error = Error;
    fn error(&self, text: fmt::Arguments<'_>) -> Error {
        match self.context.metadata_string(text) {
            Ok(text) => self.context.metadata_source((self.error)(text)),
            Err(cause) => cause,
        }
    }
    fn vector<T>(&self, count: usize) -> Result<Vec<T>, Error> {
        self.context.metadata_vec(count)
    }
    fn push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), Error> {
        self.context.reserve_metadata_vec(values, 1)?;
        values.push(value);
        Ok(())
    }
    fn controls<T>(&self) -> Result<(), Error> {
        let controls = [
            size_of::<T>(),
            size_of::<Result<T, Error>>(),
            size_of::<Self>(),
        ];
        self.context.charge_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        Ok(())
    }
    fn schedule<P>(&self, count: usize, values: Vec<P>) -> Result<LayerSchedule<P>, Error> {
        self.controls::<LayerSchedule<P>>()?;
        // Only a valid conversion reaches the Vec-to-box handoff. Spare
        // allocator capacity can require a second exact payload allocation.
        if count != 0
            && count == values.len()
            && values.capacity() != values.len()
            && size_of::<P>() != 0
        {
            self.context.charge_metadata(
                std::alloc::Layout::array::<P>(values.len())
                    .map_err(|_| WorkspaceMetadataError::Overflow)?
                    .size(),
            )?;
        }
        LayerSchedule::new(count, values).map_err(|cause| self.error(format_args!("{cause}")))
    }
    fn layout(&self, layers: LayerSchedule<LayerCachePolicy>) -> Result<StateLayout, Error> {
        StateLayout::new_with_metadata(layers, self.context)
            .map_err(|cause| self.state_error(cause))
    }
    fn segmented(
        &self,
        layers: LayerSchedule<LayerCachePolicy>,
        segments: Vec<StateSegmentSpec>,
    ) -> Result<StateLayout, Error> {
        StateLayout::segmented_with_metadata(layers, segments, self.context)
            .map_err(|cause| self.state_error(cause))
    }
    fn segment(
        &self,
        id: &str,
        layers: Range<usize>,
        lifetime: StateSegmentLifetime,
        offset: i32,
    ) -> Result<StateSegmentSpec, Error> {
        StateSegmentSpec::new_with_metadata(id, layers, lifetime, offset, self.context)
            .map_err(|cause| self.state_error(cause))
    }
}
