//! Destinations for the one component placement constructor.
use super::*;
use crate::partitioned_execution::source_allocation::{Allocation, Cause};
use eredu_collections::ordered_map::TryInsertError;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use std::fmt;

#[derive(Clone, Copy)]
pub(crate) struct Destination<'a>(pub(crate) Option<&'a HostMetadataFunding>);
fn cause(value: Cause) -> ComponentPartitionError {
    match value {
        Cause::Funding(cause) => ComponentPartitionError::MetadataFunding(cause),
        Cause::Allocation(cause) => ComponentPartitionError::MetadataAllocation(cause),
        Cause::Capacity => ComponentPartitionError::MetadataCapacity,
        Cause::Semantic(cause) => ComponentPartitionError::ParameterLayout(cause),
    }
}
impl<'a> Destination<'a> {
    pub(crate) fn allocation(self) -> Allocation<'a> {
        Allocation(self.0)
    }
    pub(crate) fn controls<T>(self) -> Result<(), ComponentPartitionError> {
        self.allocation().controls::<T>().map_err(cause)
    }
    pub(crate) fn controls_of<T>(self, value: &T) -> Result<(), ComponentPartitionError> {
        self.allocation().controls_of(value).map_err(cause)
    }
    pub(crate) fn text(self, source: &str) -> Result<String, ComponentPartitionError> {
        self.allocation().text(source).map_err(cause)
    }
    pub(crate) fn format(
        self,
        args: fmt::Arguments<'_>,
    ) -> Result<String, ComponentPartitionError> {
        self.allocation().format(args).map_err(cause)
    }
    pub(crate) fn vector<T>(self, count: usize) -> Result<Vec<T>, ComponentPartitionError> {
        self.allocation().vector(count).map_err(cause)
    }
    pub(crate) fn copy<T: Copy>(self, source: &[T]) -> Result<Vec<T>, ComponentPartitionError> {
        self.allocation().copy(source).map_err(cause)
    }
    pub(crate) fn grow<T>(
        self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), ComponentPartitionError> {
        self.allocation().grow(values, additional).map_err(cause)
    }
    pub(crate) fn source_error(self, error: Cause) -> ComponentPartitionError {
        cause(error)
    }
    pub(crate) fn push<T>(
        self,
        target: &mut Vec<T>,
        value: T,
    ) -> Result<(), ComponentPartitionError> {
        self.allocation().push(target, value).map_err(cause)
    }
    pub(crate) fn diagnostic(
        self,
        constructor: fn(String) -> ComponentPartitionError,
        args: fmt::Arguments<'_>,
    ) -> ComponentPartitionError {
        match self.format(args) {
            Ok(text) => constructor(text),
            Err(error) => error,
        }
    }
    pub(crate) fn invalid(self, name: &str) -> ComponentPartitionError {
        self.diagnostic(
            ComponentPartitionError::InvalidPlacement,
            format_args!("{name}"),
        )
    }
    pub(crate) fn missing(self, name: &str) -> ComponentPartitionError {
        self.diagnostic(
            ComponentPartitionError::MissingWeight,
            format_args!("{name}"),
        )
    }
    pub(crate) fn duplicate(self, name: &str) -> ComponentPartitionError {
        self.diagnostic(
            ComponentPartitionError::DuplicateIdentity,
            format_args!("{name}"),
        )
    }
    pub(crate) fn capture_invalid(self, args: fmt::Arguments<'_>) -> ComponentPartitionError {
        self.diagnostic(
            |text| eredu_core::capture::CaptureError::Invalid(text).into(),
            args,
        )
    }
    pub(crate) fn capture_unsupported(self, args: fmt::Arguments<'_>) -> ComponentPartitionError {
        self.diagnostic(
            |text| eredu_core::capture::CaptureError::Unsupported(text).into(),
            args,
        )
    }
    pub(crate) fn capture_missing(self, name: &str) -> ComponentPartitionError {
        self.diagnostic(
            |text| eredu_core::capture::CaptureError::MissingPath(text).into(),
            format_args!("{name}"),
        )
    }
    pub(crate) fn scope(
        self,
        descriptor: &ArchitectureDescriptor,
        node: &str,
    ) -> Result<eredu_core::speculative::SpeculativeCaptureScope, ComponentPartitionError> {
        self.controls::<(
            &ArchitectureDescriptor,
            &str,
            crate::speculative_execution::ScopeError<'_>,
            Option<&str>,
            usize,
        )>()?;
        self.allocation().reserve(crate::speculative_execution::SpeculativeActivationExecution::scope_validation_control_bytes().ok_or(HostMetadataFundingError::Overflow)?).map_err(cause)?;
        crate::speculative_execution::resolve_scope(descriptor, node).map_err(|error| {
            if matches!(error, crate::speculative_execution::ScopeError::Undeclared) {
                self.capture_unsupported(format_args!("{error}"))
            } else {
                self.capture_invalid(format_args!("{error}"))
            }
        })
    }
    pub(crate) fn routed_geometry(
        self,
        geometry: eredu_core::capture::RoutedUnitGeometry,
    ) -> Result<u64, ComponentPartitionError> {
        let result = match self.0 {
            Some(funding) => geometry.components_with_metadata(funding),
            None => geometry.components(),
        };
        result.map_err(|error| match error {
            eredu_core::capture::CaptureError::AdmissionStorage(
                eredu_core::capture::CaptureAdmissionStorageError::Funding(error),
            ) => ComponentPartitionError::MetadataFunding(error),
            error => ComponentPartitionError::Capture(error),
        })
    }
    pub(crate) fn routed_ownership(
        self,
        value: &eredu_core::capture::RoutedUnitCaptureOwnership,
    ) -> Result<eredu_core::capture::RoutedUnitCaptureOwnership, ComponentPartitionError> {
        self.controls::<(
            &eredu_core::capture::RoutedUnitCaptureOwnership,
            eredu_core::capture::RoutedUnitCaptureOwnership,
        )>()?;
        Ok(eredu_core::capture::RoutedUnitCaptureOwnership {
            coordinates: eredu_core::component::RoutedComponentCoordinateMap::new(
                self.coordinates(value.coordinates.experts())?,
                self.coordinates(value.coordinates.units())?,
            ),
            source_peer: value.source_peer,
            source_peers: value.source_peers,
        })
    }
    pub(crate) fn validate_scope(
        self,
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
        scope: eredu_core::speculative::SpeculativeCaptureScope,
    ) -> Result<(), ComponentPartitionError> {
        self.controls::<(
            &crate::speculative_execution::SpeculativeActivationExecution,
            eredu_core::speculative::SpeculativeCaptureScope,
            bool,
        )>()?;
        execution.validate_scope_by(scope, || {
            self.capture_invalid(format_args!(
                "invocation scope differs from selected prediction strategy or depth"
            ))
        })
    }
    pub(crate) fn axes(
        self,
        source: &[eredu_core::TensorAxis],
    ) -> Result<Vec<eredu_core::TensorAxis>, ComponentPartitionError> {
        self.controls::<(
            &[eredu_core::TensorAxis],
            Vec<eredu_core::TensorAxis>,
            eredu_core::TensorAxis,
        )>()?;
        let mut result = self.vector(source.len())?;
        for axis in source {
            result.push(eredu_core::TensorAxis {
                name: self.text(&axis.name)?,
                dimension: axis.dimension.clone(),
            });
        }
        Ok(result)
    }
    pub(crate) fn indices(
        self,
        count: usize,
        values: Vec<usize>,
    ) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
        self.controls::<(usize, Vec<usize>, ComponentCoordinateMap)>()?;
        match self.0 {
            Some(funding) => ComponentCoordinateMap::indices_with_funding(count, values, funding)
                .map_err(coordinate_error),
            None => ComponentCoordinateMap::indices(count, values).map_err(Into::into),
        }
    }
    pub(crate) fn coordinates(
        self,
        value: &ComponentCoordinateMap,
    ) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
        self.controls::<(&ComponentCoordinateMap, ComponentCoordinateMap)>()?;
        match self.0 {
            Some(funding) => value
                .try_clone_with_funding(funding)
                .map_err(coordinate_error),
            None => Ok(value.clone()),
        }
    }
    pub(crate) fn optional_coordinates(
        self,
        value: Option<&ComponentCoordinateMap>,
    ) -> Result<Option<ComponentCoordinateMap>, ComponentPartitionError> {
        self.controls::<(
            Option<&ComponentCoordinateMap>,
            Option<ComponentCoordinateMap>,
        )>()?;
        value.map(|value| self.coordinates(value)).transpose()
    }
    pub(crate) fn observation(
        self,
        value: &PartitionedObservation,
    ) -> Result<PartitionedObservation, ComponentPartitionError> {
        self.controls::<(&PartitionedObservation, PartitionedObservation)>()?;
        Ok(PartitionedObservation {
            axis: self.text(&value.axis)?,
            coordinates: self.optional_coordinates(value.coordinates.as_ref())?,
            exports: value.exports,
            site: value.site,
            combination: value.combination,
        })
    }
    pub(crate) fn insert<K: Ord, V>(
        self,
        map: &mut SourceMap<K, V>,
        key: K,
        value: V,
    ) -> Result<Option<V>, ComponentPartitionError> {
        self.controls::<(&mut SourceMap<K, V>, K, V, Option<V>)>()?;
        self.allocation()
            .reserve(
                map.insertion_control_bytes(&key)
                    .ok_or(HostMetadataFundingError::Overflow)?,
            )
            .map_err(cause)?;
        map.try_insert_with(key, value, |layout| {
            self.allocation().reserve(layout.size()).map_err(cause)
        })
        .map_err(|error| match error {
            TryInsertError::Funding(error) => error,
            TryInsertError::SizeOverflow => HostMetadataFundingError::Overflow.into(),
        })
    }
}
fn coordinate_error(
    error: eredu_core::component::ComponentCoordinateConstructionError,
) -> ComponentPartitionError {
    use eredu_core::component::ComponentCoordinateConstructionError as E;
    match error {
        E::Coordinates(error) => error.into(),
        E::Funding(error) => error.into(),
        E::Allocation(error) => error.into(),
        E::Capacity => ComponentPartitionError::MetadataCapacity,
    }
}

#[cfg(test)]
pub(crate) mod tests;
