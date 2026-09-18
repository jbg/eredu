//! One destination-aware producer for the actual ordered routing declarations.
use super::{RoutedBankId, RoutedObservationPoint};
use eredu_collections::ordered_map::{Map, TryInsertError};
use eredu_core::HostMetadataFunding;
use eredu_nn::{Error, workspace::{WorkspaceContext, WorkspaceMetadataError}};
use std::{fmt, mem::size_of, sync::Arc};

/// Independently identified routed observation points within one logical layer.
/// The paths and ordered nodes retire before their exact construction account.
#[derive(Debug, Clone)]
pub struct RoutedObservationPoints {
    points: Arc<Map<RoutedBankId, RoutedObservationPoint>>,
    funding: Option<HostMetadataFunding>,
}
impl PartialEq for RoutedObservationPoints {
    fn eq(&self, other: &Self) -> bool { self.points == other.points }
}
impl Eq for RoutedObservationPoints {}
impl RoutedObservationPoints {
    /// Declares the first routing bank through the caller's actual destination.
    pub fn new(bank: RoutedBankId, path: fmt::Arguments<'_>, expert_count: i32,
        metadata: Option<&WorkspaceContext>) -> Result<Self, Error> {
        if let Some(context) = metadata {
            context.charge_metadata(size_of::<(RoutedBankId, fmt::Arguments<'_>, i32,
                Option<&WorkspaceContext>, Self, Result<Self, Error>)>())?;
        }
        Self::empty(metadata)?.with_bank(bank, path, expert_count, metadata)
    }
    fn empty(metadata: Option<&WorkspaceContext>) -> Result<Self, Error> {
        if let Some(context) = metadata {
            context.charge_metadata(size_of::<(Option<&WorkspaceContext>, Map<RoutedBankId, RoutedObservationPoint>,
                Arc<Map<RoutedBankId, RoutedObservationPoint>>, Self, Result<Self, Error>)>())?;
        }
        let points = match metadata {
            Some(context) => context.metadata_arc(Map::new())?,
            None => Arc::new(Map::new()),
        };
        Ok(Self { points, funding: metadata.and_then(WorkspaceContext::metadata_funding) })
    }
    /// Adds one distinct bank using the same construction account.
    pub fn with_bank(mut self, bank: RoutedBankId, path: fmt::Arguments<'_>, expert_count: i32,
        metadata: Option<&WorkspaceContext>) -> Result<Self, Error> {
        if let Some(context) = metadata {
            context.charge_metadata(size_of::<(Self, RoutedBankId, fmt::Arguments<'_>, i32,
                Option<&WorkspaceContext>, Option<HostMetadataFunding>, String, RoutedObservationPoint,
                Option<usize>, Result<Option<RoutedObservationPoint>, TryInsertError<WorkspaceMetadataError>>,
                Result<Self, Error>)>())?;
        }
        let funding = metadata.and_then(WorkspaceContext::metadata_funding);
        let same = match (&self.funding, &funding) {
            (Some(left), Some(right)) => left.same_account(right),
            (None, None) => true,
            _ => false,
        };
        if !same { return Err(WorkspaceMetadataError::Unqualified.into()); }
        if let Some(context) = metadata {
            context.charge_metadata(self.points.insertion_control_bytes(&bank)
                .ok_or(WorkspaceMetadataError::Overflow)?)?;
        }
        if self.points.contains_key(&bank) {
            return Err(match metadata {
                Some(context) => context.metadata_error(format_args!("duplicate routed observation bank")),
                None => Error::backend("duplicate routed observation bank"),
            });
        }
        if Arc::get_mut(&mut self.points).is_none() {
            // Mutation of a shared declaration copies each actual path and node
            // through this same producer. Cloning the source itself never allocates.
            let mut copied = Self::empty(metadata)?;
            let iter = self.iter();
            if let Some(context) = metadata { context.charge_metadata(std::mem::size_of_val(&iter)
                + size_of::<Self>() + size_of::<(RoutedBankId, &RoutedObservationPoint)>())?; }
            for (id, point) in iter {
                copied = copied.with_bank(id, format_args!("{}", point.path), point.expert_count, metadata)?;
            }
            return copied.with_bank(bank, path, expert_count, metadata);
        }
        let path = match metadata {
            Some(context) => context.metadata_string(path)?,
            None => path.to_string(),
        };
        Arc::get_mut(&mut self.points).expect("unique destination").try_insert_with(bank, RoutedObservationPoint { path, expert_count }, |layout| {
            match metadata {
                Some(context) => context.charge_metadata(layout.size()),
                None => Ok(()),
            }
        }).map_err(|cause| match cause {
            TryInsertError::Funding(error) => Error::from(error),
            TryInsertError::SizeOverflow => WorkspaceMetadataError::Overflow.into(),
        })?;
        Ok(self)
    }
    /// Borrows every bank in stable order without cloning its declaration.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (RoutedBankId, &RoutedObservationPoint)> {
        self.points.iter().map(|(bank, point)| (*bank, point))
    }
    /// Resolves one bank's canonical path and global cardinality.
    pub fn bank(&self, bank: RoutedBankId) -> Option<&RoutedObservationPoint> { self.points.get(&bank) }
}
