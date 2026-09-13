use super::CaptureError;

/// Original route identity in a source peer's input, not a prediction index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedUnitOrigin {
    /// Source peer in the retained exchange group; absent without exchange.
    pub source_peer: Option<usize>,
    /// Flattened token row in the original operator input.
    pub token: usize,
    /// Original top-k slot; duplicate expert selections retain distinct slots.
    pub slot: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_origins_preserve_idle_peers_and_distinct_duplicate_positions() {
        let origins = RoutedUnitOrigins::new(&[1, 0, 1], &[0, 0], 2).unwrap();
        assert_eq!(origins.peer_count(), 3);
        assert_eq!(origins.row_count(), 2);
        assert_eq!(
            origins.resolve(0),
            Some(RoutedUnitOrigin {
                source_peer: Some(0),
                token: 0,
                slot: 0
            })
        );
        assert_eq!(
            origins.resolve(1),
            Some(RoutedUnitOrigin {
                source_peer: Some(2),
                token: 0,
                slot: 0
            })
        );
        assert_eq!(origins.resolve(2), None);
        assert!(RoutedUnitOrigins::new(&[usize::MAX, 1], &[], 1).is_err());
        assert!(RoutedUnitOrigins::new(&[1], &[], 1).is_err());
        assert!(RoutedUnitOrigins::new(&[0], &[], 0).is_err());
    }
}

/// Checked borrowed tags in peer-major exchange receive order. This is coordinate
/// metadata, not admission, completion evidence or authority to copy native data.
#[derive(Debug, Clone, Copy)]
pub struct RoutedUnitOrigins<'a> {
    peer_counts: &'a [usize],
    route_positions: &'a [usize],
    routes_per_token: usize,
}

impl<'a> RoutedUnitOrigins<'a> {
    /// Checks exact metadata coverage without allocating or copying host vectors.
    pub fn new(
        peer_counts: &'a [usize],
        route_positions: &'a [usize],
        routes_per_token: usize,
    ) -> Result<Self, CaptureError> {
        if routes_per_token == 0
            || peer_counts
                .iter()
                .try_fold(0usize, |n, &m| n.checked_add(m))
                != Some(route_positions.len())
        {
            return Err(CaptureError::Invalid(
                "invalid routed-unit exchange origin geometry".into(),
            ));
        }
        Ok(Self {
            peer_counts,
            route_positions,
            routes_per_token,
        })
    }
    /// Number of source peers, including peers with no received routes.
    pub fn peer_count(&self) -> usize {
        self.peer_counts.len()
    }
    /// Number of received native input rows described by these tags.
    pub fn row_count(&self) -> usize {
        self.route_positions.len()
    }
    /// Original selected route slots per source token.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }
    /// Resolves one received row through the original route tag.
    pub fn resolve(&self, row: usize) -> Option<RoutedUnitOrigin> {
        let position = *self.route_positions.get(row)?;
        let mut start = 0usize;
        for (peer, &count) in self.peer_counts.iter().enumerate() {
            let end = start.checked_add(count)?;
            if row < end {
                return Some(RoutedUnitOrigin {
                    source_peer: Some(peer),
                    token: position / self.routes_per_token,
                    slot: position % self.routes_per_token,
                });
            }
            start = end;
        }
        None
    }
}
