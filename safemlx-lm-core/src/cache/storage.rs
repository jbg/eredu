//! Backend-neutral cache storage transition protocol.

use serde::{Deserialize, Serialize};

use super::{CacheBlockId, CacheTier};

/// Stable phase of one cache block's physical resources.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStoragePhase {
    /// The execution backend owns device resources.
    Device,
    /// Device resources are retained while a host copy is produced.
    DemotingToHost,
    /// Host resources exist without a durable backing.
    HostUnbacked,
    /// Host resources are retained by an exact disk-write operation.
    HostWriting,
    /// Host resources have a durable backing.
    HostBacked,
    /// Only the durable backing is resident.
    DiskReady,
    /// A disk-read operation owns the backing and an admitted host allocation.
    DiskReading,
}

impl CacheStoragePhase {
    /// Logical accounting tier for the phase.
    pub const fn tier(self) -> CacheTier {
        match self {
            Self::Device | Self::DemotingToHost => CacheTier::Device,
            Self::HostUnbacked | Self::HostWriting | Self::HostBacked => CacheTier::Host,
            Self::DiskReady | Self::DiskReading => CacheTier::Disk,
        }
    }
}

/// Kind of asynchronous backing-store operation.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheIoOperationKind {
    /// Publish host resources to a durable backing.
    Write,
    /// Reconstruct host resources from a durable backing.
    Read,
}

/// Exact identity of one backing-store operation.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct CacheIoOperationKey {
    /// Cache generation in which the operation was submitted.
    pub generation: u64,
    /// Block whose resources the operation owns.
    pub id: CacheBlockId,
    /// Direction of the operation.
    pub kind: CacheIoOperationKind,
}

/// Opaque backend host-demotion completion with stable identity.
pub trait CacheHostDemotionOperation {
    /// Block whose device resources the operation owns.
    fn block_id(&self) -> &CacheBlockId;

    /// Monotonic operation identity within the backend cache session.
    fn operation_id(&self) -> u64;
}

/// Opaque backend I/O completion with an exact neutral key.
pub trait CacheIoOperation {
    /// Returns the operation identity owned by this completion.
    fn key(&self) -> &CacheIoOperationKey;
}

/// Exact rollback ownership returned by a host-to-device promotion.
#[derive(Debug)]
pub struct CacheHostPromotion<H> {
    id: CacheBlockId,
    source: CacheStoragePhase,
    host: H,
}

/// Canonical physical-resource state machine for one cache block.
///
/// `D`, `H`, and `B` are backend-owned device, host, and backing resources.
/// `HD` and `IO` are backend-owned exact completions. Private resource slots
/// prevent a backend from constructing contradictory phase/resource states.
#[derive(Debug, Clone)]
pub struct CacheBlockStorage<D, H, B, HD, IO> {
    id: CacheBlockId,
    phase: CacheStoragePhase,
    device: Option<D>,
    host: Option<H>,
    backing: Option<B>,
    host_demotion: Option<HD>,
    io: Option<IO>,
}

impl<D, H, B, HD, IO> CacheBlockStorage<D, H, B, HD, IO> {
    /// Creates device residency, optionally retaining an existing backing.
    pub fn device(id: CacheBlockId, device: D, backing: Option<B>) -> Self {
        Self {
            id,
            phase: CacheStoragePhase::Device,
            device: Some(device),
            host: None,
            backing,
            host_demotion: None,
            io: None,
        }
    }

    /// Creates host residency, optionally retaining an existing backing.
    pub fn host(id: CacheBlockId, host: H, backing: Option<B>) -> Self {
        Self {
            id,
            phase: if backing.is_some() {
                CacheStoragePhase::HostBacked
            } else {
                CacheStoragePhase::HostUnbacked
            },
            device: None,
            host: Some(host),
            backing,
            host_demotion: None,
            io: None,
        }
    }

    /// Creates ready backing-only residency.
    pub fn disk(id: CacheBlockId, backing: B) -> Self {
        Self {
            id,
            phase: CacheStoragePhase::DiskReady,
            device: None,
            host: None,
            backing: Some(backing),
            host_demotion: None,
            io: None,
        }
    }

    /// Current canonical phase.
    pub const fn phase(&self) -> CacheStoragePhase {
        self.phase
    }

    /// Block whose physical resources are owned by this state machine.
    pub const fn id(&self) -> &CacheBlockId {
        &self.id
    }

    /// Current logical accounting tier.
    pub const fn tier(&self) -> CacheTier {
        self.phase.tier()
    }

    /// Backend device resources, including while a demotion is pending.
    pub fn device_resource(&self) -> Option<&D> {
        self.device.as_ref()
    }

    /// Backend host resources.
    pub fn host_resource(&self) -> Option<&H> {
        self.host.as_ref()
    }

    /// Durable backing retained by the current phase.
    pub fn backing(&self) -> Option<&B> {
        self.backing.as_ref()
    }

    /// Exact pending host-demotion completion.
    pub fn host_demotion(&self) -> Option<&HD> {
        self.host_demotion.as_ref()
    }

    /// Exact pending backing-store completion.
    pub fn io(&self) -> Option<&IO> {
        self.io.as_ref()
    }
}

impl<D, H, B, HD: CacheHostDemotionOperation, IO: CacheIoOperation>
    CacheBlockStorage<D, H, B, HD, IO>
{
    /// Returns whether the exact backing-store operation is pending.
    pub fn io_matches(&self, key: &CacheIoOperationKey) -> bool {
        self.io.as_ref().is_some_and(|io| io.key() == key)
    }

    /// Fails the operation only when the exact key is still pending.
    pub fn fail_io_if_matches(&mut self, key: &CacheIoOperationKey) -> bool {
        if self.io_matches(key) {
            self.fail_io(key)
                .expect("matching pending I/O has a valid source phase");
            true
        } else {
            false
        }
    }

    /// Begins an exact device-to-host transition while retaining device state.
    pub fn begin_host_demotion(&mut self, operation: HD) -> Result<(), CacheStorageError> {
        self.require_phase(CacheStoragePhase::Device)?;
        self.require_block(operation.block_id())?;
        if self.backing.is_some() {
            return Err(CacheStorageError::BackingAlreadyExists);
        }
        self.host_demotion = Some(operation);
        self.phase = CacheStoragePhase::DemotingToHost;
        Ok(())
    }

    /// Commits the matching host demotion and returns released device resources.
    pub fn finish_host_demotion(
        &mut self,
        operation_id: u64,
        host: H,
    ) -> Result<(D, HD), CacheStorageError> {
        self.require_host_demotion(operation_id)?;
        let device = self
            .device
            .take()
            .expect("demoting phase retains device resources");
        let operation = self
            .host_demotion
            .take()
            .expect("demoting phase retains its exact operation");
        self.host = Some(host);
        self.phase = CacheStoragePhase::HostUnbacked;
        Ok((device, operation))
    }

    /// Abandons the matching host demotion and restores device residency.
    pub fn fail_host_demotion(&mut self, operation_id: u64) -> Result<HD, CacheStorageError> {
        self.require_host_demotion(operation_id)?;
        let operation = self
            .host_demotion
            .take()
            .expect("demoting phase retains its exact operation");
        self.phase = CacheStoragePhase::Device;
        Ok(operation)
    }

    /// Releases backed device resources directly to backing-only residency.
    pub fn release_device_to_disk(&mut self) -> Result<D, CacheStorageError> {
        self.require_phase(CacheStoragePhase::Device)?;
        if self.backing.is_none() {
            return Err(CacheStorageError::BackingRequired);
        }
        let device = self
            .device
            .take()
            .expect("device phase owns device resources");
        self.phase = CacheStoragePhase::DiskReady;
        Ok(device)
    }

    /// Releases backed host resources directly to backing-only residency.
    pub fn release_host_to_disk(&mut self) -> Result<H, CacheStorageError> {
        self.require_phase(CacheStoragePhase::HostBacked)?;
        let host = self
            .host
            .take()
            .expect("host-backed phase owns host resources");
        self.phase = CacheStoragePhase::DiskReady;
        Ok(host)
    }

    /// Starts the exact write that owns unbacked host resources.
    pub fn begin_write(&mut self, operation: IO) -> Result<(), CacheStorageError> {
        self.require_phase(CacheStoragePhase::HostUnbacked)?;
        self.require_block(&operation.key().id)?;
        self.require_kind(operation.key(), CacheIoOperationKind::Write)?;
        self.io = Some(operation);
        self.phase = CacheStoragePhase::HostWriting;
        Ok(())
    }

    /// Commits the matching write and returns released host resources.
    pub fn finish_write(
        &mut self,
        key: &CacheIoOperationKey,
        backing: B,
    ) -> Result<(H, IO), CacheStorageError> {
        self.require_io(CacheStoragePhase::HostWriting, key)?;
        let host = self
            .host
            .take()
            .expect("host-writing phase retains host resources");
        let operation = self.io.take().expect("host-writing phase retains I/O");
        self.backing = Some(backing);
        self.phase = CacheStoragePhase::DiskReady;
        Ok((host, operation))
    }

    /// Starts the exact read that owns backing-only resources.
    pub fn begin_read(&mut self, operation: IO) -> Result<(), CacheStorageError> {
        self.require_phase(CacheStoragePhase::DiskReady)?;
        self.require_block(&operation.key().id)?;
        self.require_kind(operation.key(), CacheIoOperationKind::Read)?;
        self.io = Some(operation);
        self.phase = CacheStoragePhase::DiskReading;
        Ok(())
    }

    /// Commits the matching read while retaining its durable backing.
    pub fn finish_read(
        &mut self,
        key: &CacheIoOperationKey,
        host: H,
    ) -> Result<IO, CacheStorageError> {
        self.require_io(CacheStoragePhase::DiskReading, key)?;
        let operation = self.io.take().expect("disk-reading phase retains I/O");
        self.host = Some(host);
        self.phase = CacheStoragePhase::HostBacked;
        Ok(operation)
    }

    /// Cancels or fails the matching I/O and restores its stable source phase.
    pub fn fail_io(&mut self, key: &CacheIoOperationKey) -> Result<IO, CacheStorageError> {
        let stable = match self.phase {
            CacheStoragePhase::HostWriting => CacheStoragePhase::HostUnbacked,
            CacheStoragePhase::DiskReading => CacheStoragePhase::DiskReady,
            actual => return Err(CacheStorageError::IoNotPending { actual }),
        };
        self.require_io(self.phase, key)?;
        let operation = self.io.take().expect("pending phase retains I/O");
        self.phase = stable;
        Ok(operation)
    }

    /// Promotes stable host resources to device residency.
    pub fn promote_host(&mut self, device: D) -> Result<CacheHostPromotion<H>, CacheStorageError> {
        if !matches!(
            self.phase,
            CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked
        ) {
            return Err(CacheStorageError::InvalidPhase {
                expected: CacheStoragePhase::HostUnbacked,
                actual: self.phase,
            });
        }
        let source = self.phase;
        let host = self
            .host
            .take()
            .expect("stable host phase owns host resources");
        self.device = Some(device);
        self.phase = CacheStoragePhase::Device;
        Ok(CacheHostPromotion {
            id: self.id.clone(),
            source,
            host,
        })
    }

    /// Restores host resources after a rejected promotion.
    pub fn restore_host(
        &mut self,
        promotion: CacheHostPromotion<H>,
    ) -> Result<D, CacheStorageError> {
        self.require_phase(CacheStoragePhase::Device)?;
        self.require_block(&promotion.id)?;
        let expected_source = if self.backing.is_some() {
            CacheStoragePhase::HostBacked
        } else {
            CacheStoragePhase::HostUnbacked
        };
        if promotion.source != expected_source {
            return Err(CacheStorageError::InvalidPhase {
                expected: expected_source,
                actual: promotion.source,
            });
        }
        let device = self
            .device
            .take()
            .expect("device phase owns device resources");
        self.host = Some(promotion.host);
        self.phase = promotion.source;
        Ok(device)
    }

    fn require_phase(&self, expected: CacheStoragePhase) -> Result<(), CacheStorageError> {
        if self.phase == expected {
            Ok(())
        } else {
            Err(CacheStorageError::InvalidPhase {
                expected,
                actual: self.phase,
            })
        }
    }

    fn require_host_demotion(&self, operation_id: u64) -> Result<(), CacheStorageError> {
        self.require_phase(CacheStoragePhase::DemotingToHost)?;
        let actual = self
            .host_demotion
            .as_ref()
            .expect("demoting phase retains its exact operation")
            .operation_id();
        if actual == operation_id {
            Ok(())
        } else {
            Err(CacheStorageError::HostDemotionMismatch {
                expected: actual,
                actual: operation_id,
            })
        }
    }

    fn require_io(
        &self,
        phase: CacheStoragePhase,
        key: &CacheIoOperationKey,
    ) -> Result<(), CacheStorageError> {
        self.require_phase(phase)?;
        let expected = self
            .io
            .as_ref()
            .expect("pending I/O phase retains its exact operation")
            .key();
        if expected == key {
            Ok(())
        } else {
            Err(CacheStorageError::IoMismatch {
                expected: Box::new(expected.clone()),
                actual: Box::new(key.clone()),
            })
        }
    }

    fn require_kind(
        &self,
        key: &CacheIoOperationKey,
        expected: CacheIoOperationKind,
    ) -> Result<(), CacheStorageError> {
        if key.kind == expected {
            Ok(())
        } else {
            Err(CacheStorageError::IoKindMismatch {
                expected,
                actual: key.kind,
            })
        }
    }

    fn require_block(&self, actual: &CacheBlockId) -> Result<(), CacheStorageError> {
        if actual == &self.id {
            Ok(())
        } else {
            Err(CacheStorageError::BlockMismatch {
                expected: Box::new(self.id.clone()),
                actual: Box::new(actual.clone()),
            })
        }
    }
}

/// Illegal cache storage phase transition or completion observation.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CacheStorageError {
    /// An operation belongs to a different cache block.
    #[error("cache storage block mismatch: expected {expected:?}, observed {actual:?}")]
    BlockMismatch {
        /// Block owned by the state machine.
        expected: Box<CacheBlockId>,
        /// Block owned by the presented operation.
        actual: Box<CacheBlockId>,
    },
    /// An operation is not legal from the current phase.
    #[error("cache storage phase is {actual:?}, expected {expected:?}")]
    InvalidPhase {
        /// Required source phase.
        expected: CacheStoragePhase,
        /// Observed source phase.
        actual: CacheStoragePhase,
    },
    /// No backing-store operation is pending in the current phase.
    #[error("cache storage phase {actual:?} has no pending I/O")]
    IoNotPending {
        /// Observed stable or unrelated phase.
        actual: CacheStoragePhase,
    },
    /// A device block already had a durable backing.
    #[error("cache storage already has a durable backing")]
    BackingAlreadyExists,
    /// A direct release required a durable backing.
    #[error("cache storage requires a durable backing")]
    BackingRequired,
    /// A different host-demotion completion attempted to resolve the phase.
    #[error("host demotion operation mismatch: expected {expected}, observed {actual}")]
    HostDemotionMismatch {
        /// Operation retained by the state machine.
        expected: u64,
        /// Operation presented by the backend.
        actual: u64,
    },
    /// A different backing-store completion attempted to resolve the phase.
    #[error("cache I/O operation mismatch: expected {expected:?}, observed {actual:?}")]
    IoMismatch {
        /// Operation retained by the state machine.
        expected: Box<CacheIoOperationKey>,
        /// Operation presented by the backend.
        actual: Box<CacheIoOperationKey>,
    },
    /// A read/write completion was submitted to the opposite transition.
    #[error("cache I/O kind is {actual:?}, expected {expected:?}")]
    IoKindMismatch {
        /// Required operation direction.
        expected: CacheIoOperationKind,
        /// Observed operation direction.
        actual: CacheIoOperationKind,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheRepresentation;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct HostOp {
        id: CacheBlockId,
        operation_id: u64,
    }

    impl CacheHostDemotionOperation for HostOp {
        fn block_id(&self) -> &CacheBlockId {
            &self.id
        }

        fn operation_id(&self) -> u64 {
            self.operation_id
        }
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct IoOp(CacheIoOperationKey);

    impl CacheIoOperation for IoOp {
        fn key(&self) -> &CacheIoOperationKey {
            &self.0
        }
    }

    fn block() -> CacheBlockId {
        CacheBlockId {
            session_id: 1,
            global_layer: 2,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 4,
            rank: None,
        }
    }

    fn io(kind: CacheIoOperationKind, generation: u64) -> IoOp {
        IoOp(CacheIoOperationKey {
            generation,
            id: block(),
            kind,
        })
    }

    #[test]
    fn exact_host_demotion_commits_or_rolls_back_without_losing_device_state() {
        let id = block();
        let mut storage = CacheBlockStorage::<_, String, String, _, IoOp>::device(
            id.clone(),
            "device".to_owned(),
            None,
        );
        storage
            .begin_host_demotion(HostOp {
                id,
                operation_id: 7,
            })
            .unwrap();
        assert_eq!(storage.phase(), CacheStoragePhase::DemotingToHost);
        assert_eq!(
            storage.finish_host_demotion(8, "host".to_owned()),
            Err(CacheStorageError::HostDemotionMismatch {
                expected: 7,
                actual: 8,
            })
        );
        assert_eq!(
            storage.device_resource().map(String::as_str),
            Some("device")
        );
        storage.fail_host_demotion(7).unwrap();
        assert_eq!(storage.phase(), CacheStoragePhase::Device);
    }

    #[test]
    fn write_read_and_promotion_follow_one_exact_transaction_chain() {
        let mut storage = CacheBlockStorage::<String, String, String, HostOp, _>::host(
            block(),
            "host".to_owned(),
            None,
        );
        let write = io(CacheIoOperationKind::Write, 3);
        storage.begin_write(write.clone()).unwrap();
        let stale = io(CacheIoOperationKind::Write, 2);
        assert!(matches!(
            storage.finish_write(stale.key(), "disk".to_owned()),
            Err(CacheStorageError::IoMismatch { .. })
        ));
        let (host, observed) = storage
            .finish_write(write.key(), "disk".to_owned())
            .unwrap();
        assert_eq!(host, "host");
        assert_eq!(observed, write);
        assert_eq!(storage.phase(), CacheStoragePhase::DiskReady);

        let read = io(CacheIoOperationKind::Read, 4);
        storage.begin_read(read.clone()).unwrap();
        storage
            .finish_read(read.key(), "host-2".to_owned())
            .unwrap();
        assert_eq!(storage.phase(), CacheStoragePhase::HostBacked);
        let promotion = storage.promote_host("device-2".to_owned()).unwrap();
        assert_eq!(promotion.host, "host-2");
        assert_eq!(storage.backing().map(String::as_str), Some("disk"));
        storage.release_device_to_disk().unwrap();
        assert_eq!(storage.phase(), CacheStoragePhase::DiskReady);
    }

    #[test]
    fn failed_io_restores_the_stable_source_phase() {
        let mut write = CacheBlockStorage::<String, _, String, HostOp, _>::host(block(), 3u8, None);
        let write_op = io(CacheIoOperationKind::Write, 1);
        write.begin_write(write_op.clone()).unwrap();
        assert_eq!(write.fail_io(write_op.key()).unwrap(), write_op);
        assert_eq!(write.phase(), CacheStoragePhase::HostUnbacked);

        let mut read =
            CacheBlockStorage::<String, u8, _, HostOp, _>::disk(block(), "disk".to_owned());
        let read_op = io(CacheIoOperationKind::Read, 1);
        read.begin_read(read_op.clone()).unwrap();
        assert_eq!(read.fail_io(read_op.key()).unwrap(), read_op);
        assert_eq!(read.phase(), CacheStoragePhase::DiskReady);
    }

    #[test]
    fn operation_for_another_block_fails_without_changing_phase() {
        let owned = block();
        let mut foreign = block();
        foreign.global_layer += 1;
        let mut storage =
            CacheBlockStorage::<String, u8, String, HostOp, _>::host(owned.clone(), 3, None);
        let operation = IoOp(CacheIoOperationKey {
            generation: 1,
            id: foreign.clone(),
            kind: CacheIoOperationKind::Write,
        });

        assert_eq!(
            storage.begin_write(operation),
            Err(CacheStorageError::BlockMismatch {
                expected: Box::new(owned),
                actual: Box::new(foreign),
            })
        );
        assert_eq!(storage.phase(), CacheStoragePhase::HostUnbacked);
    }

    #[test]
    fn phase_and_operation_identity_round_trip_portably() {
        let key = io(CacheIoOperationKind::Read, 9).0;
        let encoded = serde_json::to_string(&(CacheStoragePhase::DiskReading, &key)).unwrap();
        let decoded: (CacheStoragePhase, CacheIoOperationKey) =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, (CacheStoragePhase::DiskReading, key));
    }
}
