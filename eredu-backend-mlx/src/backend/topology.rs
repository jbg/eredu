//! MLX device assignment attached to backend-neutral rank topology.

use std::ops::Deref;

use eredu_core::{BackendError, ParallelRankTopology, ParallelTopology};
use safemlx::{distributed, distributed::Group, Device, DeviceType, Stream};

use crate::backend::error::Error;

/// Explicit process-local MLX execution-device assignment.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeviceAssignment {
    /// MLX device family used by this process.
    pub device_type: DeviceType,
    /// Index within this process's visible devices.
    pub local_index: usize,
}

impl DeviceAssignment {
    /// Creates an explicit process-local assignment.
    pub const fn new(device_type: DeviceType, local_index: usize) -> Self {
        Self {
            device_type,
            local_index,
        }
    }

    /// Resolves this assignment to an MLX device.
    pub fn device(self) -> Result<Device, Error> {
        Ok(distributed::device_for_local_rank(
            self.device_type,
            self.local_index,
        )?)
    }
}

/// Canonical backend-neutral rank topology bound to one process-local MLX device.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MlxParallelContext {
    rank: ParallelRankTopology,
    /// Explicit process-local MLX device assignment.
    pub device: DeviceAssignment,
}

impl MlxParallelContext {
    /// Binds one validated rank topology to an MLX device.
    pub fn new(
        topology: ParallelTopology,
        global_rank: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        if topology.data != 1 {
            return Err(Error::Backend(BackendError::Unsupported {
                backend: "mlx".into(),
                capability: "data-parallel model sessions".into(),
            }));
        }
        if i32::try_from(device.local_index).is_err() {
            return Err(Error::Parallel(format!(
                "local device index {} does not fit in MLX's i32 device index",
                device.local_index
            )));
        }
        Ok(Self {
            rank: ParallelRankTopology::new(topology, global_rank)?,
            device,
        })
    }

    /// Constructs and binds a tensor/pipeline/expert topology for one rank.
    pub fn for_rank(
        global_rank: usize,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        let topology = ParallelTopology::new(
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            1,
        )?;
        Self::new(topology, global_rank, device)
    }

    /// Constructs a topology and snapshots its rank from an MLX world group.
    pub fn for_group(
        group: &Group,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        let topology = ParallelTopology::new(
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            1,
        )?;
        if group.size() != topology.world_size() {
            return Err(Error::Parallel(format!(
                "parallel topology expects world size {} but received {}",
                topology.world_size(),
                group.size()
            )));
        }
        Self::new(topology, group.rank(), device)
    }

    /// Returns the canonical topology shape.
    pub fn topology(self) -> ParallelTopology {
        self.rank.topology()
    }

    /// Returns the canonical backend-neutral rank snapshot.
    pub const fn rank_topology(self) -> ParallelRankTopology {
        self.rank
    }

    /// Verifies that an execution stream uses the assigned MLX device.
    pub fn validate_execution_stream(self, stream: &Stream) -> Result<(), Error> {
        let actual = stream.get_device()?;
        let actual_type = actual.get_type()?;
        let actual_index = actual.get_index()?;
        let expected_index = i32::try_from(self.device.local_index)
            .expect("MLX parallel context validated the local device index");
        if actual_type == self.device.device_type && actual_index == expected_index {
            Ok(())
        } else {
            Err(Error::Parallel(format!(
                "execution stream uses {actual_type:?} device {actual_index}, but this rank is assigned {:?} device {expected_index}",
                self.device.device_type
            )))
        }
    }
}

impl Deref for MlxParallelContext {
    type Target = ParallelRankTopology;

    fn deref(&self) -> &Self::Target {
        &self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_binding_adds_only_local_device_identity() {
        let context = MlxParallelContext::new(
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
            3,
            DeviceAssignment::new(DeviceType::Gpu, 0),
        )
        .unwrap();
        assert_eq!(context.rank_topology().global_rank, 3);
        assert_eq!(context.coordinates().pipeline, 1);
        assert_eq!(context.coordinates().tensor, 1);
        assert_eq!(context.device.local_index, 0);
    }

    #[test]
    fn mlx_fails_closed_for_unimplemented_data_parallel_sessions() {
        let error = MlxParallelContext::new(
            ParallelTopology::new(1, 1, 1, 2).unwrap(),
            0,
            DeviceAssignment::new(DeviceType::Gpu, 0),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Backend(BackendError::Unsupported { .. })
        ));
    }
}
