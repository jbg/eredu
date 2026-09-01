//! MLX device assignment attached to a mechanism-only world rank.

use safemlx::{Device, DeviceType, Stream};

use crate::backend::error::Error;

/// Explicit process-local MLX execution-device assignment.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeviceAssignment {
    device_type: DeviceType,
    local_index: usize,
}

impl DeviceAssignment {
    /// Creates an explicit process-local assignment.
    pub const fn new(device_type: DeviceType, local_index: usize) -> Self {
        Self {
            device_type,
            local_index,
        }
    }

    /// Returns the selected MLX device family.
    pub const fn device_type(self) -> DeviceType {
        self.device_type
    }

    /// Returns the process-local device index.
    pub const fn local_index(self) -> usize {
        self.local_index
    }

    /// Resolves this assignment to an MLX device.
    pub fn device(self) -> Result<Device, Error> {
        let index = i32::try_from(self.local_index)
            .map_err(|_| Error::Parallel("local device index does not fit in i32".into()))?;
        Ok(Device::new(self.device_type, index))
    }
}

/// Mechanism-only world rank bound to one process-local MLX device.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MlxRankContext {
    world_size: usize,
    global_rank: usize,
    device: DeviceAssignment,
}

impl MlxRankContext {
    /// Binds a validated mechanism-only world rank to one MLX device.
    pub fn new(
        world_size: usize,
        global_rank: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        if world_size == 0 || global_rank >= world_size {
            return Err(Error::Parallel(format!(
                "world rank {global_rank} is outside world size {world_size}"
            )));
        }
        if i32::try_from(device.local_index).is_err() {
            return Err(Error::Parallel(format!(
                "local device index {} does not fit in MLX's i32 device index",
                device.local_index
            )));
        }
        Ok(Self {
            world_size,
            global_rank,
            device,
        })
    }

    /// Returns the selected process count.
    pub const fn world_size(self) -> usize {
        self.world_size
    }

    /// Returns this process's world rank.
    pub const fn global_rank(self) -> usize {
        self.global_rank
    }

    /// Resolves the process-local execution device.
    pub fn device(self) -> Result<Device, Error> {
        self.device.device()
    }

    /// Verifies that an execution stream uses the selected process-local device.
    pub fn validate_execution_stream(self, stream: &Stream) -> Result<(), Error> {
        let actual = stream.get_device()?;
        let actual_type = actual.get_type()?;
        let actual_index = actual.get_index()?;
        let expected_index = i32::try_from(self.device.local_index)
            .expect("MLX rank context validated the local device index");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_binding_adds_only_world_rank_and_local_device_identity() {
        let context = MlxRankContext::new(4, 3, DeviceAssignment::new(DeviceType::Gpu, 0)).unwrap();
        assert_eq!(context.global_rank(), 3);
        assert_eq!(context.world_size(), 4);
        assert_eq!(context.device.local_index(), 0);
    }

    #[test]
    fn mlx_rank_context_rejects_out_of_range_rank() {
        assert!(MlxRankContext::new(1, 1, DeviceAssignment::new(DeviceType::Gpu, 0)).is_err());
    }
}
