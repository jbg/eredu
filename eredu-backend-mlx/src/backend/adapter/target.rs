use super::*;

/// Native execution resources retained by every prepared MLX model.
///
/// Streams may change on the same device. A distributed world, unlike its
/// rank/size metadata, must be the exact retained native communicator.
#[doc(hidden)]
pub struct MlxPreparedTarget {
    device: Device,
    world: Option<safemlx::distributed::Group>,
}

impl MlxPreparedTarget {
    pub(crate) fn new(
        stream: &Stream,
        distributed: Option<&MlxDistributedSession>,
    ) -> Result<Self, Error> {
        Ok(Self {
            device: stream.get_device()?,
            world: distributed.map(|distributed| distributed.native_world().clone()),
        })
    }

    pub(super) fn validate(
        &self,
        stream: &Stream,
        world: Option<&safemlx::distributed::Group>,
    ) -> Result<(), Error> {
        validate_native_execution_target(&self.device, self.world.as_ref(), stream, world)
    }
}

pub(crate) fn validate_native_execution_target(
    expected_device: &Device,
    expected_world: Option<&safemlx::distributed::Group>,
    stream: &Stream,
    world: Option<&safemlx::distributed::Group>,
) -> Result<(), Error> {
    if expected_device != &stream.get_device()? {
        return Err(Error::Parallel(
            "prepared MLX execution and supplied context use different native devices".into(),
        ));
    }
    if let Some(expected) = expected_world {
        if !world.is_some_and(|actual| expected.shares_native_handle(actual)) {
            return Err(Error::Parallel(
                "prepared MLX execution and supplied context use different native worlds".into(),
            ));
        }
    }
    Ok(())
}
