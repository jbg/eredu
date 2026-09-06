use super::*;

/// Native execution resources retained by every prepared MLX model.
///
/// Streams may change on the same device. A distributed world, unlike its
/// rank/size metadata, must be the exact retained native communicator.
pub(crate) struct MlxPreparedTarget {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_target_compares_native_devices_without_gpu_execution() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let target = MlxPreparedTarget {
            device: Device::new(DeviceType::Gpu, 0),
            world: None,
        };
        crate::tests::support::path_instrumentation::reset();
        assert!(target
            .validate(&stream, None)
            .unwrap_err()
            .to_string()
            .contains("different native devices"));
        assert_eq!(
            crate::tests::support::path_instrumentation::snapshot(),
            Default::default()
        );
    }

    #[test]
    fn prepared_target_requires_exact_world_handle_but_accepts_clones() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let world =
            safemlx::distributed::Group::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let clone = world.clone();
        let other =
            safemlx::distributed::Group::init(false, safemlx::distributed::Backend::Ring).unwrap();
        assert_eq!((world.size(), world.rank()), (other.size(), other.rank()));
        assert!(!world.shares_native_handle(&other));
        let manifest = eredu_runtime::CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let communication =
            MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
        let target = MlxPreparedTarget::new(&stream, Some(&communication)).unwrap();
        crate::tests::support::path_instrumentation::reset();
        target.validate(&stream, Some(&clone)).unwrap();
        for replacement in [None, Some(&other)] {
            assert!(target
                .validate(&stream, replacement)
                .unwrap_err()
                .to_string()
                .contains("different native worlds"));
        }
        assert_eq!(
            crate::tests::support::path_instrumentation::snapshot(),
            Default::default()
        );
        assert_eq!(
            crate::tests::support::path_instrumentation::session_reset_attempts(),
            0
        );
        assert_eq!(
            crate::tests::support::path_instrumentation::session_input_creation_attempts(),
            0
        );
    }
}
