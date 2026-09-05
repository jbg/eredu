use super::*;

pub(in crate::backend) fn device_capabilities(has_world: bool) -> DeviceCapabilities {
    DeviceCapabilities::new(true, true, has_world)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum MlxAcceleratorFamily {
    Metal,
    Cuda,
}

impl MlxAcceleratorFamily {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub(crate) const fn is_compiled(self) -> bool {
        match self {
            Self::Metal => cfg!(all(feature = "metal", target_vendor = "apple")),
            Self::Cuda => cfg!(feature = "cuda"),
        }
    }

    pub(crate) fn is_available(self) -> Result<bool, Error> {
        match self {
            Self::Metal => {
                #[cfg(all(feature = "metal", target_vendor = "apple"))]
                {
                    safemlx::metal::is_available().map_err(Into::into)
                }
                #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
                {
                    Ok(false)
                }
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    safemlx::cuda::is_available().map_err(Into::into)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(false)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct MlxDeviceIdentity {
    kind: DeviceType,
    family: &'static str,
    index: i32,
}

impl MlxDeviceIdentity {
    pub(crate) fn from_realized_device(
        device: &Device,
        accelerator_family: Option<MlxAcceleratorFamily>,
    ) -> Result<Self, Error> {
        let kind = device.get_type()?;
        let family = match (kind, accelerator_family) {
            (DeviceType::Cpu, None) => "cpu",
            (DeviceType::Gpu, Some(family)) => family.as_str(),
            (DeviceType::Cpu, Some(family)) => {
                return Err(Error::AutomaticPlanning(format!(
                    "realized CPU device cannot have accelerator family {}",
                    family.as_str()
                )))
            }
            (DeviceType::Gpu, None) => {
                return Err(Error::AutomaticPlanning(
                    "realized GPU device is missing its concrete accelerator family".into(),
                ))
            }
        };
        Ok(Self {
            kind,
            family,
            index: device.get_index()?,
        })
    }

    pub(super) fn validate_device(&self, device: &Device) -> Result<(), Error> {
        let kind = device.get_type()?;
        let index = device.get_index()?;
        if kind != self.kind || index != self.index {
            return Err(Error::AutomaticPlanning(format!(
                "realized device identity {}:{} does not match backend stream device {kind:?}:{index}",
                self.family, self.index
            )));
        }
        Ok(())
    }

    pub(super) fn descriptor(&self) -> DeviceDescriptor {
        DeviceDescriptor::new(
            format!("{}:{}", self.family, self.index),
            format!("MLX {} {}", self.family, self.index),
            self.family,
            None,
        )
    }
}

pub(super) fn infer_native_device_identity(device: &Device) -> Result<MlxDeviceIdentity, Error> {
    if device.get_type()? == DeviceType::Cpu {
        return MlxDeviceIdentity::from_realized_device(device, None);
    }

    let mut available = Vec::new();
    for family in [MlxAcceleratorFamily::Metal, MlxAcceleratorFamily::Cuda] {
        if family.is_compiled() && family.is_available()? {
            available.push(family);
        }
    }
    let family = available.first().copied().ok_or_else(|| {
        Error::AutomaticPlanning(
            "cannot identify native MLX GPU stream because no compiled accelerator family is available"
                .into(),
        )
    })?;
    if available.len() != 1 {
        return Err(Error::AutomaticPlanning(
            "cannot identify native MLX GPU stream because multiple accelerator families are available"
                .into(),
        ));
    }
    MlxDeviceIdentity::from_realized_device(device, Some(family))
}
