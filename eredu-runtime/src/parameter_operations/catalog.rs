//! Loaded metadata exchange and complete effective-parameter ownership.
use super::*;
use eredu_core::parameters::{LoadedParameter, ParameterCoordinateMap, ParameterRegion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Metadata from one actual prepared slot, without retaining its tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalParameterFacts {
    /// Global effective shape and the native operations selected for this slot.
    pub parameter: LoadedParameter,
    /// Effective local shape after the retained placement and encoding transforms.
    pub local_shape: Vec<u64>,
}

/// Complete loaded facts and actual owning ranks. Cold declarations are used
/// only to validate coordinates; they cannot manufacture a loaded slot.
pub struct PartitionParameterCatalog {
    parameters: Vec<LoadedParameter>,
    coordinates: BTreeMap<String, Vec<Option<ParameterCoordinateMap>>>,
}
impl PartitionParameterCatalog {
    /// Joins actual reports and verifies every reported value has complete global
    /// source coverage. The coordinate callback consumes retained architecture
    /// placement and validates each slot's declared identity and sharing.
    pub fn new<R: CaptureReservation>(
        ranks: &[Vec<LocalParameterFacts>],
        mut coordinate: impl FnMut(
            usize,
            &LoadedParameter,
            &mut R,
        ) -> Result<Option<ParameterCoordinateMap>, ParameterError>,
        reservation: &mut R,
    ) -> Result<Self, ParameterError> {
        if ranks.is_empty() {
            return Err(ParameterError::Incomplete(
                "empty loaded parameter world".into(),
            ));
        }
        let mut parameters = BTreeMap::<String, LoadedParameter>::new();
        let mut coordinates = BTreeMap::<String, Vec<Option<ParameterCoordinateMap>>>::new();
        let mut populated = BTreeSet::new();
        for (rank, facts) in ranks.iter().enumerate() {
            let mut seen = BTreeSet::new();
            for facts in facts {
                validate(facts)?;
                let parameter = &facts.parameter;
                charge(reservation, add(512, mul(parameter.id.len() as u64, 4)?)?)?;
                if !seen.insert(&parameter.id) {
                    return Err(invalid("duplicate rank-local parameter identity"));
                }
                let map = coordinate(rank, parameter, reservation)?.ok_or_else(|| {
                    invalid("loaded parameter is absent from retained rank ownership")
                })?;
                if map.global_shape() != parameter.shape || map.local_shape() != facts.local_shape {
                    return Err(invalid(
                        "loaded parameter differs from retained effective geometry",
                    ));
                }
                if !parameters.contains_key(&parameter.id) {
                    charge(
                        reservation,
                        add(
                            add(512, mul(ranks.len() as u64, 256)?)?,
                            descriptor_bytes(parameter)?,
                        )?,
                    )?;
                    parameters.insert(parameter.id.clone(), parameter.clone());
                    coordinates.insert(
                        parameter.id.clone(),
                        (0..ranks.len()).map(|_| None).collect(),
                    );
                } else {
                    let global = parameters
                        .get_mut(&parameter.id)
                        .expect("inserted descriptor");
                    if global.shape != parameter.shape || global.shared_id != parameter.shared_id {
                        return Err(invalid("ranks disagree on parameter geometry or sharing"));
                    }
                    if !facts.local_shape.contains(&0) && !populated.contains(&parameter.id) {
                        *global = parameter.clone();
                    } else if !facts.local_shape.contains(&0) {
                        let left = global.access();
                        let right = parameter.access();
                        let same_dtype = global.dtype == parameter.dtype;
                        global.access = Some(eredu_core::parameters::ParameterAccess {
                            query: left.query && right.query && same_dtype,
                            projection: left.projection && right.projection && same_dtype,
                            replacement: left.replacement && right.replacement && same_dtype,
                        });
                        if !same_dtype {
                            global.dtype = None;
                            global.condition =
                                "Owning ranks use different effective floating dtypes".into();
                        }
                        if global.input_transform != parameter.input_transform {
                            global.input_transform =
                                eredu_core::parameters::ProjectionInputTransform::Unspecified;
                            global.condition = "Owning ranks use different projection input transformations; a single global transform is unavailable".into();
                        }
                    }
                }
                if !facts.local_shape.contains(&0) && !populated.contains(&parameter.id) {
                    populated.insert(parameter.id.clone());
                }
                coordinates.get_mut(&parameter.id).expect("inserted owners")[rank] = Some(map);
            }
        }
        if parameters.is_empty() {
            return Err(ParameterError::Incomplete(
                "no actual loaded parameter slots".into(),
            ));
        }
        for parameter in parameters.values() {
            let Some(shared) = parameters.get(&parameter.shared_id) else {
                return Err(ParameterError::Incomplete(format!(
                    "shared parameter {} has no loaded canonical slot",
                    parameter.shared_id
                )));
            };
            if shared.shared_id != shared.id || shared.shape != parameter.shape {
                return Err(invalid(
                    "cyclic sharing or incompatible shared parameter geometry",
                ));
            }
            let maps = &coordinates[&parameter.id];
            charge(reservation, add(128, mul(maps.len() as u64, 16)?)?)?;
            let maps = maps.iter().map(Option::as_ref).collect::<Vec<_>>();
            let region = ParameterRegion {
                starts: vec![0; parameter.shape.len()],
                shape: parameter.shape.clone(),
            };
            PartitionParameterReadPlan::new(
                &parameter.shape,
                &region,
                None,
                &maps,
                65_536,
                reservation,
            )?;
        }
        for parameter in parameters.values_mut() {
            let access = parameter.access();
            parameter.supported = access.query && access.projection && access.replacement;
        }
        Ok(Self {
            parameters: parameters.into_values().collect(),
            coordinates,
        })
    }
    /// Deterministically ordered actual global loaded descriptors.
    pub fn parameters(&self) -> &[LoadedParameter] {
        &self.parameters
    }
    /// Actual source owners for a canonical slot or loaded alias.
    pub fn coordinates(&self, parameter: &str) -> Option<&[Option<ParameterCoordinateMap>]> {
        self.coordinates.get(parameter).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests;

fn validate(facts: &LocalParameterFacts) -> Result<(), ParameterError> {
    let p = &facts.parameter;
    if p.id.is_empty()
        || p.id.len() > 4096
        || p.shared_id.is_empty()
        || p.shared_id.len() > 4096
        || p.condition.len() > 4096
        || p.shape.len() > 32
        || p.shape.contains(&0)
        || facts.local_shape.len() != p.shape.len()
    {
        return Err(invalid("loaded parameter metadata bounds"));
    }
    let access = p.access();
    if (access.query || access.projection || access.replacement) && p.dtype.is_none() {
        return Err(invalid(
            "supported parameter has no effective floating dtype",
        ));
    }
    Ok(())
}
fn descriptor_bytes(p: &LoadedParameter) -> Result<u64, ParameterError> {
    Ok(add(
        mul(
            (p.id.len() + p.shared_id.len() + p.condition.len()) as u64,
            2,
        )?,
        mul(p.shape.len() as u64, 32)?,
    )?)
}
fn invalid(message: &str) -> ParameterError {
    ParameterError::Invalid(message.into())
}
fn charge(
    reservation: &mut (impl CaptureReservation + ?Sized),
    bytes: u64,
) -> Result<(), ParameterError> {
    reservation.reserve_quota(CaptureUsage {
        host_bytes: bytes,
        ..Default::default()
    })?;
    Ok(())
}

/// Encodes bounded prepared metadata into portable words. No native value or
/// tensor is serialized, and buffers are charged before allocation.
pub fn encode_parameter_catalog(
    facts: &[LocalParameterFacts],
    max_bytes: usize,
    reservation: &mut impl CaptureReservation,
) -> Result<Vec<u32>, ParameterError> {
    for fact in facts {
        validate(fact)?;
    }
    charge(reservation, add(128, mul(max_bytes as u64, 4)?)?)?;
    struct Writer {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self
                .bytes
                .len()
                .checked_add(bytes.len())
                .is_none_or(|n| n > self.limit)
            {
                return Err(std::io::Error::other("parameter catalogue encoded limit"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Writer {
        bytes: Vec::new(),
        limit: max_bytes,
    };
    serde_json::to_writer(&mut writer, facts).map_err(|error| invalid(&error.to_string()))?;
    let count = writer.bytes.len() as u64;
    let mut words = Vec::with_capacity(writer.bytes.len().div_ceil(4) + 2);
    words.extend([count as u32, (count >> 32) as u32]);
    for chunk in writer.bytes.chunks(4) {
        let mut bytes = [0; 4];
        bytes[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(bytes));
    }
    Ok(words)
}

/// Checks exact encoded length and padding, then decodes within prepaid finite
/// metadata bounds. The returned facts still require architecture validation.
pub fn decode_parameter_catalog(
    words: &[u32],
    max_bytes: usize,
    reservation: &mut impl CaptureReservation,
) -> Result<Vec<LocalParameterFacts>, ParameterError> {
    if words.len() < 2 {
        return Err(invalid("truncated parameter catalogue"));
    }
    let bytes = usize::try_from(words[0] as u64 | ((words[1] as u64) << 32))
        .map_err(|_| ParameterError::Overflow)?;
    if bytes > max_bytes
        || words.len()
            != bytes
                .div_ceil(4)
                .checked_add(2)
                .ok_or(ParameterError::Overflow)?
    {
        return Err(invalid("parameter catalogue length exceeds its bound"));
    }
    charge(reservation, add(256, mul(bytes as u64, 32)?)?)?;
    let mut decoded = Vec::with_capacity(bytes);
    for word in &words[2..] {
        for byte in word.to_le_bytes() {
            if decoded.len() < bytes {
                decoded.push(byte);
            } else if byte != 0 {
                return Err(invalid("parameter catalogue padding is nonzero"));
            }
        }
    }
    let facts: Vec<LocalParameterFacts> =
        serde_json::from_slice(&decoded).map_err(|error| invalid(&error.to_string()))?;
    for fact in &facts {
        validate(fact)?;
    }
    Ok(facts)
}
