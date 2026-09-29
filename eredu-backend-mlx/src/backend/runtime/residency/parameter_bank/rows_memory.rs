//! Buffer facts for the synchronous compact row decoder, not its source cache.
use eredu_nn::{mechanism_memory::*, TensorElementType};
use eredu_runtime::{RowEncoding, RowLookupDescriptor, RowLookupError};

pub(super) fn describe(
    prepared: &RowLookupDescriptor,
) -> Result<MechanismMemoryContract, RowLookupError> {
    let mut result = prepared.decode_memory()?;
    result.missing = vec![
        "MLX allocator capacity and kernel-private workspace are undescribed".into(),
        "encoded row/scale owner backing and residency transfer buffers are described separately".into(),
        "portable lookup planning, concatenation/reordering and scalar preparation are separate invocations".into(),
    ];
    let rows = prepared.maximum_acquisition_rows();
    let encoded = &result.values[0];
    let output = result
        .values
        .iter()
        .find(|v| v.name == "output")
        .expect("prepared output");
    let output_bytes = output.logical_bytes()?;
    let float_bytes = rows
        .checked_mul(prepared.spec().dimensions as u64)
        .and_then(|n| n.checked_mul(4))
        .ok_or(RowLookupError::Geometry)?;
    let mut compact = allocation(
        "encoded_compact",
        encoded.logical_bytes()?,
        MechanismStorageRole::Scratch,
        StorageRetention::NativeCompletion,
    );
    // Concatenation can elide allocation for a single acquired source row.
    compact.payload.lower = 0;
    compact.capacity.lower = 0;
    result.storage.push(compact);
    result.storage.push(allocation(
        "row_indices",
        rows.checked_mul(4).ok_or(RowLookupError::Geometry)?,
        MechanismStorageRole::Scratch,
        StorageRetention::NativeCompletion,
    ));
    let decoded_element = match prepared.spec().encoding {
        RowEncoding::Dense => encoded.element,
        RowEncoding::ScalarE4M3 { .. } => {
            result.storage.push(allocation(
                "decoded_f32",
                float_bytes,
                MechanismStorageRole::Scratch,
                StorageRetention::NativeCompletion,
            ));
            result.storage.push(allocation(
                "scaled_f32",
                float_bytes,
                MechanismStorageRole::Scratch,
                StorageRetention::NativeCompletion,
            ));
            if result
                .values
                .iter()
                .find(|v| v.name == "scale")
                .expect("prepared scale")
                .element
                != TensorElementType::F32
            {
                result.storage.push(allocation(
                    "scale_f32",
                    4,
                    MechanismStorageRole::Scratch,
                    StorageRetention::NativeCompletion,
                ));
            }
            TensorElementType::F32
        }
        RowEncoding::Gguf { .. } => {
            if super::scalar_decoder(&prepared.spec().encoding).is_some() {
                let mut host = allocation(
                    "decoded_host",
                    float_bytes,
                    MechanismStorageRole::Scratch,
                    StorageRetention::NativeCompletion,
                );
                host.placement = MechanismPlacement::Host;
                host.detail = "caller-owned F32 output of the allocation-free scalar block decoder; released before native return".into();
                result.storage.push(host);
                let mut staging = allocation(
                    "decoded_staging",
                    float_bytes,
                    MechanismStorageRole::Scratch,
                    StorageRetention::NativeCompletion,
                );
                staging.placement = MechanismPlacement::Unknown;
                staging.detail = "native-owned staging copied from host decoded values; its physical pool is not established".into();
                result.storage.push(staging);
                let mut copied = allocation(
                    "decoded_stream",
                    float_bytes,
                    MechanismStorageRole::Scratch,
                    StorageRetention::NativeCompletion,
                );
                copied.backing = MechanismBacking::Unknown;
                copied.payload.lower = 0;
                copied.capacity.lower = 0;
                copied.detail =
                    "stream copy may alias native staging; allocation identity is not established"
                        .into();
                result.storage.push(copied);
                result.missing.push("scalar-decoder native staging/copy physical pools and aliasing are not established".into());
                TensorElementType::F32
            } else {
                // Device-dependent quantized embedding paths need their own exact
                // mechanism facts. Do not fabricate an always-F32 intermediate.
                result.missing.push(
                    "GGUF embedding decoding intermediates depend on the selected native path"
                        .into(),
                );
                result.storage.push(allocation(
                    "output",
                    output_bytes,
                    MechanismStorageRole::Output,
                    StorageRetention::Returned,
                ));
                result.validate()?;
                return Ok(result);
            }
        }
    };
    if decoded_element != prepared.spec().output_type {
        result.storage.push(allocation(
            "cast_output",
            output_bytes,
            MechanismStorageRole::Scratch,
            StorageRetention::NativeCompletion,
        ));
    }
    result.storage.push(allocation(
        "output",
        output_bytes,
        MechanismStorageRole::Output,
        StorageRetention::Returned,
    ));
    result.validate()?;
    Ok(result)
}
fn allocation(
    name: &str,
    bytes: u64,
    role: MechanismStorageRole,
    retention: StorageRetention,
) -> MechanismStorage {
    MechanismStorage {
        name: name.into(), role,
        payload: MechanismBytes::exact(bytes),
        capacity: MechanismBytes::unknown(bytes),
        backing: MechanismBacking::Invocation,
        placement: MechanismPlacement::Execution,
        retention,
        detail: "MLX row decoding completes and synchronizes dependencies before returning an independent gathered output; allocator capacity is unknown".into(),
    }
}

/// Conservative decoder reservation: encoded compaction, up to four F32
/// intermediates per element (six for scalar decoding), and device I32 indices shared by GGUF decoding
/// and the final ownership gather. Residency transfers are charged separately.
pub(super) fn workspace(
    rows: u64,
    encoded_row_bytes: u64,
    spec: &eredu_runtime::RowLookupSpec,
) -> Result<u64, RowLookupError> {
    let dimensions = spec.dimensions as u64;
    if rows == 0 || rows > i32::MAX as u64 || dimensions == 0 || dimensions > i32::MAX as u64 {
        return Err(RowLookupError::Geometry);
    }
    // The scalar fallback additionally owns a host decode result and native
    // staging/copy. Keep the same bound in cold selection and live acquisition.
    let floats = if super::scalar_decoder(&spec.encoding).is_some() {
        24
    } else {
        16
    };
    dimensions
        .checked_mul(floats)
        .and_then(|n| n.checked_add(encoded_row_bytes))
        .and_then(|n| n.checked_add(4))
        .and_then(|n| n.checked_mul(rows))
        .ok_or(RowLookupError::Geometry)
}
