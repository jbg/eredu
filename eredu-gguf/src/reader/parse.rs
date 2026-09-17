//! The same ordered parser with closed owning or immutable destinations.
use super::*;
pub(super) struct Parsed<'h, R> {
    pub(super) inner: R,
    pub(super) endian: Endian,
    pub(super) version: u32,
    pub(super) alignment: u64,
    pub(super) metadata: Cow<'h, BTreeMap<String, MetadataValue>>,
    pub(super) tensors: Cow<'h, Vec<TensorDescriptor>>,
    pub(super) limits: Limits,
    pub(super) prepared_header: Option<Arc<PreparedHeader>>,
}
pub(super) fn parse<'h, R: Read + Seek>(
    mut inner: R,
    limits: Limits,
    policy: HeaderPolicy,
    source: Option<&'h PreparedHeader>,
    scratch: Option<&mut [u8]>,
) -> Result<Parsed<'h, R>> {
    let file_size = inner
        .seek(SeekFrom::End(0))
        .map_err(|source| Error::Io { offset: 0, source })?;
    inner
        .seek(SeekFrom::Start(0))
        .map_err(|source| Error::Io { offset: 0, source })?;
    let mut parser = Parser {
        inner,
        endian: Endian::Little,
        version: 0,
        limits: &limits,
        policy,
        scratch,
    };
    let mut magic = [0; 4];
    parser.exact(&mut magic)?;
    parser.endian = match &magic {
        b"GGUF" => Endian::Little,
        b"FUGG" => Endian::Big,
        _ => return Err(Error::InvalidHeader(format!("invalid magic {magic:?}"))),
    };
    let version = parser.u32()?;
    if !(1..=3).contains(&version) {
        return Err(Error::UnsupportedVersion(version));
    }
    parser.version = version;
    let tensor_count = parser.count()?;
    check_limit("tensor count", tensor_count, limits.max_tensor_count)?;
    let metadata_count = parser.count()?;
    check_limit(
        "metadata entries",
        metadata_count,
        limits.max_metadata_entries,
    )?;

    let mut metadata = storage::Metadata::new(source)?;
    for _ in 0..metadata_count {
        let key = parser.string(metadata.expected_key()?)?;
        if key.len() > u16::MAX as usize || !key.is_ascii() {
            return Err(Error::InvalidMetadata {
                key: storage::owned(key)?,
                reason: "keys must be ASCII and at most 65535 bytes".into(),
            });
        }
        let ty = parser.u32()?;
        let value = parser.value(ty, 0, metadata.expected_value(key.as_str())?)?;
        let stored_key = key.clone();
        parser.policy.typed::<u8>(
            HeaderStorageKind::MetadataKey,
            stored_key.len(),
            stored_key.capacity(),
        )?;
        if metadata.insert(stored_key, value)?.is_some() {
            return Err(Error::DuplicateMetadata(storage::owned(key)?));
        }
        parser
            .policy
            .collection(HeaderStorageKind::MetadataEntries, metadata.len(), None)?;
        parser.policy.keep_metadata_key(key)?;
    }
    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(MetadataValue::Uint32(v)) => u64::from(*v),
        Some(MetadataValue::Uint64(v)) => *v,
        Some(_) => {
            return Err(Error::InvalidMetadata {
                key: "general.alignment".into(),
                reason: "must be uint32 or uint64".into(),
            });
        }
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::InvalidHeader(format!(
            "invalid alignment {alignment}"
        )));
    }

    let tensor_capacity =
        usize::try_from(tensor_count).map_err(|_| Error::Overflow("tensor count"))?;
    type RawDescriptor = (String, Vec<u64>, GgmlType, u64);
    let final_source = source.map(PreparedHeader::final_header).transpose()?;
    let mut raw = storage::Raw::new(tensor_capacity, final_source)?;
    parser.policy.typed::<RawDescriptor>(
        HeaderStorageKind::RawDescriptors,
        tensor_capacity,
        raw.capacity(),
    )?;
    let mut names = storage::Names::new(tensor_capacity, final_source)?;
    parser.policy.collection(
        HeaderStorageKind::TensorNames,
        tensor_capacity,
        names.capacity(),
    )?;
    for _ in 0..tensor_count {
        let expected = raw.expected()?;
        let name = parser.string(expected.map(|d| &d.name))?;
        if name.is_empty() {
            return Err(Error::tensor(name.as_str(), "empty tensor name"));
        }
        let stored_name = name.clone();
        parser.policy.typed::<u8>(
            HeaderStorageKind::TensorName,
            stored_name.len(),
            stored_name.capacity(),
        )?;
        if !names.insert(stored_name)? {
            return Err(Error::DuplicateTensor(storage::owned(name)?));
        }
        let rank = parser.u32()?;
        if rank > limits.max_rank {
            return Err(Error::Limit {
                resource: "tensor rank",
                actual: rank.into(),
                limit: limits.max_rank.into(),
            });
        }
        let mut dimensions = storage::Vector::new(rank as usize, expected.map(|d| &d.dimensions))?;
        parser.policy.typed::<u64>(
            HeaderStorageKind::Dimensions,
            rank as usize,
            dimensions.capacity(),
        )?;
        for _ in 0..rank {
            dimensions.push(Cow::Owned(parser.dimension()?))?;
        }
        let ggml_type = GgmlType::from_code(parser.u32()?);
        let relative_offset = parser.u64()?;
        raw.push(name, dimensions.finish()?, ggml_type, relative_offset)?;
    }
    let descriptor_end = parser.pos()?;
    let data_start = align_up(descriptor_end, alignment)?;
    // A metadata-only GGUF has no tensor-data section, so writers are not
    // required to materialize padding up to the aligned data start.
    if tensor_count != 0 && data_start > file_size {
        return Err(Error::InvalidHeader(
            "tensor data starts beyond end of file".into(),
        ));
    }

    let mut tensors = storage::Final::new(raw.len(), final_source)?;
    parser.policy.typed::<TensorDescriptor>(
        HeaderStorageKind::FinalDescriptors,
        raw.len(),
        tensors.capacity(),
    )?;
    for (name, dimensions, ggml_type, relative_offset) in raw.into_iter() {
        if relative_offset % alignment != 0 {
            return Err(Error::tensor(
                name.as_str(),
                format!("relative offset {relative_offset} is not aligned to {alignment}"),
            ));
        }
        let elements = dimensions.iter().try_fold(1u64, |a, &b| {
            a.checked_mul(b)
                .ok_or(Error::Overflow("tensor element count"))
        })?;
        let (block, bytes) = ggml_type.block_and_bytes()?;
        if elements != 0
            && (dimensions.first().copied().unwrap_or(1) % block != 0 || elements % block != 0)
        {
            return Err(Error::tensor(
                name.as_str(),
                format!("shape {dimensions:?} is not divisible by block size {block}"),
            ));
        }
        let byte_len = (elements / block)
            .checked_mul(bytes)
            .ok_or(Error::Overflow("tensor byte length"))?;
        check_limit("tensor allocation", byte_len, limits.max_allocation_bytes)?;
        let data_offset = data_start
            .checked_add(relative_offset)
            .ok_or(Error::Overflow("tensor offset"))?;
        let end = data_offset
            .checked_add(byte_len)
            .ok_or(Error::Overflow("tensor end offset"))?;
        if end > file_size {
            return Err(Error::tensor(
                name.as_str(),
                format!("data range {data_offset}..{end} exceeds file size {file_size}"),
            ));
        }
        tensors.push(
            name,
            dimensions,
            ggml_type,
            relative_offset,
            data_offset,
            byte_len,
        )?;
    }
    // Keep the ordinary range Vec alive across the same final checks. The
    // prepared branch borrows cold coordinates and creates no range backing.
    let mut ranges: Vec<_> = if final_source.is_none() {
        tensors
            .as_slice()
            .iter()
            .filter(|t| t.byte_len != 0)
            .map(|t| (t.data_offset, t.data_offset + t.byte_len, &t.name))
            .collect()
    } else {
        Vec::new()
    };
    if let Some(header) = final_source {
        parser
            .policy
            .collection(HeaderStorageKind::Ranges, header.ranges.len(), None)?;
        for pair in header.ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Error::tensor(
                    &header.tensors[pair[1].2].name,
                    format!("data overlaps tensor {:?}", &header.tensors[pair[0].2].name),
                ));
            }
        }
    } else {
        parser.policy.ranges(&ranges)?;
        ranges.sort_by_key(|r| r.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Error::tensor(
                    pair[1].2,
                    format!("data overlaps tensor {:?}", pair[0].2),
                ));
            }
        }
    }
    let prepared_header = parser.policy.finish()?;
    Ok(Parsed {
        inner: parser.inner,
        endian: parser.endian,
        version,
        alignment,
        metadata: metadata.finish()?,
        tensors: tensors.finish()?,
        limits,
        prepared_header,
    })
}
