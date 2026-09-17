// Independent pre-G6 parser. Only the Reader result adds an absent new capability.
use crate::reader::*;
pub(super) fn read<R: Read + Seek>(mut inner: R, limits: Limits) -> Result<Reader<R>> {
    let file_size = inner
        .seek(SeekFrom::End(0))
        .map_err(|source| Error::Io { offset: 0, source })?;
    inner
        .seek(SeekFrom::Start(0))
        .map_err(|source| Error::Io { offset: 0, source })?;
    let mut parser = ReferenceParser {
        inner,
        endian: Endian::Little,
        version: 0,
        limits: &limits,
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

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = parser.string()?;
        if key.len() > u16::MAX as usize || !key.is_ascii() {
            return Err(Error::InvalidMetadata {
                key,
                reason: "keys must be ASCII and at most 65535 bytes".into(),
            });
        }
        let ty = parser.u32()?;
        let value = parser.value(ty, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(Error::DuplicateMetadata(key));
        }
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
    let mut raw = Vec::with_capacity(tensor_capacity);
    let mut names = HashSet::with_capacity(tensor_capacity);
    for _ in 0..tensor_count {
        let name = parser.string()?;
        if name.is_empty() {
            return Err(Error::tensor(name, "empty tensor name"));
        }
        if !names.insert(name.clone()) {
            return Err(Error::DuplicateTensor(name));
        }
        let rank = parser.u32()?;
        if rank > limits.max_rank {
            return Err(Error::Limit {
                resource: "tensor rank",
                actual: rank.into(),
                limit: limits.max_rank.into(),
            });
        }
        let mut dimensions = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dimensions.push(parser.dimension()?);
        }
        let ggml_type = GgmlType::from_code(parser.u32()?);
        let relative_offset = parser.u64()?;
        raw.push((name, dimensions, ggml_type, relative_offset));
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

    let mut tensors = Vec::with_capacity(raw.len());
    for (name, dimensions, ggml_type, relative_offset) in raw {
        if relative_offset % alignment != 0 {
            return Err(Error::tensor(
                &name,
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
                &name,
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
                &name,
                format!("data range {data_offset}..{end} exceeds file size {file_size}"),
            ));
        }
        tensors.push(TensorDescriptor {
            name,
            dimensions,
            ggml_type,
            relative_offset,
            data_offset,
            byte_len,
        });
    }
    let mut ranges: Vec<_> = tensors
        .iter()
        .filter(|t| t.byte_len != 0)
        .map(|t| (t.data_offset, t.data_offset + t.byte_len, &t.name))
        .collect();
    ranges.sort_by_key(|r| r.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Error::tensor(
                pair[1].2,
                format!("data overlaps tensor {:?}", pair[0].2),
            ));
        }
    }

    Ok(Reader {
        inner: parser.inner,
        endian: parser.endian,
        version,
        alignment,
        metadata,
        tensors,
        limits,
        prepared_header: None,
    })
}

struct ReferenceParser<'a, R> {
    inner: R,
    endian: Endian,
    version: u32,
    limits: &'a Limits,
}

impl<R: Read + Seek> ReferenceParser<'_, R> {
    fn pos(&mut self) -> Result<u64> {
        self.inner
            .stream_position()
            .map_err(|source| Error::Io { offset: 0, source })
    }
    fn exact(&mut self, out: &mut [u8]) -> Result<()> {
        let offset = self.pos()?;
        self.inner
            .read_exact(out)
            .map_err(|source| Error::Io { offset, source })
    }
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0];
        self.exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.exact(&mut b)?;
        Ok(self.endian.u16(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.exact(&mut b)?;
        Ok(self.endian.u32(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.exact(&mut b)?;
        Ok(self.endian.u64(b))
    }
    fn count(&mut self) -> Result<u64> {
        if self.version == 1 {
            self.u32().map(Into::into)
        } else {
            self.u64()
        }
    }
    fn dimension(&mut self) -> Result<u64> {
        self.count()
    }
    fn string(&mut self) -> Result<String> {
        let len = self.count()?;
        check_limit("string bytes", len, self.limits.max_string_bytes)?;
        let mut bytes =
            vec![0; usize::try_from(len).map_err(|_| Error::Overflow("string length"))?];
        self.exact(&mut bytes)?;
        String::from_utf8(bytes)
            .map_err(|e| Error::InvalidHeader(format!("invalid UTF-8 string: {e}")))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<MetadataValue> {
        Ok(match ty {
            0 => MetadataValue::Uint8(self.u8()?),
            1 => MetadataValue::Int8(self.u8()? as i8),
            2 => MetadataValue::Uint16(self.u16()?),
            3 => MetadataValue::Int16(self.u16()? as i16),
            4 => MetadataValue::Uint32(self.u32()?),
            5 => MetadataValue::Int32(self.u32()? as i32),
            6 => MetadataValue::Float32(f32::from_bits(self.u32()?)),
            7 => MetadataValue::Bool(match self.u8()? {
                0 => false,
                1 => true,
                v => return Err(Error::InvalidHeader(format!("invalid boolean {v}"))),
            }),
            8 => MetadataValue::String(self.string()?),
            9 => MetadataValue::Array(self.array(depth + 1)?),
            10 => MetadataValue::Uint64(self.u64()?),
            11 => MetadataValue::Int64(self.u64()? as i64),
            12 => MetadataValue::Float64(f64::from_bits(self.u64()?)),
            other => return Err(Error::UnsupportedMetadataType(other)),
        })
    }
    fn array(&mut self, depth: u32) -> Result<MetadataArray> {
        if depth > self.limits.max_metadata_depth {
            return Err(Error::Limit {
                resource: "metadata nesting depth",
                actual: depth.into(),
                limit: self.limits.max_metadata_depth.into(),
            });
        }
        let ty = self.u32()?;
        let len = self.count()?;
        check_limit(
            "metadata array elements",
            len,
            self.limits.max_array_elements,
        )?;
        let n = usize::try_from(len).map_err(|_| Error::Overflow("array length"))?;
        macro_rules! vals {
            ($variant:ident,$expr:expr) => {{
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push($expr?);
                }
                MetadataArray::$variant(v)
            }};
        }
        Ok(match ty {
            0 => vals!(Uint8, self.u8()),
            1 => vals!(Int8, self.u8().map(|v| v as i8)),
            2 => vals!(Uint16, self.u16()),
            3 => vals!(Int16, self.u16().map(|v| v as i16)),
            4 => vals!(Uint32, self.u32()),
            5 => vals!(Int32, self.u32().map(|v| v as i32)),
            6 => vals!(Float32, self.u32().map(f32::from_bits)),
            7 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(match self.u8()? {
                        0 => false,
                        1 => true,
                        x => return Err(Error::InvalidHeader(format!("invalid boolean {x}"))),
                    })
                }
                MetadataArray::Bool(v)
            }
            8 => vals!(String, self.string()),
            9 => vals!(Array, self.array(depth + 1)),
            10 => vals!(Uint64, self.u64()),
            11 => vals!(Int64, self.u64().map(|v| v as i64)),
            12 => vals!(Float64, self.u64().map(f64::from_bits)),
            other => return Err(Error::UnsupportedMetadataType(other)),
        })
    }
}
