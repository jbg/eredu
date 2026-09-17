//! Closed owning/borrowed destinations for the one ordered header parser.
use super::*;
use std::borrow::Cow;

pub(super) type Name<'h> = Cow<'h, String>;
pub(super) type Values<'h, T> = Cow<'h, Vec<T>>;
pub(super) type RawDescriptor = (String, Vec<u64>, GgmlType, u64);
pub(super) type RangeCoordinate = (u64, u64, usize);

#[derive(Debug)]
pub(super) struct FinalHeader {
    pub(super) metadata: BTreeMap<String, MetadataValue>,
    pub(super) tensors: Vec<TensorDescriptor>,
    pub(super) ranges: Vec<RangeCoordinate>,
}
impl FinalHeader {
    pub(super) fn new(
        metadata: BTreeMap<String, MetadataValue>,
        tensors: Vec<TensorDescriptor>,
    ) -> Self {
        let mut ranges: Vec<_> = tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.byte_len != 0)
            .map(|(i, t)| (t.data_offset, t.data_offset + t.byte_len, i))
            .collect();
        ranges.sort_by_key(|r| r.0);
        Self {
            metadata,
            tensors,
            ranges,
        }
    }
}
pub(super) fn refused(resource: &'static str) -> Error {
    Error::PreparedHeaderStorage { resource }
}
pub(super) fn owned<T: Clone>(value: Cow<'_, T>) -> Result<T> {
    match value {
        Cow::Owned(value) => Ok(value),
        Cow::Borrowed(_) => Err(refused("owning header destination")),
    }
}

pub(super) enum Vector<'h, T: Clone> {
    Owned(Vec<T>),
    Borrowed { source: &'h Vec<T>, filled: usize },
}
impl<'h, T: Clone> Vector<'h, T> {
    pub(super) fn new(count: usize, source: Option<&'h Vec<T>>) -> Result<Self> {
        if let Some(source) = source {
            if source.len() != count {
                return Err(refused("header vector length"));
            }
            Ok(Self::Borrowed { source, filled: 0 })
        } else {
            Ok(Self::Owned(Vec::with_capacity(count)))
        }
    }
    pub(super) fn record_array(&self, policy: &mut HeaderPolicy, count: usize) -> Result<()> {
        policy.typed::<T>(HeaderStorageKind::Array, count, self.capacity())
    }
    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            Self::Borrowed { source, .. } => source.capacity(),
        }
    }
    pub(super) fn push(&mut self, value: Cow<'h, T>) -> Result<()> {
        match self {
            Self::Owned(v) => v.push(owned(value)?),
            Self::Borrowed { source, filled } => {
                if *filled >= source.len() {
                    return Err(refused("header vector emission"));
                }
                *filled += 1; // Actual bytes were compared before decoding; no numeric equality authority.
            }
        }
        Ok(())
    }
    pub(super) fn finish(self) -> Result<Values<'h, T>> {
        match self {
            Self::Owned(v) => Ok(Cow::Owned(v)),
            Self::Borrowed { source, filled } => {
                if source.len() != filled {
                    return Err(refused("header vector completion"));
                }
                Ok(Cow::Borrowed(source))
            }
        }
    }
}

pub(super) enum Metadata<'h> {
    Owned(BTreeMap<String, MetadataValue>),
    Borrowed {
        header: &'h PreparedHeader,
        filled: usize,
    },
}
impl<'h> Metadata<'h> {
    pub(super) fn new(header: Option<&'h PreparedHeader>) -> Result<Self> {
        match header {
            None => Ok(Self::Owned(BTreeMap::new())),
            Some(header) => {
                header.final_header()?;
                Ok(Self::Borrowed { header, filled: 0 })
            }
        }
    }
    pub(super) fn expected_key(&self) -> Result<Option<&'h String>> {
        match self {
            Self::Owned(_) => Ok(None),
            Self::Borrowed { header, filled } => header
                .metadata_order
                .get(*filled)
                .map(Some)
                .ok_or_else(|| refused("metadata key coordinate")),
        }
    }
    pub(super) fn expected_value(&self, key: &str) -> Result<Option<&'h MetadataValue>> {
        match self {
            Self::Owned(_) => Ok(None),
            Self::Borrowed { header, .. } => header
                .final_header()?
                .metadata
                .get(key)
                .map(Some)
                .ok_or_else(|| refused("metadata value coordinate")),
        }
    }
    pub(super) fn insert(
        &mut self,
        key: Name<'h>,
        value: Cow<'h, MetadataValue>,
    ) -> Result<Option<MetadataValue>> {
        match self {
            Self::Owned(map) => Ok(map.insert(owned(key)?, owned(value)?)),
            Self::Borrowed { header, filled } => {
                let expected = header
                    .metadata_order
                    .get(*filled)
                    .ok_or_else(|| refused("metadata insertion coordinate"))?;
                if key.as_str() != expected.as_str() {
                    return Err(refused("metadata insertion identity"));
                }
                let expected_value = header
                    .final_header()?
                    .metadata
                    .get(expected)
                    .ok_or_else(|| refused("metadata insertion value"))?;
                if !matches!(value,Cow::Borrowed(actual) if std::ptr::eq(actual,expected_value)) {
                    return Err(refused("metadata value ownership"));
                }
                // Successful cold input has unique keys; its exact order/bytes are authenticated.
                *filled += 1;
                Ok(None)
            }
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Owned(m) => m.len(),
            Self::Borrowed { filled, .. } => *filled,
        }
    }
    pub(super) fn get(&self, key: &str) -> Option<&MetadataValue> {
        match self {
            Self::Owned(m) => m.get(key),
            Self::Borrowed { header, .. } => header.parsed.as_ref()?.metadata.get(key),
        }
    }
    pub(super) fn finish(self) -> Result<Cow<'h, BTreeMap<String, MetadataValue>>> {
        match self {
            Self::Owned(m) => Ok(Cow::Owned(m)),
            Self::Borrowed { header, filled } => {
                let m = &header.final_header()?.metadata;
                if filled != m.len() {
                    return Err(refused("metadata completion"));
                }
                Ok(Cow::Borrowed(m))
            }
        }
    }
}

pub(super) enum Names<'h> {
    Owned(HashSet<String>),
    Borrowed {
        source: &'h [TensorDescriptor],
        filled: usize,
    },
}
impl<'h> Names<'h> {
    pub(super) fn new(count: usize, source: Option<&'h FinalHeader>) -> Result<Self> {
        if let Some(s) = source {
            if s.tensors.len() != count {
                return Err(refused("name population"));
            }
            Ok(Self::Borrowed {
                source: &s.tensors,
                filled: 0,
            })
        } else {
            Ok(Self::Owned(HashSet::with_capacity(count)))
        }
    }
    pub(super) fn capacity(&self) -> Option<usize> {
        match self {
            Self::Owned(s) => Some(s.capacity()),
            Self::Borrowed { .. } => None,
        }
    }
    pub(super) fn insert(&mut self, name: Name<'h>) -> Result<bool> {
        match self {
            Self::Owned(s) => Ok(s.insert(owned(name)?)),
            Self::Borrowed { source, filled } => {
                let expected = source
                    .get(*filled)
                    .ok_or_else(|| refused("name insertion"))?;
                if name.as_str() != expected.name {
                    return Err(refused("name coordinate"));
                }
                *filled += 1;
                Ok(true)
            }
        }
    }
}

pub(super) enum Raw<'h> {
    Owned(Vec<RawDescriptor>),
    Borrowed {
        source: &'h Vec<TensorDescriptor>,
        filled: usize,
    },
}
impl<'h> Raw<'h> {
    pub(super) fn new(count: usize, source: Option<&'h FinalHeader>) -> Result<Self> {
        if let Some(s) = source {
            if s.tensors.len() != count {
                return Err(refused("raw population"));
            }
            Ok(Self::Borrowed {
                source: &s.tensors,
                filled: 0,
            })
        } else {
            Ok(Self::Owned(Vec::with_capacity(count)))
        }
    }
    pub(super) fn expected(&self) -> Result<Option<&'h TensorDescriptor>> {
        match self {
            Self::Owned(_) => Ok(None),
            Self::Borrowed { source, filled } => source
                .get(*filled)
                .map(Some)
                .ok_or_else(|| refused("raw coordinate")),
        }
    }
    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            Self::Borrowed { source, .. } => source.len(),
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Owned(v) => v.len(),
            Self::Borrowed { filled, .. } => *filled,
        }
    }
    pub(super) fn push(
        &mut self,
        name: Name<'h>,
        dimensions: Values<'h, u64>,
        ty: GgmlType,
        offset: u64,
    ) -> Result<()> {
        match self {
            Self::Owned(v) => v.push((owned(name)?, owned(dimensions)?, ty, offset)),
            Self::Borrowed { source, filled } => {
                let d = source.get(*filled).ok_or_else(|| refused("raw emission"))?;
                if name.as_str() != d.name
                    || dimensions.as_slice() != d.dimensions
                    || ty != d.ggml_type
                    || offset != d.relative_offset
                {
                    return Err(refused("raw descriptor identity"));
                }
                *filled += 1;
            }
        }
        Ok(())
    }
    pub(super) fn into_iter(self) -> RawIter<'h> {
        match self {
            Self::Owned(v) => RawIter::Owned(v.into_iter()),
            Self::Borrowed { source, filled } => RawIter::Borrowed(source[..filled].iter()),
        }
    }
}
pub(super) enum RawIter<'h> {
    Owned(std::vec::IntoIter<RawDescriptor>),
    Borrowed(std::slice::Iter<'h, TensorDescriptor>),
}
impl<'h> Iterator for RawIter<'h> {
    type Item = (Name<'h>, Values<'h, u64>, GgmlType, u64);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(v) => v
                .next()
                .map(|(n, d, t, o)| (Cow::Owned(n), Cow::Owned(d), t, o)),
            Self::Borrowed(v) => v.next().map(|d| {
                (
                    Cow::Borrowed(&d.name),
                    Cow::Borrowed(&d.dimensions),
                    d.ggml_type,
                    d.relative_offset,
                )
            }),
        }
    }
}

pub(super) enum Final<'h> {
    Owned(Vec<TensorDescriptor>),
    Borrowed {
        source: &'h Vec<TensorDescriptor>,
        filled: usize,
    },
}
impl<'h> Final<'h> {
    pub(super) fn new(count: usize, source: Option<&'h FinalHeader>) -> Result<Self> {
        if let Some(s) = source {
            if s.tensors.len() != count {
                return Err(refused("final population"));
            }
            Ok(Self::Borrowed {
                source: &s.tensors,
                filled: 0,
            })
        } else {
            Ok(Self::Owned(Vec::with_capacity(count)))
        }
    }
    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            Self::Borrowed { source, .. } => source.capacity(),
        }
    }
    pub(super) fn push(
        &mut self,
        name: Name<'h>,
        dimensions: Values<'h, u64>,
        ty: GgmlType,
        relative: u64,
        offset: u64,
        len: u64,
    ) -> Result<()> {
        match self {
            Self::Owned(v) => v.push(TensorDescriptor {
                name: owned(name)?,
                dimensions: owned(dimensions)?,
                ggml_type: ty,
                relative_offset: relative,
                data_offset: offset,
                byte_len: len,
            }),
            Self::Borrowed { source, filled } => {
                let d = source
                    .get(*filled)
                    .ok_or_else(|| refused("final emission"))?;
                if name.as_str() != d.name
                    || dimensions.as_slice() != d.dimensions
                    || ty != d.ggml_type
                    || relative != d.relative_offset
                    || offset != d.data_offset
                    || len != d.byte_len
                {
                    return Err(refused("final descriptor identity"));
                }
                *filled += 1;
            }
        }
        Ok(())
    }
    pub(super) fn as_slice(&self) -> &[TensorDescriptor] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed { source, filled } => &source[..*filled],
        }
    }
    pub(super) fn finish(self) -> Result<Cow<'h, Vec<TensorDescriptor>>> {
        match self {
            Self::Owned(v) => Ok(Cow::Owned(v)),
            Self::Borrowed { source, filled } => {
                if filled != source.len() {
                    return Err(refused("final descriptor completion"));
                }
                Ok(Cow::Borrowed(source))
            }
        }
    }
}
