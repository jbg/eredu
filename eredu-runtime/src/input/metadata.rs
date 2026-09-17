//! Shared prepared descriptor construction with explicit metadata destinations.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};

#[derive(Clone, Copy)]
pub(super) struct Destination<'a>(pub(super) Option<&'a WorkspaceContext>);
pub(super) enum Failure {
    Prepared(PreparedInputError),
    Metadata(Error),
}
impl Failure {
    pub(super) fn ordinary(self) -> PreparedInputError {
        match self {
            Self::Prepared(cause) => cause,
            Self::Metadata(_) => unreachable!("ordinary input has no metadata destination"),
        }
    }
    fn metadata(self, context: &WorkspaceContext) -> Error {
        match self {
            Self::Prepared(cause) => context.metadata_source(cause),
            Self::Metadata(cause) => cause,
        }
    }
}
impl Destination<'_> {
    fn controls<T>(self) -> Result<(), Failure> {
        if let Some(context) = self.0 {
            let controls = [
                std::mem::size_of::<T>(),
                std::mem::size_of::<Self>(),
                std::mem::size_of::<Failure>(),
                std::mem::size_of::<Result<T, Failure>>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or_else(|| Failure::Metadata(WorkspaceMetadataError::Overflow.into()))?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| Failure::Metadata(cause.into()))?;
        }
        Ok(())
    }
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Failure> {
        match self.0 {
            Some(context) => context.metadata_vec(count).map_err(Failure::Metadata),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    fn boxed<T>(self, values: Vec<T>) -> Result<Box<[T]>, Failure> {
        self.controls::<(Vec<T>, Box<[T]>)>()?;
        // Vec -> Box does not allocate when its complete capacity is occupied.
        // If the allocator supplied spare capacity, admit the actual shrink
        // request before conversion; the existing Vec remains live meanwhile.
        if values.capacity() != values.len() {
            if let Some(context) = self.0 {
                let bytes = std::alloc::Layout::array::<T>(values.len())
                    .map_err(|_| Failure::Metadata(WorkspaceMetadataError::Overflow.into()))?
                    .size();
                context
                    .charge_metadata(bytes)
                    .map_err(|cause| Failure::Metadata(cause.into()))?;
            }
        }
        Ok(values.into_boxed_slice())
    }
    fn map<K: Ord, V>(self, values: Vec<(K, V)>) -> Result<InputIdentityMap<K, V>, Failure> {
        let entries = self.boxed(values)?;
        Ok(InputIdentityMap::from_sorted_entries(entries)
            .unwrap_or_else(|_| unreachable!("validated source retains unique canonical keys")))
    }
}

pub(super) fn descriptor<T>(
    source: &PreparedInputPart<T>,
    destination: Destination<'_>,
    describe: &mut impl FnMut(&T) -> Result<InputTensorIdentity, Failure>,
) -> Result<InputPartDescriptor, Failure> {
    destination.controls::<(
        InputPartDescriptor,
        InputTensorIdentity,
        &PreparedInputPart<T>,
        Vec<(InputMetadataKey, InputTensorIdentity)>,
        Vec<(u32, InputExtent)>,
    )>()?;
    // All rows are reserved before calling any potentially allocating value
    // descriptor. Their keys and extents come from this validated source part.
    let mut metadata = destination.vector(source.metadata.len())?;
    let mut extents = destination.vector(source.extents.len())?;
    let payload = describe(source.payload.value())?;
    for (key, value) in &source.metadata {
        metadata.push((*key, describe(value)?));
    }
    for extent in &source.extents {
        extents.push((extent.identity_key(), *extent));
    }
    extents.sort_unstable_by_key(|(key, _)| *key);
    InputPartDescriptor::from_fixed_entries(
        source.modality,
        source.payload.kind(),
        payload,
        destination.map(metadata)?,
        destination.map(extents)?,
    )
    .map_err(Failure::Prepared)
}

pub(super) fn input<T>(
    parts: Vec<PreparedInputPart<T>>,
    destination: Destination<'_>,
    describe: &mut impl FnMut(&T) -> Result<InputTensorIdentity, Failure>,
) -> Result<PreparedModelInput<T>, Failure> {
    destination.controls::<(
        PreparedModelInput<T>,
        Vec<PreparedInputPart<T>>,
        PreparedInputIdentity,
        Vec<InputPartDescriptor>,
        InputPartDescriptor,
    )>()?;
    let mut descriptors = destination.vector(parts.len())?;
    for part in &parts {
        descriptors.push(descriptor(part, destination, describe)?);
    }
    let identity = PreparedInputIdentity::new(descriptors).map_err(Failure::Prepared)?;
    Ok(PreparedModelInput { parts, identity })
}

impl<T> PreparedInputPart<T> {
    /// Re-describes the actual borrowed values with the same descriptor worker,
    /// reserving rows before invoking the caller's paid identity constructor.
    /// The enclosing result/error owner must retain metadata funding.
    pub fn descriptor_with_metadata(
        &self,
        context: &WorkspaceContext,
        mut describe: impl FnMut(&T) -> Result<InputTensorIdentity, Error>,
    ) -> Result<InputPartDescriptor, Error> {
        descriptor(self, Destination(Some(context)), &mut |value| {
            describe(value).map_err(Failure::Metadata)
        }).map_err(|cause| cause.metadata(context))
    }

    /// Maps actual values from this validated part through an admitted metadata
    /// destination. Modality, keys, extents and payload kind are preserved;
    /// this creates no source, tensor or submission authority. The caller must
    /// retain independent funding through the returned part's retirement.
    pub fn map_with_metadata<'a, U>(
        &'a self,
        context: &WorkspaceContext,
        mut map: impl FnMut(&'a T) -> Result<U, Error>,
    ) -> Result<PreparedInputPart<U>, Error> {
        let destination = Destination(Some(context));
        let result = (|| {
            destination.controls::<(
                &Self,
                PreparedInputPart<U>,
                PreparedInputPayload<U>,
                Vec<(InputMetadataKey, U)>,
                Vec<InputExtent>,
            )>()?;
            let mut metadata = destination.vector(self.metadata.len())?;
            let mut extents = destination.vector(self.extents.len())?;
            let value = map(self.payload.value()).map_err(Failure::Metadata)?;
            let payload = match &self.payload {
                PreparedInputPayload::TokenIds(_) => PreparedInputPayload::TokenIds(value),
                PreparedInputPayload::Tensor(_) => PreparedInputPayload::Tensor(value),
                PreparedInputPayload::Embeddings(_) => PreparedInputPayload::Embeddings(value),
            };
            for (key, value) in &self.metadata {
                metadata.push((*key, map(value).map_err(Failure::Metadata)?));
            }
            extents.extend_from_slice(&self.extents);
            Ok(PreparedInputPart {
                modality: self.modality,
                payload,
                metadata: destination.map(metadata)?,
                extents,
            })
        })();
        result.map_err(|cause: Failure| cause.metadata(context))
    }
}

impl<T> PreparedModelInput<T> {
    /// Uses the ordinary descriptor/identity worker with counted storage.
    /// The caller's descriptor must pay its actual shape/name construction;
    /// the surrounding owner retains funding through this input's retirement.
    pub fn new_with_metadata(
        parts: Vec<PreparedInputPart<T>>,
        context: &WorkspaceContext,
        mut describe: impl FnMut(&T) -> Result<InputTensorIdentity, Error>,
    ) -> Result<Self, Error> {
        input(parts, Destination(Some(context)), &mut |value| {
            describe(value).map_err(Failure::Metadata)
        })
        .map_err(|cause| cause.metadata(context))
    }
}
