//! Row operator binding over the ordinary shared residency/completion manager.
use super::*;
#[path = "rows_memory.rs"]
mod memory;
use crate::backend::nn::native_quantization::{NativeQuantizationFormat, NativeQuantizedTensor};
use eredu_nn::{ParameterId, TensorElementType};
use eredu_runtime::{ParameterBank, PreparedRowLookup, RowEncoding, RowLookupBank, RowLookupSpec};

/// Exact row decoder and shared compact residency namespace. No vocabulary-sized
/// catalog or second cache is constructed here.
pub struct MlxRowBank {
    manager: ResidencyManager,
    range: eredu_runtime::RowResidencyRange,
    spec: RowLookupSpec,
    scale: Option<MlxTensor>,
    decode_bytes: u64,
}
impl MlxRowBank {
    /// Binds an already prepared row range and optional authoritative scalar.
    pub fn new(
        manager: ResidencyManager,
        prepared: &PreparedRowLookup,
        scale: Option<(ParameterId, MlxTensor)>,
        decode_bytes: u64,
    ) -> Result<Self, Error> {
        let range = preflight(&manager, prepared, decode_bytes)?;
        Self::from_preflight(manager, prepared, range, scale, decode_bytes)
    }
    fn from_preflight(
        manager: ResidencyManager,
        prepared: &PreparedRowLookup,
        range: eredu_runtime::RowResidencyRange,
        scale: Option<(ParameterId, MlxTensor)>,
        decode_bytes: u64,
    ) -> Result<Self, Error> {
        let spec = prepared.spec().clone();
        let scale_dtype = prepared
            .scale()
            .map(|scale| {
                scale
                    .recipe
                    .infer(scale.source.as_ref())
                    .map(|metadata| metadata.dtype)
            })
            .transpose()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        match (prepared.scale(), &scale) {
            (None, None) => {}
            (Some(expected), Some((id, value)))
                if id == &expected.parameter
                    && value.as_array().size() == 1
                    && scale_dtype.as_ref()
                        == Some(
                            &crate::backend::runtime::checkpoint::recipe::recipe_dtype_from_mlx(
                                value.as_array().dtype(),
                            ),
                        )
                    && matches!(
                        value.as_array().dtype(),
                        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                    ) => {}
            _ => {
                return Err(Error::ArchitectureModel(
                    "row decoder requires its exact prepared scalar companion".into(),
                ))
            }
        }
        Ok(Self {
            manager,
            range,
            spec,
            scale: scale.map(|(_, v)| v),
            decode_bytes,
        })
    }
}
impl ParameterBank<MlxNeuralBackend> for MlxRowBank {
    type Acquisition = AcquiredParameters;
    type Report = ResidencyReport;
    type Error = Error;
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        (key.bank() == self.spec.bank
            && key.unit() == self.spec.unit
            && (key.member() as u64) < self.spec.rows)
            .then_some(self.range.range().bytes())
    }
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        stream: &Stream,
    ) -> Result<AcquiredParameters, Error> {
        if request.entries().is_empty() {
            return Err(AddressableParameterBankError::EmptyDemand.into());
        }
        let pass = match request.access() {
            ParameterBankAccess::Bulk => BankAccessClass::Bulk,
            ParameterBankAccess::Incremental => BankAccessClass::Incremental,
            _ => {
                return Err(Error::ArchitectureModel(
                    "unknown row bank access class".into(),
                ))
            }
        };
        // Cold and runtime admission share the same bounded decoder geometry.
        let scratch = memory::workspace(
            request.entries().len() as u64,
            self.range.range().bytes(),
            &self.spec,
        )?;
        if scratch > self.decode_bytes {
            return Err(eredu_runtime::RowLookupError::Budget {
                resource: "native row decode scratch",
                required: scratch,
                limit: self.decode_bytes,
            }
            .into());
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut requests = Vec::with_capacity(request.entries().len());
        for &(key, demand) in request.entries() {
            if self.member_bytes(key).is_none() {
                return Err(
                    AddressableParameterBankError::MissingOwnedEntry { identity: key }.into(),
                );
            }
            if demand == 0 {
                return Err(AddressableParameterBankError::ZeroDemand { identity: key }.into());
            }
            if !seen.insert(key) {
                return Err(
                    AddressableParameterBankError::DuplicateDemand { identity: key }.into(),
                );
            }
            let id = self
                .range
                .range()
                .member_id(self.range.range().start() + key.member() as u64)
                .expect("validated row");
            requests.push((id, demand));
        }
        let transfer = self
            .manager
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        Ok(AcquiredParameters {
            identities: request.entries().iter().map(|(k, _)| *k).collect(),
            demand: request.entries().iter().map(|(_, n)| *n).collect(),
            scratch_bytes: scratch,
            pass,
            transfer,
        })
    }
    fn complete(
        &mut self,
        acquisition: AcquiredParameters,
        output: &MlxTensor,
        _: &Stream,
    ) -> Result<(), Error> {
        super::movement::complete_parameters(acquisition, |_| Ok(output.clone())).map(|_| ())
    }
    fn report(&self) -> Result<ResidencyReport, Error> {
        Ok(self.manager.report()?)
    }
}
impl RowLookupBank<MlxNeuralBackend> for MlxRowBank {
    fn rows(
        &mut self,
        acquisition: AcquiredParameters,
        spec: &RowLookupSpec,
        stream: &Stream,
    ) -> Result<MlxTensor, Error> {
        if spec != &self.spec {
            return Err(Error::ArchitectureModel(
                "row decoder specification differs from prepared bank".into(),
            ));
        }
        super::movement::complete_parameters(acquisition, |acquired| {
            let encoded = acquired.compact_binding(self.range.binding(), stream)?;
            let rows = i32::try_from(acquired.identities.len()).map_err(|_| {
                Error::ArchitectureModel("row acquisition exceeds native indexing".into())
            })?;
            let indices = Array::arange::<_, i32>(Some(0), rows, None::<i32>, stream)?;
            let decoded = match spec.encoding {
                RowEncoding::Dense => encoded,
                RowEncoding::ScalarE4M3 { .. } => {
                    encoded.from_fp8(Dtype::Float32, stream)?.multiply(
                        self.scale
                            .as_ref()
                            .expect("validated scale")
                            .as_array()
                            .as_dtype(Dtype::Float32, stream)?,
                        stream,
                    )?
                }
                RowEncoding::Gguf { encoding, endian } => {
                    if let Some(decoder) = scalar_decoder(&spec.encoding) {
                        // Only this already admitted compact acquisition is made
                        // host-visible. No encoded copy or conversion companion is
                        // allocated; output storage was reserved before acquisition.
                        eval([&encoded])?;
                        let evaluated = encoded.evaluated()?;
                        let bytes = evaluated.as_slice::<u8>();
                        let count = decoder
                            .output_len(bytes.len())
                            .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
                        if count != rows as usize * spec.dimensions as usize {
                            return Err(Error::ArchitectureModel(
                                "encoded row result geometry differs from admission".into(),
                            ));
                        }
                        let mut decoded = vec![0f32; count];
                        decoder
                            .decode_into(bytes, &mut decoded)
                            .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
                        Array::from_slice(&decoded, &[rows, spec.dimensions]).copy(stream)?
                    } else {
                        let matrix = NativeQuantizedTensor::from_iq_array(
                            encoded,
                            &[rows, spec.dimensions],
                            encoding,
                            endian,
                        )?;
                        matrix.embedding(&indices, stream)?
                    }
                }
            };
            let dtype = match spec.output_type {
                TensorElementType::F32 => Dtype::Float32,
                TensorElementType::F16 => Dtype::Float16,
                TensorElementType::Bf16 => Dtype::Bfloat16,
                _ => unreachable!("validated arithmetic"),
            };
            // MLX copy/contiguous may alias. Gather establishes independent
            // compact output even for one dense row, before source lease release.
            Ok(MlxTensor::from_array(
                decoded
                    .as_dtype(dtype, stream)?
                    .take_axis(&indices, 0, stream)?,
            ))
        })
    }
}

/// Checks the complete decoder binding before scalar reads or native allocation.
fn preflight(
    manager: &ResidencyManager,
    prepared: &PreparedRowLookup,
    decode_bytes: u64,
) -> Result<eredu_runtime::RowResidencyRange, Error> {
    let range = manager
        .row_range(prepared.range().range().prefix())
        .cloned()
        .ok_or_else(|| Error::ArchitectureModel("missing prepared row residency range".into()))?;
    if !range.same_binding(prepared.range()) {
        return Err(Error::ArchitectureModel(
            "row manager differs from retained binding source".into(),
        ));
    }
    let workspace = eredu_runtime::RowLookupMechanismSupport::workspace(
        &MlxRowLookupSupport,
        prepared.descriptor(),
    )?
    .ok_or_else(|| {
        Error::ArchitectureModel("row encoding or scalar recipe has no native decoder".into())
    })?;
    let required = prepared.maximum_acquisition_rows() * range.range().bytes();
    if let Some(limit) = manager.offload_config()?.device_budget_bytes() {
        if required > limit {
            return Err(eredu_runtime::RowLookupError::Budget {
                resource: "row acquisition residency",
                required,
                limit,
            }
            .into());
        }
    }
    if workspace.decode_bytes > decode_bytes {
        return Err(eredu_runtime::RowLookupError::Budget {
            resource: "native row decode scratch",
            required: workspace.decode_bytes,
            limit: decode_bytes,
        }
        .into());
    }
    Ok(range)
}

/// Exact, device-independent decoder and storage facts for retained row recipes.
pub struct MlxRowLookupSupport;

fn scalar_decoder(encoding: &RowEncoding) -> Option<eredu_gguf::BlockDecoder> {
    let RowEncoding::Gguf { encoding, endian } = *encoding else {
        return None;
    };
    if NativeQuantizationFormat::from_ggml_type(encoding).is_some() {
        return None;
    }
    eredu_gguf::BlockDecoder::new(encoding, endian).ok()
}

impl eredu_runtime::RowLookupMechanismSupport for MlxRowLookupSupport {
    fn storage(&self) -> Option<eredu_runtime::AddressableStorageCapabilities> {
        Some(
            eredu_runtime::AddressableStorageCapabilities::new(true, true, true, u64::MAX)
                .with_tiers(eredu_runtime::AddressableStorageTiers::new(
                    true, true, true,
                )),
        )
    }
    fn decode_memory(
        &self,
        prepared: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_runtime::RowLookupError>
    {
        memory::describe(prepared)
    }
    fn workspace(
        &self,
        prepared: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<eredu_runtime::RowLookupWorkspace>, eredu_runtime::RowLookupError> {
        if let RowEncoding::Gguf { encoding, .. } = prepared.spec().encoding {
            if NativeQuantizationFormat::from_ggml_type(encoding).is_none()
                && scalar_decoder(&prepared.spec().encoding).is_none()
            {
                return Ok(None);
            }
        }
        let decode_bytes = memory::workspace(
            prepared.maximum_acquisition_rows(),
            prepared.range().bytes(),
            prepared.spec(),
        )?;
        let scalar_bytes = if let Some(scale) = prepared.scale() {
            let recipe = scale.recipe();
            let Ok(workspace) =
                crate::backend::runtime::checkpoint::recipe::native_recipe_workspace(recipe, scale)
            else {
                return Ok(None);
            };
            if crate::backend::runtime::checkpoint::recipe::preflight_mlx_recipe_metadata(
                recipe,
                scale,
                &|key| {
                    scale
                        .catalog()
                        .get(key)
                        .map(|entry| entry.encoding.clone())
                        .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor {
                            key: key.into(),
                        })
                },
            )
            .is_err()
            {
                return Ok(None);
            }
            workspace
        } else {
            0
        };
        Ok(Some(eredu_runtime::RowLookupWorkspace {
            decode_bytes,
            scalar_bytes,
        }))
    }
}

impl SharedAddressableParameterBank {
    /// Reports the existing shared row pool without acquiring entries or reading sources.
    pub(crate) fn row_report(
        &self,
        requirements: eredu_runtime::SelectedRowLookupRequirements,
    ) -> Result<super::RowLookupPoolReport, Error> {
        let pool = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
        })?;
        Ok(super::RowLookupPoolReport {
            pool_id: pool.pool_id,
            residency: pool.manager.report()?,
            requirements,
        })
    }
}

type LocalRowProviders =
    eredu_runtime::RowLookupProviders<eredu_runtime::BoundedRowLookup<MlxRowBank>>;
type TensorRowProviders = eredu_runtime::RowLookupProviders<
    eredu_runtime::TensorParallelRowLookup<
        MlxNeuralBackend,
        eredu_runtime::BoundedRowLookup<MlxRowBank>,
        crate::backend::runtime::distributed::Group,
    >,
>;
enum RowProviders {
    Local(LocalRowProviders),
    Tensor(TensorRowProviders),
}
struct RowPool {
    id: u64,
    manager: ResidencyManager,
}

/// Selected row decoders over the shared parameter-bank pool. Tensor peers retain
/// only compact declarations and the collective group; pool reports exist on owners.
pub struct MlxRowLookups {
    providers: RowProviders,
    pool: Option<RowPool>,
    requirements: eredu_runtime::SelectedRowLookupRequirements,
}
impl MlxRowLookups {
    /// Binds exact retained sources. Every decoder and retained-scalar budget is
    /// checked before materializing a scalar. No table row is read here.
    pub fn bind_shared(
        pool: &SharedAddressableParameterBank,
        selected: &eredu_runtime::SelectedRowLookups,
        source_stream: &Stream,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let (pool, banks) = Self::bind_shared_banks(pool, selected, source_stream, stream)?;
        Ok(Self {
            providers: RowProviders::Local(selected.prepared().bind(banks)?),
            pool: Some(pool),
            requirements: selected.requirements(),
        })
    }

    /// Binds the exact native member after group realization. Only the declared
    /// owner supplies a pool and acquires scalar/source authority; peers supply None.
    pub(crate) fn bind_tensor_parallel(
        pool: Option<&SharedAddressableParameterBank>,
        selected: &eredu_runtime::SelectedRowLookups,
        parallel: crate::backend::runtime::distributed::Group,
        status_parallel: crate::backend::runtime::distributed::Group,
        rank: usize,
        owner: usize,
        source_stream: &Stream,
        stream: &Stream,
    ) -> Result<Self, Error> {
        if status_parallel.size() == 0
            || status_parallel.size() > i32::MAX as usize
            || status_parallel.rank() >= status_parallel.size()
            || rank != parallel.rank()
            || rank >= parallel.size()
            || owner >= parallel.size()
            || pool.is_some() != (rank == owner)
        {
            return Err(eredu_runtime::RowLookupError::Geometry.into());
        }
        let (pool, banks) = if let Some(pool) = pool {
            let (pool, banks) = Self::bind_shared_banks(pool, selected, source_stream, stream)?;
            (Some(pool), banks)
        } else {
            selected.validate_binding(selected.options().offload(), &MlxRowLookupSupport)?;
            (None, BTreeMap::new())
        };
        let providers = selected
            .prepared()
            .bind_tensor_parallel::<MlxNeuralBackend, _, _>(
                banks,
                parallel,
                status_parallel,
                rank,
                owner,
            )?;
        Ok(Self {
            providers: RowProviders::Tensor(providers),
            pool,
            requirements: selected.requirements(),
        })
    }

    fn bind_shared_banks(
        pool: &SharedAddressableParameterBank,
        selected: &eredu_runtime::SelectedRowLookups,
        source_stream: &Stream,
        stream: &Stream,
    ) -> Result<(RowPool, BTreeMap<ParameterId, MlxRowBank>), Error> {
        let pool = pool.inner.lock().map_err(|_| {
            Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
        })?;
        Self::bind_banks(&pool, selected, source_stream, stream)
    }

    // All source and mechanism checks precede the first scalar materialization.
    // Both ordinary and tensor owners use this one acquisition/completion path.
    fn bind_banks(
        pool: &AddressableParameterBank,
        selected: &eredu_runtime::SelectedRowLookups,
        source_stream: &Stream,
        stream: &Stream,
    ) -> Result<(RowPool, BTreeMap<ParameterId, MlxRowBank>), Error> {
        let manager = pool.residency_manager().clone();
        selected.validate_binding(manager.offload_config()?, &MlxRowLookupSupport)?;
        let prepared = selected.prepared();
        let mut scalar_bindings = BTreeMap::new();
        let mut ranges = BTreeMap::new();
        for (id, entry) in prepared.entries() {
            let range = preflight(
                &manager,
                entry,
                selected
                    .workspace(id)
                    .expect("selected exact row set")
                    .decode_bytes,
            )?;
            ranges.insert(id.clone(), range);
            if let Some(scale) = entry.scale() {
                let bytes = scale.recipe.infer(scale.source.as_ref())?.byte_len;
                let binding = WeightBinding::from_recipe(
                    scale.parameter.as_str(),
                    scale.recipe.clone(),
                    bytes,
                )?;
                eredu_runtime::preflight_bindings::<MlxNeuralBackend>(
                    scale.source.as_ref(),
                    std::slice::from_ref(&binding),
                )
                .map_err(|e| Error::ArchitectureModel(e.to_string()))?;
                scalar_bindings.insert(id.clone(), binding);
            }
        }
        let mut banks = BTreeMap::new();
        for (id, entry) in prepared.entries() {
            let scale = if let Some(scale) = entry.scale() {
                let binding = scalar_bindings.remove(id).expect("preflighted row scalar");
                let mut values =
                    crate::backend::runtime::checkpoint::binding::materialize_module_bindings(
                        scale.source.as_ref(),
                        std::slice::from_ref(&binding),
                        source_stream,
                        stream,
                    )?;
                let value = values
                    .remove(scale.parameter.as_str())
                    .expect("exact row scalar binding");
                Some((scale.parameter.clone(), MlxTensor::from_array(value)))
            } else {
                None
            };
            banks.insert(
                id.clone(),
                MlxRowBank::from_preflight(
                    manager.clone(),
                    entry,
                    ranges.remove(id).expect("preflighted row range"),
                    scale,
                    selected
                        .workspace(id)
                        .expect("selected exact row set")
                        .decode_bytes,
                )?,
            );
        }
        Ok((
            RowPool {
                id: pool.pool_id,
                manager,
            },
            banks,
        ))
    }

    /// Aggregate pool residency on its owner; tensor peers have no row pool.
    pub fn report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.pool
            .as_ref()
            .map(|pool| pool.manager.report().map_err(Error::from))
            .transpose()
    }
    /// Selected row facts and one physical pool identity, only on its owner.
    pub fn pool_report(&self) -> Result<Option<super::RowLookupPoolReport>, Error> {
        self.pool
            .as_ref()
            .map(|pool| {
                Ok(super::RowLookupPoolReport {
                    pool_id: pool.id,
                    residency: pool.manager.report()?,
                    requirements: self.requirements,
                })
            })
            .transpose()
    }
    /// Retained scalar and per-invocation workspace declarations.
    pub const fn requirements(&self) -> eredu_runtime::SelectedRowLookupRequirements {
        self.requirements
    }
}
impl eredu_runtime::RowLookupProvider<MlxNeuralBackend> for MlxRowLookups {
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        match &self.providers {
            RowProviders::Local(providers) => providers.has_row_parameter(parameter),
            RowProviders::Tensor(providers) => providers.has_row_parameter(parameter),
        }
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &Stream,
    ) -> Result<MlxTensor, eredu_runtime::RowLookupError> {
        match &mut self.providers {
            RowProviders::Local(providers) => providers.lookup_rows(spec, rows, access, context),
            RowProviders::Tensor(providers) => providers.lookup_rows(spec, rows, access, context),
        }
    }
}
