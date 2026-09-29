//! Source-backed numeric rows shared by session conformance fixtures.
use super::*;
use eredu_nn::{DistributedNeuralBackend, ParameterId};
use eredu_runtime::*;
pub(super) struct SourceRows {
    spec: RowLookupSpec,
    range: RowResidencyRange,
    rows_per_acquisition: usize,
    completed: usize,
}
impl ParameterBank<NumericBackend> for SourceRows {
    type Acquisition = Vec<Vec<u8>>;
    type Report = usize;
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
        _: &NumericContext,
    ) -> Result<Self::Acquisition, Error> {
        assert!(request.entries().len() <= self.rows_per_acquisition);
        request
            .entries()
            .iter()
            .map(|(key, _)| {
                assert!(self.member_bytes(*key).is_some());
                let id = self.range.range().member_id(key.member() as u64).unwrap();
                let unit = self.range.unit(&id).unwrap().unwrap();
                let read = unit.bindings()[0]
                    .source_recipe()
                    .prepare_encoded_read(self.range.source())
                    .unwrap()
                    .unwrap();
                let mut bytes = vec![0; self.range.range().bytes() as usize];
                read.read_into(&mut bytes).unwrap();
                Ok(bytes)
            })
            .collect()
    }
    fn complete(
        &mut self,
        _: Self::Acquisition,
        _: &NumericTensor,
        _: &NumericContext,
    ) -> Result<(), Error> {
        self.completed += 1;
        Ok(())
    }
    fn report(&self) -> Result<usize, Error> {
        Ok(self.completed)
    }
}
impl RowLookupBank<NumericBackend> for SourceRows {
    fn rows(
        &mut self,
        acquired: Self::Acquisition,
        spec: &RowLookupSpec,
        ctx: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let output = NumericTensor::new(
            [acquired.len() as i32, spec.dimensions],
            acquired
                .iter()
                .flat_map(|row| {
                    row.chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                })
                .collect(),
        );
        self.complete(acquired, &output, ctx)?;
        Ok(output)
    }
}
impl SourceRows {
    pub(super) fn new(entry: &PreparedRowLookup) -> Self {
        assert_eq!(entry.spec().encoding, RowEncoding::Dense);
        assert_eq!(
            entry.range().metadata().dtype,
            eredu_checkpoint::recipe::RecipeDtype::F32
        );
        Self {
            range: entry.range().clone(),
            spec: entry.spec().clone(),
            rows_per_acquisition: entry.limits().rows_per_acquisition,
            completed: 0,
        }
    }
    pub(super) fn completed(&self) -> usize {
        self.completed
    }
}

pub(super) enum PartitionRows {
    Local(RowLookupProviders<BoundedRowLookup<SourceRows>>),
    Parallel(
        RowLookupProviders<
            TensorParallelRowLookup<
                NumericBackend,
                BoundedRowLookup<SourceRows>,
                NumericParallelContext,
            >,
        >,
    ),
}
impl RowLookupProvider<NumericBackend> for PartitionRows {
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        match self {
            Self::Local(provider) => provider.has_row_parameter(parameter),
            Self::Parallel(provider) => provider.has_row_parameter(parameter),
        }
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &NumericContext,
    ) -> Result<NumericTensor, RowLookupError> {
        match self {
            Self::Local(provider) => provider.lookup_rows(spec, rows, access, context),
            Self::Parallel(provider) => provider.lookup_rows(spec, rows, access, context),
        }
    }
}

pub(super) fn bind_partition_rows(
    rows: Option<SelectedRowLookups>,
    owner: Option<usize>,
    parallel: Option<&NumericParallelContext>,
    status_parallel: Option<&NumericParallelContext>,
) -> Result<Option<PartitionRows>, RowLookupError> {
    let Some(rows) = rows else { return Ok(None) };
    let owner = owner.ok_or(RowLookupError::Geometry)?;
    // EP-only ranks still join the wave status; their row data remains local.
    let singleton = (parallel.is_none() && status_parallel.is_some())
        .then(|| NumericParallelContext::new(0, NumericParallelGroup::new(1)));
    let parallel = parallel.or(singleton.as_ref());
    let rank = parallel.map(NumericBackend::parallel_rank).unwrap_or(0);
    let banks = rows
        .prepared()
        .entries()
        .iter()
        .filter(|_| rank == owner)
        .map(|(id, entry)| (id.clone(), SourceRows::new(entry)))
        .collect();
    match parallel {
        Some(parallel) => rows
            .prepared()
            .bind_tensor_parallel::<NumericBackend, _, _>(
                banks,
                parallel.clone(),
                status_parallel.unwrap_or(parallel).clone(),
                rank,
                owner,
            )
            .map(PartitionRows::Parallel)
            .map(Some),
        None if owner == 0 && status_parallel.is_none() => rows
            .prepared()
            .bind(banks)
            .map(PartitionRows::Local)
            .map(Some),
        None => Err(RowLookupError::Geometry),
    }
}
