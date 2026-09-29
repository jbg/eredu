use super::*;
use eredu_nn::ParameterId;
use eredu_runtime::{
    BoundedRowLookup, ParameterBank, RowEncoding, RowLookupBank, RowLookupError, RowLookupLimits,
    RowLookupProvider, RowLookupSpec,
};

#[derive(Default)]
struct Rows {
    calls: Vec<Vec<(ParameterBankKey, u64)>>,
    completed: usize,
    fail: bool,
}
impl ParameterBank<NumericBackend> for Rows {
    type Acquisition = Vec<(ParameterBankKey, u64)>;
    type Report = usize;
    type Error = Error;
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        (key.bank() == 2 && key.unit() == 7).then_some(8)
    }
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        _: &NumericContext,
    ) -> Result<Self::Acquisition, Error> {
        assert!(request.entries().len() <= 2);
        self.calls.push(request.entries().to_vec());
        Ok(request.entries().to_vec())
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
impl RowLookupBank<NumericBackend> for Rows {
    fn rows(
        &mut self,
        acquired: Self::Acquisition,
        _: &RowLookupSpec,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        if self.fail {
            return Err(Error::backend("injected row decode failure"));
        }
        // Do not encode global IDs as floats: adjacent values above 2^53 must
        // remain distinct. This fixture's small residue is its explicit payload.
        let data = acquired
            .iter()
            .flat_map(|(key, _)| {
                let x = (key.member() % 97) as f32;
                [x + 0.25, -x - 0.5]
            })
            .collect();
        let tensor = NumericTensor::new(vec![acquired.len() as i32, 2], data);
        self.complete(acquired, &tensor, context)?;
        Ok(tensor)
    }
}
fn spec() -> RowLookupSpec {
    RowLookupSpec {
        parameter: ParameterId::new("logical.table").unwrap(),
        bank: 2,
        unit: 7,
        rows: (1u64 << 54) + 10,
        dimensions: 2,
        encoding: RowEncoding::Dense,
        output_type: eredu_nn::TensorElementType::F32,
    }
}
fn limits() -> RowLookupLimits {
    RowLookupLimits {
        requests: 16,
        rows_per_acquisition: 2,
        acquisition_bytes: 16,
        host_bytes: 2048,
        output_bytes: 384,
    }
}
#[test]
fn bounded_rows_preserve_large_integers_duplicates_and_complete_each_chunk() {
    let spec = spec();
    let context = NumericContext::default();
    let mut provider = BoundedRowLookup::new(Rows::default(), spec.clone(), limits()).unwrap();
    let ids = [(1 << 54) + 1, 3, 1 << 54, 3, 0];
    let result = provider
        .lookup_rows(
            &spec,
            &ids,
            eredu_runtime::ParameterBankAccess::Bulk,
            &context,
        )
        .unwrap();
    let expected = ids
        .iter()
        .flat_map(|id| {
            let x = (id % 97) as f32;
            [x + 0.25, -x - 0.5]
        })
        .collect::<Vec<_>>();
    assert_eq!(result.data, expected);
    assert_eq!(provider.bank().completed, 2);
    assert_eq!(
        provider.bank().calls[0],
        [
            (ParameterBankKey::new(2, 7, 0), 1),
            (ParameterBankKey::new(2, 7, 3), 2)
        ]
    );
    assert_eq!(provider.bank().calls[1][0].0.member(), 1usize << 54);
    assert_eq!(provider.bank().calls[1][1].0.member(), (1usize << 54) + 1);
}
#[test]
fn malformed_rows_and_resource_limits_fail_before_acquisition() {
    let spec = spec();
    let context = NumericContext::default();
    for (ids, limits) in [
        (vec![spec.rows], limits()),
        (vec![0; 17], limits()),
        (
            vec![1, 2],
            RowLookupLimits {
                host_bytes: 255,
                ..limits()
            },
        ),
        (
            vec![1, 2],
            RowLookupLimits {
                output_bytes: 47,
                ..limits()
            },
        ),
        (
            vec![1, 2],
            RowLookupLimits {
                acquisition_bytes: 7,
                ..limits()
            },
        ),
    ] {
        let mut provider = BoundedRowLookup::new(Rows::default(), spec.clone(), limits).unwrap();
        assert!(provider
            .lookup_rows(
                &spec,
                &ids,
                eredu_runtime::ParameterBankAccess::Bulk,
                &context
            )
            .is_err());
        assert!(provider.bank().calls.is_empty());
    }
    let mut provider = BoundedRowLookup::new(
        Rows {
            fail: true,
            ..Rows::default()
        },
        spec.clone(),
        limits(),
    )
    .unwrap();
    let error = provider
        .lookup_rows(
            &spec,
            &[0, 1, 2],
            eredu_runtime::ParameterBankAccess::Incremental,
            &context,
        )
        .unwrap_err();
    assert!(matches!(error, RowLookupError::Backend(_)));
    assert_eq!(provider.bank().calls.len(), 1);
    assert_eq!(provider.bank().completed, 0);
}

#[test]
fn composed_rows_survive_erasure_bank_dispatch_and_failure_agreement() {
    use eredu_runtime::expert::AgreeingParameterProvider;
    use eredu_runtime::{
        ParameterProvider, ParameterProviders, ResidentExpertProvider, RoutedBankId,
        RoutedBankProviders,
    };
    let spec = spec();
    let context = NumericContext::default();
    let rows = BoundedRowLookup::new(Rows::default(), spec.clone(), limits()).unwrap();
    let combined = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows,
    };
    let mut banks =
        RoutedBankProviders::new([(RoutedBankId::new(42), Box::new(combined))]).unwrap();
    assert!(ParameterProvider::<NumericBackend>::has_row_parameter(
        &banks,
        &spec.parameter
    ));
    let mut votes = Vec::new();
    let mut agreement = AgreeingParameterProvider::new(&mut banks, |ok| {
        votes.push(ok);
        Ok(ok)
    });
    let result = ParameterProvider::<NumericBackend>::lookup_rows(
        &mut agreement,
        &spec,
        &[1, 2, 1],
        eredu_runtime::ParameterBankAccess::Bulk,
        &context,
    )
    .unwrap();
    assert_eq!(result.data, [1.25, -1.5, 2.25, -2.5, 1.25, -1.5]);
    assert!(ParameterProvider::<NumericBackend>::lookup_rows(
        &mut agreement,
        &spec,
        &[spec.rows],
        eredu_runtime::ParameterBankAccess::Bulk,
        &context
    )
    .is_err());
    drop(agreement);
    assert_eq!(votes, [true, false]);
    let mut rejected = AgreeingParameterProvider::new(&mut banks, |_| Ok(false));
    assert!(ParameterProvider::<NumericBackend>::lookup_rows(
        &mut rejected,
        &spec,
        &[2],
        eredu_runtime::ParameterBankAccess::Incremental,
        &context
    )
    .is_err());
}

#[test]
fn ngram_embedding_uses_row_provider_and_resumable_integer_state() {
    use eredu_architectures::qwen4_exp::ngram::{
        NGramEmbedding, NGramEmbeddingError, NGramEmbeddingSpec, NGramError, NGramHashSpec,
    };
    use eredu_runtime::{ParameterProviders, ResidentExpertProvider};
    let context = NumericContext::default();
    let hash = NGramHashSpec::new(
        1007,
        7,
        3,
        1,
        vec![9007199254740993, 9223372036854775783, -19531231231231],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap();
    let table = RowLookupSpec { rows: 24, ..spec() };
    let specification = NGramEmbeddingSpec::new(1007, 7, 3, 1, table.clone(), 5, 16).unwrap();
    let policy = LayerCachePolicy::fixed_only(vec![specification.history_policy()]).unwrap();
    let embedding = NGramEmbedding::new(specification, hash.clone()).unwrap();
    let fresh = || ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: BoundedRowLookup::new(Rows::default(), table.clone(), limits()).unwrap(),
    };
    let mut whole_state = NumericHybridLayerState::new(&policy);
    let mut whole_provider = fresh();
    let ids = [3u64, 7, 4, 5, 6, 7, 1, 2];
    let expected = embedding
        .forward::<NumericBackend, _, _>(
            Some(&ids),
            1,
            8,
            &mut whole_state,
            &mut whole_provider,
            eredu_runtime::ParameterBankAccess::Bulk,
            &context,
        )
        .unwrap();
    whole_state.advance_fixed(8).unwrap();
    assert_eq!(expected.shape, [1, 8, 4]);
    let expected_rows = hash.select(Some(&ids), 1, 8, &[7, 7], 16).unwrap();
    assert_eq!(
        expected.data,
        expected_rows
            .rows
            .iter()
            .flat_map(|row| [*row as f32 + 0.25, -(*row as f32) - 0.5])
            .collect::<Vec<_>>()
    );
    let mut state = NumericHybridLayerState::new(&policy);
    let mut provider = fresh();
    let mut decoded = Vec::new();
    for &id in &ids {
        let output = embedding
            .forward::<NumericBackend, _, _>(
                Some(&[id]),
                1,
                1,
                &mut state,
                &mut provider,
                eredu_runtime::ParameterBankAccess::Incremental,
                &context,
            )
            .unwrap();
        decoded.extend(output.data);
        state.advance_fixed(1).unwrap();
    }
    assert_eq!(decoded, expected.data);
    assert_eq!(state.position(), whole_state.position());
    let history = eredu_runtime::IntegerHistorySpec::new(5, 2, 7).unwrap();
    assert_eq!(
        history
            .read::<NumericBackend, _>(&mut state, 1, &context)
            .unwrap(),
        [1, 2]
    );
    let saved = state.clone();
    provider.rows.bank_mut().fail = true;
    assert!(embedding
        .forward::<NumericBackend, _, _>(
            Some(&[3]),
            1,
            1,
            &mut state,
            &mut provider,
            eredu_runtime::ParameterBankAccess::Incremental,
            &context
        )
        .is_err());
    assert_eq!(
        history
            .read::<NumericBackend, _>(&mut state, 1, &context)
            .unwrap(),
        [1, 2]
    );
    assert!(matches!(
        embedding.forward::<NumericBackend, _, _>(
            None,
            1,
            1,
            &mut state,
            &mut provider,
            eredu_runtime::ParameterBankAccess::Incremental,
            &context
        ),
        Err(NGramEmbeddingError::Hash(NGramError::MissingTokenIds))
    ));
    provider.rows.bank_mut().fail = false;
    embedding
        .forward::<NumericBackend, _, _>(
            Some(&[9]),
            1,
            1,
            &mut state,
            &mut provider,
            eredu_runtime::ParameterBankAccess::Incremental,
            &context,
        )
        .unwrap();
    state = saved;
    assert_eq!(
        history
            .read::<NumericBackend, _>(&mut state, 1, &context)
            .unwrap(),
        [1, 2]
    );
}

#[path = "row_lookup/lexical.rs"]
mod lexical;
