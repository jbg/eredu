//! Actual shared extraction on CPU; strict managed CPU admission stays closed.
use super::*;

#[test]
fn cpu_candidate_extraction_preserves_terminal_row_cast_ties_and_full_sort_backing() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for vocabulary in [1i32, 17, 2048, 2049] {
            for rows in [1i32, 5] {
                for count in [1, (vocabulary / 3).max(1), vocabulary] {
                    let mut values = vec![f32::INFINITY; (rows * vocabulary) as usize];
                    let start = ((rows - 1) * vocabulary) as usize;
                    for (i, value) in values[start..].iter_mut().enumerate() {
                        // Nonzero values plus representable ties and near ties
                        // that can collapse in the actual half conversion.
                        *value = 1.25 + (i % 23) as f32 * 0.25 + (i % 3) as f32 * 0.00003;
                    }
                    if vocabulary > 2 {
                        values[start] = -0.0;
                        values[start + 1] = 0.0;
                    }
                    let source = Array::from_slice(&values, &[1, rows, vocabulary])
                        .as_dtype(dtype, &stream)
                        .unwrap();
                    drop(source.evaluated().unwrap());
                    let source_info = source.allocation_info().unwrap().unwrap();
                    let program =
                        CandidateExtraction::borrowed(source.shape(), count as u64).unwrap();
                    program.validate_source(&source).unwrap();
                    CandidateExtraction::reset_test_counts();
                    let mut roots = Vec::with_capacity(CandidateExtraction::ROOTS);
                    let (ids, scores) = program
                        .execute(&source, &stream, |a| roots.push(a.clone()))
                        .unwrap();
                    assert_eq!(CandidateExtraction::test_counts(), (1, 0));
                    assert_eq!(roots.len(), CandidateExtraction::ROOTS);
                    drop(source);
                    let ids = ids.evaluated().unwrap();
                    let scores = scores.evaluated().unwrap();
                    assert_eq!(roots[0].allocation_info().unwrap(), Some(source_info));
                    let sorted = roots[5].allocation_info().unwrap().unwrap();
                    assert_eq!(roots[6].allocation_info().unwrap(), Some(sorted));
                    assert!(sorted.bytes() >= vocabulary as usize * 4);
                    if count < vocabulary {
                        assert!(sorted.bytes() > count as usize * 4);
                    }
                    let row = roots[1].evaluated().unwrap();
                    let row = row.as_slice::<f32>();
                    assert_eq!(row.len(), vocabulary as usize);
                    assert!(row.iter().all(|value| value.is_finite()));
                    let mut expected = (0..vocabulary as u32).collect::<Vec<_>>();
                    expected.sort_by(|&a, &b| {
                        row[a as usize]
                            .partial_cmp(&row[b as usize])
                            .unwrap()
                            .then(a.cmp(&b))
                    });
                    let expected = expected
                        .into_iter()
                        .rev()
                        .take(count as usize)
                        .map(|id| (id, row[id as usize].to_bits()))
                        .collect::<Vec<_>>();
                    let actual = ids
                        .as_slice::<u32>()
                        .iter()
                        .zip(scores.as_slice::<f32>())
                        .rev()
                        .map(|(&id, &score)| (id, score.to_bits()))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual, expected,
                        "dtype={dtype:?}, R={rows}, V={vocabulary}, K={count}"
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_candidate_nonfinite_and_geometry_errors_keep_actual_prefix_before_sort() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let source = Array::from_slice(&[9.0f32, 9.0, 9.0, 9.0, -0.0, 0.0, 3.0, value], &[1, 2, 4]);
        let program = CandidateExtraction::borrowed(source.shape(), 2).unwrap();
        let mut roots = Vec::with_capacity(CandidateExtraction::ROOTS);
        CandidateExtraction::reset_test_counts();
        let error = program
            .execute(&source, &stream, |a| roots.push(a.clone()))
            .unwrap_err();
        assert!(error.to_string().contains("finite raw logits"));
        assert_eq!(CandidateExtraction::test_counts(), (1, 0));
        assert_eq!(roots.len(), 5);
        drop(source);
        assert_eq!(roots[4].evaluated().unwrap().as_slice::<u32>(), &[3]);
    }
    assert!(CandidateExtraction::borrowed(&[1, 1, 0], 0).is_err());
    let program = CandidateExtraction::borrowed(&[1, 1, 4], 2).unwrap();
    let wrong = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3]);
    CandidateExtraction::reset_test_counts();
    let mut retained = 0;
    assert!(program.execute(&wrong, &stream, |_| retained += 1).is_err());
    assert_eq!(retained, 0);
    assert_eq!(CandidateExtraction::test_counts(), (0, 0));
}
