use super::*;
use eredu_nn::{
    multimodal::{MultiAxisRotaryLayout, RotaryAxisSpec},
    sequence_layout::{
        InterpolationMode, PatchAttentionWindows, PatchEncoderTableSpec, PatchPositionTableSpec,
        PatchTraversal,
    },
};
fn rows(source: &OriginalPreparedHostInput) -> impl Iterator<Item = (i32, i32, i32)> + Clone + '_ {
    let HostTensorValues::I32(values) = source.slot(2).unwrap().values else {
        panic!("actual grid slot")
    };
    values.chunks_exact(3).map(|r| (r[0], r[1], r[2]))
}
fn layout(source: &OriginalPreparedHostInput) -> PatchEncoderTableLayout {
    PatchEncoderTableLayout::new(
        PatchEncoderTableSpec {
            merge: 2,
            positions: PatchPositionTableSpec::Interpolated {
                height: 3,
                width: 3,
                mode: InterpolationMode::AlignCorners,
                traversal: PatchTraversal::MergeMajor(2),
            },
            windows: PatchAttentionWindows::Full,
            rotary_axes: [RotaryAxisSpec {
                dimensions: 4,
                position_offset: 0,
            }; 2],
            rotary_base: 10000.,
            rotary_minimum: 0,
            rotary_layout: MultiAxisRotaryLayout::SplitHalves,
        },
        rows(source),
        Some(4),
    )
    .unwrap()
}
#[derive(Default)]
struct Counts {
    calls: AtomicUsize,
    fills: AtomicUsize,
    lowers: AtomicUsize,
    prefixes: [AtomicUsize; 2],
}
struct EncoderPlan<'a> {
    source: &'a OriginalPreparedHostInput,
    counts: &'a Counts,
    fail_fill: bool,
}
impl EncoderPlan<'_> {
    fn recipe(&self) -> SourcePlan<'_> {
        SourcePlan::new(self.source)
            .unwrap()
            .with_encoder_tables(layout(self.source))
            .unwrap()
    }
}
impl Counts {
    fn plan<'a>(
        &'a self,
        source: &'a OriginalPreparedHostInput,
        fail_fill: bool,
    ) -> EncoderPlan<'a> {
        EncoderPlan {
            source,
            counts: self,
            fail_fill,
        }
    }
}
impl PreparedNativeInputCompiler for EncoderPlan<'_> {
    type Output = Output;
    type Error = PreparedModelInputSourceError<Failed>;
    fn source(&self) -> &OriginalPreparedHostInput {
        self.source
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        Ok(self.recipe().required_storage_bytes())
    }
    fn compile(self, owner: OriginalPreparedInputCustody) -> Result<Output, (Output, Self::Error)> {
        self.counts.calls.fetch_add(1, Ordering::SeqCst);
        let layout = layout(self.source);
        let result = self.recipe().construct_with_encoder(
            owner,
            |owner| Ok(Native(owner)),
            |_, slot, _| {
                self.counts.lowers.fetch_add(1, Ordering::SeqCst);
                Ok((Slot(slot), Slot(slot)))
            },
            |integers, floats| {
                self.counts.fills.fetch_add(1, Ordering::SeqCst);
                layout.fill(rows(self.source), integers, floats)?;
                if self.fail_fill {
                    Err(PatchEncoderTableError::SourceGeometry)
                } else {
                    Ok(())
                }
            },
        );
        if let Err((body, _)) = &result {
            let prefix = body.body().encoder_prefix.as_ref().unwrap();
            self.counts.prefixes[0].store(prefix.integers.capacity(), Ordering::SeqCst);
            self.counts.prefixes[1].store(prefix.floats.capacity(), Ordering::SeqCst);
        }
        result
    }
}
#[test]
fn original_encoder_full_recipe_compares_once_before_both_buffers_and_survives_final_view() {
    use crate::working_memory::WorkingMemoryPool;
    let seed = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&seed);
    let counts = Counts::default();
    let bytes =
        WorkingMemoryPool::prepared_native_input_required_bytes(&counts.plan(&i, false)).unwrap();
    let ibytes = i.original_bytes();
    let old_bytes = SourcePlan::new(&i).unwrap().required_storage_bytes();
    let table = layout(&i);
    assert!(bytes as usize >= old_bytes + 4 * (table.integer_count() + table.float_count()));
    drop(i);
    drop(seed);
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(ibytes + bytes - u64::from(short), 0).unwrap();
        let i = source(&pool);
        let counts = Counts::default();
        let result = pool.compile_prepared_native_input(counts.plan(&i, false));
        if short {
            let error = result.unwrap_err();
            assert_eq!(counts.calls.load(Ordering::SeqCst), 0);
            assert_eq!(counts.fills.load(Ordering::SeqCst), 0);
            assert_eq!(counts.lowers.load(Ordering::SeqCst), 0);
            assert_eq!(error.retained_bytes(), 0);
            assert!(
                matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded {
                required_bytes, available_bytes }) if *required_bytes == bytes && *available_bytes == bytes-1)
            );
            drop(error);
            drop(i);
        } else {
            let output = result.unwrap();
            assert_eq!(output.original_bytes(), bytes);
            assert_eq!(counts.calls.load(Ordering::SeqCst), 1);
            assert_eq!(counts.fills.load(Ordering::SeqCst), 1);
            assert_eq!(counts.lowers.load(Ordering::SeqCst), 3);
            let view = output.storage().prepared().unwrap().clone();
            let alias = view.clone();
            let a = view.original_encoder_tables().unwrap();
            let b = alias.original_encoder_tables().unwrap();
            assert_eq!(a.full_chunks(), [4]);
            assert_eq!(a.spatial_positions(), [0, 0, 0, 1, 1, 0, 1, 1]);
            assert!(a.learned_weights().iter().any(|x| *x > 0.));
            assert!(std::ptr::eq(
                a.learned_indices().as_ptr(),
                b.learned_indices().as_ptr()
            ));
            assert!(std::ptr::eq(
                a.rotary().frequencies().as_ptr(),
                b.rotary().frequencies().as_ptr()
            ));
            drop(output);
            drop(i);
            drop(view);
            assert_eq!(pool.used_bytes().unwrap(), ibytes + bytes);
            assert_eq!(
                alias
                    .original_encoder_tables()
                    .unwrap()
                    .rotary()
                    .frequencies(),
                [1., 0.01, 1., 0.01]
            );
            drop(alias);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn original_encoder_each_real_reserve_and_filled_error_retain_complete_prefix_until_error_drop() {
    use crate::working_memory::WorkingMemoryPool;
    for failure in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let i = source(&pool);
        let counts = Counts::default();
        let table = layout(&i);
        if failure < 2 {
            FAIL_HOST_RESERVATION.set(Some(failure));
        }
        let error = pool
            .compile_prepared_native_input(counts.plan(&i, failure == 2))
            .unwrap_err();
        assert_eq!(counts.calls.load(Ordering::SeqCst), 1);
        assert_eq!(counts.lowers.load(Ordering::SeqCst), 0);
        if failure < 2 {
            assert_eq!(FAIL_HOST_RESERVATION.get(), None);
            assert!(matches!(
                error.compiler_failure(),
                Some(PreparedModelInputSourceError::Allocation(_))
            ));
            assert_eq!(counts.fills.load(Ordering::SeqCst), 0);
        } else {
            assert!(matches!(
                error.compiler_failure(),
                Some(PreparedModelInputSourceError::Encoder(
                    PatchEncoderTableError::SourceGeometry
                ))
            ));
            assert_eq!(counts.fills.load(Ordering::SeqCst), 1);
        }
        assert_eq!(
            counts.prefixes[0].load(Ordering::SeqCst),
            if failure == 0 {
                0
            } else {
                table.integer_count()
            }
        );
        assert_eq!(
            counts.prefixes[1].load(Ordering::SeqCst),
            if failure == 2 { table.float_count() } else { 0 }
        );
        let b = error.retained_bytes();
        assert!(b > 0);
        let total = i.original_bytes() + b;
        drop(i);
        assert_eq!(pool.used_bytes().unwrap(), total);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_encoder_erased_and_typed_aliases_share_real_buffers_and_final_custody() {
    use crate::working_memory::WorkingMemoryPool;
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let i = source(&pool);
    let equal = source(&pool);
    assert_eq!(i.content_digest(), equal.content_digest());
    assert!(!i.same_source(&equal));
    let counts = Counts::default();
    let output = pool
        .compile_prepared_native_input(counts.plan(&i, false))
        .unwrap();
    let typed = output.storage().prepared().unwrap().clone();
    let erased = typed.original_encoder_projection().unwrap();
    assert!(erased.source().same_source(&i));
    assert!(!erased.source().same_source(&equal));
    assert_eq!(erased.identity(), typed.identity());
    assert!(std::ptr::eq(
        erased.tables().learned_weights().as_ptr(),
        typed
            .original_encoder_tables()
            .unwrap()
            .learned_weights()
            .as_ptr()
    ));
    let plain = pool
        .compile_prepared_native_input(plan(&equal, &counts.calls))
        .unwrap();
    assert!(plain
        .storage()
        .prepared()
        .unwrap()
        .original_encoder_projection()
        .is_none());
    let ordinary: PreparedModelInputOwner<Slot> =
        PreparedModelInput::new(typed.parts().to_vec(), |slot| {
            identity::<Failed>(i.slot(slot.0).unwrap()).map_err(|error| match error {
                PreparedModelInputSourceError::Descriptor(error) => error,
                _ => panic!("same valid original descriptors"),
            })
        })
        .unwrap()
        .into();
    assert!(ordinary.original_encoder_projection().is_none());
    drop(plain);
    drop(ordinary);
    drop(equal);
    let held = pool.used_bytes().unwrap();
    let mut owners = Vec::new();
    for _ in 0..4 {
        owners.push((typed.clone(), erased.clone()));
    }
    drop(output);
    drop(i);
    drop(typed);
    let barrier = Arc::new(std::sync::Barrier::new(owners.len() + 1));
    let workers = owners
        .into_iter()
        .map(|pair| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                drop(pair);
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert_eq!(erased.tables().full_chunks(), [4]);
    drop(erased);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
