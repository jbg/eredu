use super::*;
use crate::{Device, DeviceType};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
#[derive(Debug)]
struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        assert!(crate::utils::runtime_lock::can_reclaim_submission_resources());
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn prepared_stream_aliases_retain_one_owner_and_ordinary_clone_stays_independent() {
    let cpu = Device::new(DeviceType::Cpu, 0);
    let source = Stream::new_with_device(&cpu);
    let replacement = Stream::new_with_device(&cpu);
    let index = source.get_index().unwrap();
    let plan = StreamCopyPlan::capture(&source).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let copy = plan.realize(Owner(drops.clone())).unwrap();
    let alias = copy.clone();
    assert!(copy.same(&alias));
    assert_ne!(source.as_ptr().ctx, copy.as_stream().as_ptr().ctx);
    assert_eq!(copy.as_stream().get_index().unwrap(), index);
    let ordinary = copy.as_stream().clone();
    assert_ne!(ordinary.as_ptr().ctx, copy.as_stream().as_ptr().ctx);
    let mut handle = ordinary.as_ptr();
    // SAFETY: mutate only the independent ordinary clone's live C wrapper.
    assert_eq!(
        unsafe { safemlx_sys::mlx_stream_set(&mut handle, replacement.as_ptr()) },
        0
    );
    assert_eq!(
        ordinary.get_index().unwrap(),
        replacement.get_index().unwrap()
    );
    assert_eq!(copy.as_stream().get_index().unwrap(), index);
    drop((ordinary, source));
    let child = std::thread::spawn(move || {
        assert_eq!(alias.as_stream().get_index().unwrap(), index);
        let values = crate::Array::from_slice(&[2.0f32, 7.0], &[2]);
        let out = values.add(&values, alias.as_stream()).unwrap();
        assert_eq!(out.evaluated().unwrap().as_slice::<f32>(), &[4.0, 14.0]);
        alias
    })
    .join()
    .unwrap();
    drop(copy);
    super::super::reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    {
        let _loan = crate::utils::runtime_lock::enter();
        drop(child);
        assert_eq!(super::super::reclaim_allocation_owners(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    super::super::reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
#[test]
fn stream_scalar_plan_outlives_source_and_invalid_snapshot_preserves_outputs() {
    let source = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let index = source.get_index().unwrap();
    let plan = StreamCopyPlan::capture(&source).unwrap();
    assert!(plan.native_wrapper_bytes() > 0 && plan.control_bytes().unwrap() > 0);
    drop(source);
    let copy = plan.realize(()).unwrap();
    assert_eq!(copy.as_stream().get_index().unwrap(), index);
    let mut sentinel = safemlx_sys::mlx_stream_copy_value {
        index: 11,
        device_index: 13,
        device_kind: 17,
        cpu_matmul: 19,
    };
    // SAFETY: valid output and explicit null input; this fixed-status API rejects it.
    assert_eq!(
        unsafe {
            safemlx_sys::mlx_stream_copy_snapshot(
                &mut sentinel,
                safemlx_sys::mlx_stream {
                    ctx: ptr::null_mut(),
                },
            )
        },
        2
    );
    assert_eq!(
        (sentinel.index, sentinel.device_index, sentinel.device_kind, sentinel.cpu_matmul),
        (11, 13, 17, 19)
    );
    drop(copy);
    super::super::reclaim_allocation_owners();
}

#[test]
fn selected_cpu_matmul_copy_preserves_context_choice_and_owner_retirement() {
    let facts=CpuMatmulFacts::inspect().unwrap();
    assert_eq!(facts.tile_edge(),16);assert!(facts.reduction_lanes()>0);
    let stream=Stream::new_with_device(&Device::new(DeviceType::Cpu,0));
    let physical=stream.get_index().unwrap();
    let plan=StreamCopyPlan::<Owner>::capture(&stream).unwrap();
    assert_eq!(plan.cpu_matmul(),CpuMatmulKernel::PlatformDefault);
    let selected=plan.with_cpu_matmul(CpuMatmulKernel::Float32Tiles).unwrap();
    assert!(!selected.matches_source(&stream));
    let retired=Arc::new(AtomicUsize::new(0));
    let copy=selected.realize(Owner(retired.clone())).unwrap();
    assert_eq!(copy.as_stream().get_index().unwrap(),physical);
    assert_eq!(copy.as_stream(),&stream); // physical queue identity is unchanged
    assert!(selected.matches_source(copy.as_stream()));
    let retained=copy.clone();drop((copy,stream));
    assert_eq!(retired.load(Ordering::SeqCst),0);
    assert_eq!(StreamCopyPlan::<()>::capture(retained.as_stream()).unwrap().cpu_matmul(),CpuMatmulKernel::Float32Tiles);
    drop(retained);super::super::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst),1);
}
