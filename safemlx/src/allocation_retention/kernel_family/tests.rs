use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
#[derive(Debug)]
struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn family_copies_six_sources_reuses_signatures_and_retires_after_last_native_alias() {
    let mut body = String::from("uint i=thread_position_in_grid.x; if(i>=threads_per_grid.x) return; output[i]=read_value(input,i)*GAIN;");
    let header = "inline float read_value(constant const float& x,uint i){return x;}\ninline float read_value(constant const float* x,uint i){return x[i];}\ninline float read_value(device const float* x,uint i){return x[i];}\n";
    let negative = [BorrowedKernelTemplate::Int(c"GAIN", -2)];
    let positive = [BorrowedKernelTemplate::Int(c"GAIN", 3)];
    let signature = |class, templates| KernelSpecialization {
        inputs: [KernelInputSignature {
            dtype: Dtype::Float32,
            class,
        }],
        outputs: [Dtype::Float32],
        templates,
    };
    let plan = MetalKernelFamilyPlan {
        definition: MetalKernelDefinitionPlan {
            name: "owned_source_family",
            source: &body,
            header,
            inputs: ["input"],
            outputs: ["output"],
            ensure_row_contiguous: true,
            atomic_outputs: false,
        },
        specializations: [
            signature(KernelInputClass::Scalar, &negative),
            signature(KernelInputClass::Small, &negative),
            signature(KernelInputClass::Device, &negative),
            signature(KernelInputClass::Scalar, &positive),
            signature(KernelInputClass::Small, &positive),
            signature(KernelInputClass::Device, &positive),
        ],
    };
    let qualified = plan.layout::<Owner>().is_ok();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_KERNEL_FAMILY").is_some() {
        assert!(qualified, "pinned finite source family must qualify");
    }
    if !qualified {
        return;
    }
    let retired = Arc::new(AtomicUsize::new(0));
    let family = plan.realize(Owner(retired.clone())).unwrap();
    // Duplicate signatures are refused by the same producer before any block
    // or node is published, and the original owner is returned intact.
    let duplicate = MetalKernelFamilyPlan {
        definition: plan.definition,
        specializations: [plan.specializations[0], plan.specializations[0]],
    };
    let rejected_retired = Arc::new(AtomicUsize::new(0));
    let refused = duplicate
        .realize(Owner(rejected_retired.clone()))
        .unwrap_err();
    assert_eq!(refused.cause(), KernelDefinitionCause::Invalid);
    assert_eq!(rejected_retired.load(Ordering::SeqCst), 0);
    drop(refused);
    assert_eq!(rejected_retired.load(Ordering::SeqCst), 1);
    // All native text and template names must have been copied synchronously.
    body.clear();
    body.push_str("invalid replacement source");
    drop(body);
    let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));
    let input = Array::try_from_slice(&[2.0f32], &[]).unwrap();
    assert!(family
        .apply_fixed_device(
            [&input],
            [BorrowedKernelOutput {
                shape: &[],
                dtype: Dtype::Float32
            }],
            &[BorrowedKernelTemplate::Int(c"GAIN", 7)],
            [1, 1, 1],
            [1, 1, 1],
            &stream
        )
        .is_err());
    let mut pending = Vec::new();
    for shape in [&[][..], &[3][..], &[8][..], &[11][..]] {
        let count = shape.iter().product::<i32>().max(1) as usize;
        let values: Vec<_> = (0..count).map(|i| i as f32 - 2.5).collect();
        let input = Array::try_from_slice(&values, shape).unwrap();
        for gain in [-2, 3] {
            let [output] = family
                .apply_fixed_device(
                    [&input],
                    [BorrowedKernelOutput {
                        shape,
                        dtype: Dtype::Float32,
                    }],
                    &[BorrowedKernelTemplate::Int(c"GAIN", gain)],
                    [count as i32, 1, 1],
                    [8, 1, 1],
                    &stream,
                )
                .unwrap();
            pending.push((
                output,
                values.iter().map(|v| *v * gain as f32).collect::<Vec<_>>(),
            ));
        }
    }
    drop(input);
    drop(family);
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    // Two different device lengths select the same declared slot. All six
    // slots compile lazily after the safe family and its source loans retire.
    for (actual, expected) in &pending {
        assert_eq!(crate::array::eval_vec::<f32>(actual), *expected);
    }
    drop(pending);
    // Evaluation completion can precede the worker's final local primitive
    // alias. Settle actual native work before asserting physical retirement.
    stream.synchronize().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while retired.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline {
        crate::try_retire_completed_submissions().unwrap();
        crate::reclaim_allocation_owners();
        std::thread::yield_now();
    }
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
