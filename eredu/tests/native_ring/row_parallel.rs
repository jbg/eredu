//! Shared row-parallel projection agrees local preparation before reducing.
#[path = "../support/component_coordinator.rs"]
mod component_coordinator;

use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    Error, LinearFormatSpec, LinearSpec, NeuralBackend, ParameterMetadata, ParameterSpec,
    ParameterVisitorMut, Parameterized, ProjectionInputObserver, Tensor,
};
use safemlx::{Array, Device, DeviceType, Stream};

const WORKER: &str = "EREDU_ROW_PARALLEL_AGREEMENT_WORKER";

fn tensor(values: &[f32], shape: &[i32]) -> MlxTensor {
    MlxTensor::from_array(Array::from_slice(values, shape))
}

fn linear(rank: usize, bias: bool, stream: &Stream) -> <Backend as NeuralBackend>::Linear {
    let input = if rank == 0 { 2 } else { 3 };
    let mut module = Backend::linear(
        LinearSpec {
            input,
            output: 2,
            weight: ParameterSpec::trainable("row.weight").unwrap(),
            bias: bias.then(|| ParameterSpec::trainable("row.bias").unwrap()),
            format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        },
        stream,
    )
    .unwrap();
    struct Bind {
        rank: usize,
    }
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
            *value = match metadata.id.as_str() {
                "row.weight" if self.rank == 0 => tensor(&[2., 3., -1., 4.], &[2, 2]),
                "row.weight" => tensor(&[1., -2., 0.5, 2., 1., -1.], &[2, 3]),
                "row.bias" => tensor(&[0.25, -0.5], &[2]),
                name => panic!("unexpected row projection parameter {name}"),
            };
        }
    }
    module.visit_parameters_mut(&mut Bind { rank });
    module
}

#[derive(Debug)]
struct ObserverFailure;
impl std::fmt::Display for ObserverFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("injected row projection observer failure")
    }
}
impl std::error::Error for ObserverFailure {}

struct Observer {
    fail: bool,
    shapes: Vec<Vec<i32>>,
}
impl ProjectionInputObserver<MlxTensor> for Observer {
    fn observe(&mut self, input: &MlxTensor) -> Result<(), Error> {
        self.shapes.push(input.shape().to_vec());
        if self.fail {
            Err(Error::backend_source(ObserverFailure))
        } else {
            Ok(())
        }
    }
    fn observe_generated(
        &mut self,
        _: &MlxTensor,
        _: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, Error>,
    ) -> Result<(), Error> {
        self.observe(&generate()?)
    }
}

fn assert_expected(output: &MlxTensor, stream: &Stream) {
    assert_eq!(output.shape(), &[1, 2]);
    // Rank 0 contributes [8,7], rank 1 [-2.5,5]. Bias is added once
    // after reduction, so doubling it or assuming equal local widths is visible.
    assert_eq!(output.to_f32_vec(stream).unwrap(), vec![5.75, 11.5]);
}

#[test]
fn native_row_parallel_agreement_worker() {
    if std::env::var_os(WORKER).is_none() {
        return;
    }
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), 2);
    let rank = world.rank();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    use eredu_runtime::replicated_execution::ReplicatedCoordinator as _;
    let mut coordinator = component_coordinator::coordinator(&world, &stream);
    assert_eq!(coordinator.is_owner(), rank == 1);
    let decision = || MlxTensor::from_array(Array::from_slice(&[rank as i32 + 7], &[1]));
    assert!(coordinator.publish(decision(), &stream).is_err());
    assert!(coordinator.agree([1; 8], true, &stream).unwrap());
    let published = coordinator.publish(decision(), &stream).unwrap();
    assert_eq!(
        published.as_array().evaluated().unwrap().as_slice::<i32>(),
        &[8]
    );

    let parallel = coordinator.parallel_context();
    let input = if rank == 0 {
        tensor(&[1., 2.], &[1, 2])
    } else {
        tensor(&[3., 4., 5.], &[1, 3])
    };
    let mut module = linear(rank, true, &stream);
    let mut observer = Observer {
        fail: false,
        shapes: Vec::new(),
    };
    let output = Backend::row_parallel_linear_with_input_observer(
        &mut module,
        &input,
        parallel,
        &stream,
        Some(&mut observer),
    )
    .unwrap();
    assert_expected(&output, &stream);
    assert_eq!(observer.shapes, vec![input.shape().to_vec()]);

    observer.fail = rank == 0;
    let error = Backend::row_parallel_linear_with_input_observer(
        &mut module,
        &input,
        parallel,
        &stream,
        Some(&mut observer),
    )
    .unwrap_err();
    if rank == 0 {
        let mut source: &(dyn std::error::Error + 'static) = &error;
        let mut found = false;
        loop {
            found |= source.downcast_ref::<ObserverFailure>().is_some();
            match source.source() {
                Some(next) => source = next,
                None => break,
            }
        }
        assert!(
            found,
            "the original typed observer error must remain the source"
        );
    }
    assert_eq!(observer.shapes.len(), 2);
    observer.fail = false;

    let malformed = tensor(&[1., 2., 3.], &[1, 3]);
    assert!(Backend::row_parallel_linear_with_input_observer(
        &mut module,
        if rank == 0 { &malformed } else { &input },
        parallel,
        &stream,
        None,
    )
    .is_err());

    // Both projections are valid locally, but different batch extents cannot
    // enter one all-reduce even though local feature widths may differ.
    let two_batches = tensor(&[1., 2., 1., 2.], &[2, 2]);
    assert!(Backend::row_parallel_linear_with_input_observer(
        &mut module,
        if rank == 0 { &two_batches } else { &input },
        parallel,
        &stream,
        None,
    )
    .is_err());

    // Rank 0's valid projection exceeds the retained reduction contract's
    // 8,192-element limit; its peer must stop before submitting its own sum.
    let oversized = tensor(&vec![1.; 4097 * 2], &[4097, 2]);
    let error = Backend::row_parallel_linear_with_input_observer(
        &mut module,
        if rank == 0 { &oversized } else { &input },
        parallel,
        &stream,
        None,
    )
    .unwrap_err();
    if rank == 0 {
        assert!(error.to_string().contains("contract"), "{error}");
    }

    let mut bias_mismatch = linear(rank, rank == 0, &stream);
    assert!(Backend::row_parallel_linear_with_input_observer(
        &mut bias_mismatch,
        &input,
        parallel,
        &stream,
        None,
    )
    .is_err());

    // Equal metadata can still fail after reduction: length-three bias cannot
    // broadcast over a two-column output. Both ranks agree that returned error.
    struct InvalidBias;
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for InvalidBias {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
            if metadata.id.as_str() == "row.bias" {
                *value = tensor(&[0.25, -0.5, 1.], &[3]);
            }
        }
    }
    let mut invalid_bias = linear(rank, true, &stream);
    invalid_bias.visit_parameters_mut(&mut InvalidBias);
    assert!(Backend::row_parallel_linear_with_input_observer(
        &mut invalid_bias,
        &input,
        parallel,
        &stream,
        None,
    )
    .is_err());

    // Agreed failures preserve this exact retained authority for a fresh call.
    let output = Backend::row_parallel_linear_with_input_observer(
        &mut module,
        &input,
        parallel,
        &stream,
        Some(&mut observer),
    )
    .unwrap();
    assert_expected(&output, &stream);
    assert_eq!(observer.shapes.len(), 3);
    let output = Backend::row_parallel_linear(&mut module, &input, parallel, &stream).unwrap();
    assert_expected(&output, &stream);
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_row_parallel_agreement_tp2_rejects_failures_and_retries() {
    super::workers::run(
        "row_parallel::native_row_parallel_agreement_worker",
        std::time::Duration::from_secs(90),
        |command, _rank| {
            command.env(WORKER, "1");
        },
    );
}
