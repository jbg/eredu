use safemlx::{
    error::Exception,
    transforms::compile::{
        clear_cache, compile_with_stream_and_captures, CompileOptions,
        CompiledWithStreamAndCaptures,
    },
    Array, Device, DeviceType, Stream,
};
use std::cell::RefCell;

type Trace = fn(&[Array], &[Array], &Stream) -> Result<Vec<Array>, Exception>;

fn square_plus_input(
    inputs: &[Array],
    _captures: &[Array],
    stream: &Stream,
) -> Result<Vec<Array>, Exception> {
    Ok(vec![inputs[0]
        .multiply(&inputs[0], stream)?
        .add(&inputs[0], stream)?])
}

thread_local! {
    static GRAPH: RefCell<CompiledWithStreamAndCaptures<Trace>> = RefCell::new(
        compile_with_stream_and_captures(square_plus_input as Trace, [], CompileOptions::new())
    );
}

#[test]
fn compiled_wrapper_survives_foreign_tls_teardown_and_lazy_output_outlives_cache() {
    std::thread::spawn(|| {
        // Construct Rust TLS before compilation initializes C++ TLS, forcing
        // the native cache to be destroyed before this wrapper on thread exit.
        GRAPH.with(|_| {});
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let lazy = GRAPH.with(|slot| {
            slot.borrow_mut()
                .call(&[Array::from_f32(3.)], &stream)
                .unwrap()
                .remove(0)
        });
        clear_cache().unwrap();
        assert_eq!(lazy.try_item::<f32>(&stream).unwrap(), 12.);
        // Leave a populated cache for native destruction before Rust TLS drop.
        let result = GRAPH.with(|slot| {
            slot.borrow_mut()
                .call(&[Array::from_f32(4.)], &stream)
                .unwrap()
                .remove(0)
        });
        assert_eq!(result.try_item::<f32>(&stream).unwrap(), 20.);
    })
    .join()
    .unwrap();
}

#[test]
fn scalar_constants_distinguish_same_shape_compiled_graphs() {
    // This factory produces the same callback type/cache identity for both
    // offsets. The supplied scalar constants must distinguish their graphs.
    fn offset_graph(
        offset: f32,
    ) -> CompiledWithStreamAndCaptures<
        impl FnMut(&[Array], &[Array], &Stream) -> Result<Vec<Array>, Exception>,
    > {
        compile_with_stream_and_captures(
            move |inputs: &[Array], _: &[Array], stream: &Stream| {
                Ok(vec![inputs[0].add(Array::from_f32(offset), stream)?])
            },
            [],
            CompileOptions::new().constant(offset.to_bits() as u64),
        )
    }
    std::thread::spawn(|| {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut first = offset_graph(1.);
        let mut second = offset_graph(2.);
        let input = [Array::from_f32(6.)];
        for (use_second, expected) in [(false, 7.), (true, 8.), (false, 7.)] {
            let graph = if use_second { &mut second } else { &mut first };
            let result = graph.call(&input, &stream).unwrap().remove(0);
            assert_eq!(result.try_item::<f32>(&stream).unwrap(), expected);
        }
    })
    .join()
    .unwrap();
}
