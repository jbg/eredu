//! Function transforms
//!
//! This mod provides functions for automatic differentiation and other
//! transformations on functions.
//!
//! **WARN**: Because function transforms including compilation works on
//! the computation graph, the user must ensure that all `Array`s are passed
//! as inputs to the function/closure. Closures with captured `Array`s may
//! not work as expected and may lead to undefined behavior.
//!
//! # Automatic Differentiation
//!
//! Automatic differentiation in MLX works on functions rather than on implicit
//! graphs.
//!
//! MLX function transformations replace graph-mutation APIs such as
//! `backward`, `zero_grad`, and `detach`, and do not use a `requires_grad`
//! property.
//!
//! You can use the [`grad()`] and [`value_and_grad()`] function to compute
//! gradients of more complex functions. These functions compute the gradient
//! with respect to the first argument, in order to manually specify the the
//! argument to compute the gradient with respect to, use
//! [`grad_with_argnums()`] or [`value_and_grad_with_argnums()`].
//!
//! ```rust,ignore
//! use safemlx::{Array, error::Result, transforms::grad, square};
//!
//! fn f(x: &Array, stream: &safemlx::Stream) -> Result<Array> {
//!     square!(x, stream=stream)
//! }
//!
//! fn calculate_grad(
//!     mut func: impl FnMut(&Array, &safemlx::Stream) -> Result<Array>,
//!     arg: &Array,
//!     stream: &safemlx::Stream,
//! ) -> Result<Array> {
//!     grad(move |x: &Array| func(x, stream))(arg)
//! }
//!
//! let x = Array::from(1.5);
//!
//! let dfdx = calculate_grad(f, &x, &stream).unwrap();
//! assert_eq!(dfdx.item::<f32>(&stream), 2.0 * 1.5);
//!
//! let dfdx2 = calculate_grad(|arg, stream| calculate_grad(f, arg, stream), &x, &stream).unwrap();
//! assert_eq!(dfdx2.item::<f32>(&stream), 2.0);
//! ```

use safemlx_sys::mlx_closure_value_and_grad;

use crate::{
    error::{get_and_clear_closure_error, Result},
    module::ModuleParamRef,
    utils::{guard::Guarded, runtime_lock, Closure, VectorArray, SUCCESS},
    Array, Event, Stream, TimedEvaluation,
};

pub mod compile;
mod grad;
mod keyed_value_and_grad;
mod transform_guard;
mod value_and_grad;

pub use grad::*;
pub use keyed_value_and_grad::*;
pub use value_and_grad::*;

/// Return `inputs` unchanged while making them depend on `dependencies`.
///
/// This sequences lazy side-effecting primitives while retaining every array
/// handle in the resulting MLX graph. At least one input is required.
pub fn depends<'a, 'b>(
    inputs: impl IntoIterator<Item = &'a Array>,
    dependencies: impl IntoIterator<Item = &'b Array>,
) -> Result<Vec<Array>> {
    let mut inputs = inputs.into_iter().peekable();
    if inputs.peek().is_none() {
        return Err(crate::error::Exception::custom(
            "depends requires at least one input",
        ));
    }
    let inputs = VectorArray::try_from_iter(inputs)?;
    let dependencies = VectorArray::try_from_iter(dependencies.into_iter())?;
    let _guard = runtime_lock::enter();
    <Vec<Array> as Guarded>::try_from_op(|result| unsafe {
        safemlx_sys::mlx_depends(result, inputs.as_ptr(), dependencies.as_ptr())
    })
}

/// Evaluate an iterator of [`Array`]s.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    let _guard = runtime_lock::enter();
    <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_eval(vec.as_ptr()) })
}

/// Evaluate a module's parameters.
///
/// This is a convenience function that flattens the parameters and evaluates them.
pub fn eval_params(params: ModuleParamRef<'_>) -> Result<()> {
    eval(params.flatten().values().copied())
}

/// Asynchronously evaluate an iterator of [`Array`]s.
///
/// Please note that this is not a rust async function.
pub fn async_eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    let _guard = runtime_lock::enter();
    <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_async_eval(vec.as_ptr()) })
}

/// Submit evaluation of `outputs` and return its completion [`Event`].
///
/// This function reconciles events with MLX's lazy execution model: it is this
/// call, not construction of the output graphs, which submits their work. The
/// event covers every dependency required to materialize the outputs. Empty
/// sets and sets whose outputs are already fully available produce an
/// already-complete event with no device identity.
///
/// The returned single-shot event may be queried or host-waited repeatedly and
/// may order multiple consumer streams on the same logical device. Use
/// [`Stream::wait_event`](crate::Stream::wait_event) before submitting a
/// consumer graph to create a backend-side dependency.
pub fn async_eval_with_event<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<Event> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    let _guard = runtime_lock::enter();
    Event::try_from_op(|event| unsafe {
        safemlx_sys::mlx_async_eval_with_event(event, vec.as_ptr())
    })
}

/// Submit `outputs` asynchronously and measure their execution timeline.
///
/// Unlike timing Rust graph construction, this function records a timestamp,
/// submits the selected lazy MLX graph, and records the ending timestamp on
/// `stream`. The call does not wait for device execution. Resolve the duration
/// later with [`TimedEvaluation::try_elapsed`] or [`TimedEvaluation::elapsed`].
///
/// The graph's selected output stream must exactly equal `stream`; a different
/// stream or device is rejected before submission. Dependencies may execute on
/// other streams and are honored without a host wait. Unrelated work queued on
/// `stream` before the starting marker is excluded. Graphs large enough to use
/// multiple command buffers remain inside the measured phase; see
/// [`TimedEvaluation`] for backend-specific treatment of waits and idle gaps.
///
/// Ordinary [`async_eval`] and [`async_eval_with_event`] do not allocate or
/// record timestamp resources, so timing-disabled execution retains its normal
/// path and overhead.
pub fn async_eval_timed<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    stream: impl AsRef<Stream>,
) -> Result<TimedEvaluation> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    let stream = stream.as_ref();
    let _guard = runtime_lock::enter();
    Event::try_from_op(|event| unsafe {
        safemlx_sys::mlx_async_eval_timed(event, vec.as_ptr(), stream.c_stream)
    })
    .map(TimedEvaluation::from_completion)
}

/// Asynchronously evaluate a module's parameters.
///
/// This is a convenience function that flattens the parameters and evaluates them.
pub fn async_eval_params(params: ModuleParamRef<'_>) -> Result<()> {
    async_eval(params.flatten().values().copied())
}

#[inline]
fn jvp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let c_primals = VectorArray::try_from_iter(primals.iter())?;
    let c_tangents = VectorArray::try_from_iter(tangents.iter())?;
    let _transform_guard = transform_guard::enter();

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        safemlx_sys::mlx_jvp(
            res_0,
            res_1,
            closure.as_ptr(),
            c_primals.as_ptr(),
            c_tangents.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

/// Compute the Jacobian-vector product.
///
/// This computes the product of the Jacobian of a function `f` evaluated at
/// `primals` with the `tangents`.
///
/// # Params:
///
/// - `f`: function which takes an array of `Array` and returns an array of
///   `Array`
/// - `primals`: array of `Array` at which to evaluate the Jacobian
/// - `tangents`: array of `Array` which are the "vector" in the Jacobian-vector
///   product.  The `tangents` should be the same in number, shape and type as
///   the inputs of `f`, e.g. the `primals`
///
/// # Returns:
///
/// Array of the Jacobian-vector products which is the same in number, shape and
/// type of the outputs of `f`
pub fn jvp<'a, F>(f: F, primals: &[Array], tangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    jvp_inner(closure, primals, tangents)
}

/// Similar to [`jvp`] but handles closures that can return an error.
pub fn fallible_jvp<'a, F>(
    f: F,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    jvp_inner(closure, primals, tangents)
}

#[inline]
fn vjp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let c_primals = VectorArray::try_from_iter(primals.iter())?;
    let c_cotangents = VectorArray::try_from_iter(cotangents.iter())?;
    let _transform_guard = transform_guard::enter();

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        safemlx_sys::mlx_vjp(
            res_0,
            res_1,
            closure.as_ptr(),
            c_primals.as_ptr(),
            c_cotangents.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

/// Compute the vector-Jacobian product.
///
/// Computes the product of the `cotangents` with the Jacobian of a function `f` evaluated at
/// `primals`.
///
/// # Params:
///
/// - f: function which takes an array of `Array` and returns an array of `Array`
/// - primals: array of `Array` at which to evaluate the Jacobian
/// - cotangents: array of `Array` which are the "vector" in the vector-Jacobian product. The
///   `cotangents` should be the same in number, shape and type as the outputs of `f`
///
/// # Returns:
///
/// array of the vector-Jacobian products which is the same in number, shape and type of the outputs
/// of `f`
pub fn vjp<'a, F>(f: F, primals: &[Array], cotangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    vjp_inner(closure, primals, cotangents)
}

/// Similar to [`vjp`] but handles closures that can return an error.
pub fn fallible_vjp<'a, F>(
    f: F,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    vjp_inner(closure, primals, cotangents)
}

pub(crate) struct ClosureValueAndGrad {
    pub(crate) c_closure_value_and_grad: mlx_closure_value_and_grad,
}

impl ClosureValueAndGrad {
    pub fn as_ptr(&self) -> mlx_closure_value_and_grad {
        self.c_closure_value_and_grad
    }
}

impl Drop for ClosureValueAndGrad {
    fn drop(&mut self) {
        let status =
            unsafe { safemlx_sys::mlx_closure_value_and_grad_free(self.c_closure_value_and_grad) };
        debug_assert_eq!(status, SUCCESS);
    }
}

fn value_and_gradient(
    value_and_grad: mlx_closure_value_and_grad,
    arrays: impl Iterator<Item = impl AsRef<Array>>,
) -> Result<(Vec<Array>, Vec<Array>)> {
    let input_vector = VectorArray::try_from_iter(arrays)?;
    let _transform_guard = transform_guard::enter();

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        safemlx_sys::mlx_closure_value_and_grad_apply(
            res_0,
            res_1,
            value_and_grad,
            input_vector.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

#[cfg(test)]
mod tests {

    use crate::{
        array,
        transforms::{jvp, vjp},
        Array,
    };

    use super::*;

    // The unit tests below are adapted from the mlx c++ codebase

    #[test]
    fn test_jvp() {
        let stream = crate::test_stream();
        let f = move |inputs: &[Array]| -> Vec<Array> {
            vec![inputs[0].add(&inputs[1], stream).unwrap()]
        };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].clone().item::<f32>(&stream), 2.0f32);
        assert_eq!(dout[0].clone().item::<f32>(&stream), 4.0f32);
    }

    #[test]
    fn test_jvp_with_error() {
        let stream = crate::test_stream();
        let f = move |inputs: &[Array]| -> Result<Vec<Array>> {
            inputs[0].add(&inputs[1], stream).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = fallible_jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].clone().item::<f32>(&stream), 2.0f32);
        assert_eq!(dout[0].clone().item::<f32>(&stream), 4.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = fallible_jvp(f, &[a, b], &[array!(1.0f32), array!(3.0f32)]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }

    #[test]
    fn test_vjp() {
        let stream = crate::test_stream();
        let f = move |inputs: &[Array]| -> Vec<Array> {
            vec![inputs[0].add(&inputs[1], stream).unwrap()]
        };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].clone().item::<f32>(&stream), 2.0f32);
        assert_eq!(dout[0].clone().item::<f32>(&stream), 1.0f32);
    }

    #[test]
    fn test_vjp_with_error() {
        let stream = crate::test_stream();
        let f = move |inputs: &[Array]| -> Result<Vec<Array>> {
            inputs[0].add(&inputs[1], stream).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = fallible_vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].clone().item::<f32>(&stream), 2.0f32);
        assert_eq!(dout[0].clone().item::<f32>(&stream), 1.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = fallible_vjp(f, &[a, b], &[array!(1.0f32)]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }

    #[test]
    fn async_eval_cpu_streams_are_concurrent_safe() {
        std::thread::scope(|scope| {
            for _ in 0..crate::test_concurrency() {
                scope.spawn(|| {
                    for _ in 0..64 {
                        let stream = crate::Stream::new_with_device(&crate::Device::new(
                            crate::DeviceType::Cpu,
                            0,
                        ));
                        let x = crate::Array::zeros::<f32>(&[1], &stream).unwrap();
                        async_eval([&x]).unwrap();
                        x.evaluated().unwrap();
                    }
                });
            }
        });
    }
}
