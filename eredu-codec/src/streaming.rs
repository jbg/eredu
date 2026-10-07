//! Shared causal convolution and overlap-add state for neural audio codecs.
//!
//! Operators accept `[batch, channels, time]`, retain only their receptive-field
//! carry. Parameter-free causal state can be cloned independently of kernels;
//! module wrappers pair carry with learned parameters. Completion is explicit:
//! convolution pads an incomplete final stride, while transposed
//! convolution uses the architecture's declared tail policy.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{Index, PadMode, Parameter, ParameterSpec, Parameterized, Tensor};

use crate::Error;

/// State of a finite streaming layer invocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StreamPhase {
    /// More chunks can be submitted.
    #[default]
    AcceptingInput,
    /// Final carry has been handled; reset starts a new invocation.
    Finished,
}

impl StreamPhase {
    fn require_input(self) -> Result<(), Error> {
        match self {
            Self::AcceptingInput => Ok(()),
            Self::Finished => Err(Error::StreamFinished),
        }
    }
}

/// Causal strided convolution geometry in backend-neutral kernel layout.
#[derive(Debug, Clone, Copy)]
pub struct Conv1dSpec {
    /// Number of input channels.
    pub input_channels: i32,
    /// Number of output channels.
    pub output_channels: i32,
    /// Kernel taps before dilation.
    pub kernel_size: i32,
    /// Input samples consumed per output sample.
    pub stride: i32,
    /// Distance between kernel taps.
    pub dilation: i32,
    /// Independent channel groups.
    pub groups: i32,
    /// Whether the checkpoint has an output bias.
    pub bias: bool,
    /// Initial left padding and final incomplete-stride padding.
    pub pad_mode: PadMode,
}

impl Conv1dSpec {
    /// Ordinary biased, undilated, ungrouped causal convolution.
    pub const fn causal(
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        stride: i32,
    ) -> Self {
        Self {
            input_channels,
            output_channels,
            kernel_size,
            stride,
            dilation: 1,
            groups: 1,
            bias: true,
            pad_mode: PadMode::Constant,
        }
    }

    fn effective_kernel(self) -> Result<i32, Error> {
        validate_channels(self.input_channels, self.output_channels, self.groups)?;
        if self.kernel_size <= 0 || self.stride <= 0 || self.dilation <= 0 {
            return Err(invalid(
                "convolution kernel, stride and dilation must be positive",
            ));
        }
        let kernel = (self.kernel_size - 1)
            .checked_mul(self.dilation)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("convolution receptive field exceeds i32"))?;
        if kernel < self.stride {
            return Err(invalid(
                "causal convolution requires receptive field >= stride",
            ));
        }
        Ok(kernel)
    }
}

/// Parameter-free causal convolution carry and finite-stream lifecycle.
///
/// Canonical family parameters can be transformed into the shared execution
/// layout for each invocation without retaining learned weights in snapshots.
#[derive(Debug, Clone)]
pub struct Conv1dState<T: Tensor> {
    spec: Conv1dSpec,
    effective_kernel: i32,
    carry: Option<T>,
    remainder: i32,
    started: bool,
    phase: StreamPhase,
}

impl<T: Tensor + SnapshotValue> SnapshotValue for Conv1dState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.carry.estimate_snapshot_value()])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            carry: self.carry.copy_snapshot_value(retention)?,
            spec: self.spec,
            effective_kernel: self.effective_kernel,
            remainder: self.remainder,
            started: self.started,
            phase: self.phase,
        })
    }
}

impl<T: Tensor> RetainedTensorRoots<T> for Conv1dState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        if let Some(carry) = &self.carry {
            visitor(carry);
        }
    }
}

impl<T: Tensor> Conv1dState<T> {
    /// Validates geometry without allocating a tensor or learned parameter.
    pub fn new(spec: Conv1dSpec) -> Result<Self, Error> {
        Ok(Self {
            spec,
            effective_kernel: spec.effective_kernel()?,
            carry: None,
            remainder: 0,
            started: false,
            phase: StreamPhase::AcceptingInput,
        })
    }
    /// Retained causal input, shaped `[batch, input_channels, retained_samples]`.
    pub fn carry(&self) -> Option<&T> {
        self.carry.as_ref()
    }
    /// Geometry to which this state belongs.
    pub const fn spec(&self) -> Conv1dSpec {
        self.spec
    }
    /// Current finite-stream lifecycle phase.
    pub const fn phase(&self) -> StreamPhase {
        self.phase
    }

    /// Number of input samples retained per channel, bounded by the receptive field.
    pub fn retained_samples(&self) -> i32 {
        self.carry.as_ref().map_or(0, |x| x.dim(2))
    }

    /// Discards prior carry and opens a new finite stream.
    pub fn reset_state(&mut self) {
        self.carry = None;
        self.remainder = 0;
        self.started = false;
        self.phase = StreamPhase::AcceptingInput;
    }

    /// Evaluates an ordinary finite input without changing streaming state.
    pub fn forward(
        &self,
        x: &T,
        weight: &T,
        bias: Option<&T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        self.validate_input(x, weight, bias)?;
        if x.dim(2) == 0 {
            return empty_output(x, self.spec.output_channels, context);
        }
        let extra = (self.spec.stride - x.dim(2) % self.spec.stride) % self.spec.stride;
        let padded = pad_bct(
            x,
            self.effective_kernel - self.spec.stride,
            extra,
            self.spec.pad_mode,
            context,
        )?;
        self.forward_unpadded(&padded, weight, bias, context)
    }

    /// Consumes an arbitrary input chunk, returning complete output strides.
    /// Empty chunks do not initialize padding. Failed computations leave carry unchanged.
    pub fn step(
        &mut self,
        x: &T,
        weight: &T,
        bias: Option<&T>,
        context: &T::Context,
    ) -> Result<Option<T>, Error> {
        self.phase.require_input()?;
        self.validate_input(x, weight, bias)?;
        if let Some(carry) = &self.carry {
            if carry.dim(0) != x.dim(0) {
                return Err(invalid("streaming convolution batch changed before reset"));
            }
        }
        if x.dim(2) == 0 {
            return Ok(None);
        }
        let padded = if self.started {
            x.clone()
        } else {
            pad_bct(
                x,
                self.effective_kernel - self.spec.stride,
                0,
                self.spec.pad_mode,
                context,
            )?
        };
        let input = match &self.carry {
            Some(previous) => T::concatenate(&[previous.clone(), padded], 2, context)?,
            None => padded,
        };
        let length = input.dim(2);
        let frames = if length < self.effective_kernel {
            0
        } else {
            (length - self.effective_kernel) / self.spec.stride + 1
        };
        let consumed = frames * self.spec.stride;
        let next_carry = input
            .index(
                &[Index::Full, Index::Full, Index::Range(consumed, length)],
                context,
            )?
            .compact(context)?;
        let output = if frames == 0 {
            None
        } else {
            let used = (frames - 1) * self.spec.stride + self.effective_kernel;
            let input = input.index(&[Index::Full, Index::Full, Index::Range(0, used)], context)?;
            Some(self.forward_unpadded(&input, weight, bias, context)?)
        };
        self.carry = Some(next_carry);
        self.remainder = ((i64::from(self.remainder) + i64::from(x.dim(2)))
            % i64::from(self.spec.stride)) as i32;
        self.started = true;
        Ok(output)
    }

    /// Pads the final incomplete stride with the declared padding mode.
    /// Repeated completion returns no output; further chunks require reset.
    pub fn finish(
        &mut self,
        weight: &T,
        bias: Option<&T>,
        context: &T::Context,
    ) -> Result<Option<T>, Error> {
        if self.phase == StreamPhase::Finished {
            return Ok(None);
        }
        let output = if self.remainder == 0 {
            None
        } else {
            let carry = self
                .carry
                .as_ref()
                .ok_or_else(|| invalid("missing convolution carry"))?;
            let padded = pad_bct(
                carry,
                0,
                self.spec.stride - self.remainder,
                self.spec.pad_mode,
                context,
            )?;
            Some(self.forward_unpadded(&padded, weight, bias, context)?)
        };
        self.carry = None;
        self.remainder = 0;
        self.phase = StreamPhase::Finished;
        Ok(output)
    }

    fn validate_input(&self, input: &T, weight: &T, bias: Option<&T>) -> Result<(), Error> {
        validate_input(input, self.spec.input_channels)?;
        if weight.shape()
            != [
                self.spec.output_channels,
                self.spec.kernel_size,
                self.spec.input_channels / self.spec.groups,
            ]
            || bias.is_some() != self.spec.bias
            || bias.is_some_and(|bias| bias.shape() != [self.spec.output_channels])
        {
            return Err(invalid(
                "convolution parameters differ from declared geometry",
            ));
        }
        Ok(())
    }

    fn forward_unpadded(
        &self,
        x: &T,
        weight: &T,
        bias: Option<&T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        let x = x.swap_axes(1, 2, context)?;
        let mut y = T::conv1d(
            &x,
            weight,
            self.spec.stride,
            0,
            self.spec.dilation,
            self.spec.groups,
            context,
        )?;
        if let Some(bias) = bias {
            y = y.add(bias, context)?;
        }
        Ok(y.swap_axes(1, 2, context)?)
    }
}

/// Causal convolution with bounded input carry and explicit finite-stream drain.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct StreamableConv1d<T: Tensor> {
    /// Kernel shaped `[output_channels, kernel_size, input_channels/groups]`.
    pub weight: Parameter<T>,
    /// Optional output-channel bias.
    pub bias: Option<Parameter<T>>,
    #[parameter(skip)]
    state: Conv1dState<T>,
}

impl<T: Tensor> StreamableConv1d<T> {
    /// Declares unloaded parameters without allocating weight payloads.
    pub fn unloaded(spec: Conv1dSpec, context: &T::Context) -> Result<Self, Error> {
        let state = Conv1dState::new(spec)?;
        Ok(Self {
            weight: parameter(
                "weight",
                &[
                    spec.output_channels,
                    spec.kernel_size,
                    spec.input_channels / spec.groups,
                ],
                context,
            )?,
            bias: spec
                .bias
                .then(|| parameter("bias", &[spec.output_channels], context))
                .transpose()?,
            state,
        })
    }
    /// Current finite-stream lifecycle phase.
    pub const fn phase(&self) -> StreamPhase {
        self.state.phase()
    }
    /// Number of retained input samples per channel.
    pub fn retained_samples(&self) -> i32 {
        self.state.retained_samples()
    }
    /// Discards prior carry and opens a new finite stream.
    pub fn reset_state(&mut self) {
        self.state.reset_state();
    }
    /// Evaluates an ordinary finite input without changing streaming state.
    pub fn forward(&self, x: &T, context: &T::Context) -> Result<T, Error> {
        self.state.forward(
            x,
            self.weight.as_ref(),
            self.bias.as_ref().map(Parameter::as_ref),
            context,
        )
    }
    /// Consumes complete strides. Failed computations leave carry unchanged.
    pub fn step(&mut self, x: &T, context: &T::Context) -> Result<Option<T>, Error> {
        self.state.step(
            x,
            self.weight.as_ref(),
            self.bias.as_ref().map(Parameter::as_ref),
            context,
        )
    }
    /// Pads a final incomplete stride and completes this finite invocation once.
    pub fn finish(&mut self, context: &T::Context) -> Result<Option<T>, Error> {
        self.state.finish(
            self.weight.as_ref(),
            self.bias.as_ref().map(Parameter::as_ref),
            context,
        )
    }
}

/// Architecture policy for the final transposed-convolution overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeTail {
    /// Discard the final `kernel_size - stride` samples, as in Mimi.
    Trim,
    /// Publish the final overlap, matching an ordinary untrimmed convolution.
    Emit,
}

/// Transposed convolution geometry and finite-stream tail policy.
#[derive(Debug, Clone, Copy)]
pub struct ConvTranspose1dSpec {
    /// Number of input channels.
    pub input_channels: i32,
    /// Number of output channels.
    pub output_channels: i32,
    /// Kernel width.
    pub kernel_size: i32,
    /// Output samples emitted per input sample before final tail handling.
    pub stride: i32,
    /// Independent channel groups.
    pub groups: i32,
    /// Whether the checkpoint has an output bias.
    pub bias: bool,
    /// Treatment of the final overlap.
    pub tail: TransposeTail,
}

/// Transposed convolution with bounded overlap-add carry.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct StreamableConvTranspose1d<T: Tensor> {
    /// Kernel shaped `[output_channels, kernel_size, input_channels/groups]`.
    pub weight: Parameter<T>,
    /// Optional output-channel bias, counted once across overlapping chunks.
    pub bias: Option<Parameter<T>>,
    #[parameter(skip)]
    spec: ConvTranspose1dSpec,
    #[parameter(skip)]
    carry: Option<T>,
    #[parameter(skip)]
    phase: StreamPhase,
}

impl<T: Tensor> StreamableConvTranspose1d<T> {
    /// Declares unloaded parameters without allocating weight payloads.
    pub fn unloaded(spec: ConvTranspose1dSpec, context: &T::Context) -> Result<Self, Error> {
        validate_channels(spec.input_channels, spec.output_channels, spec.groups)?;
        if spec.stride <= 0 || spec.kernel_size < spec.stride {
            return Err(invalid(
                "transposed convolution requires kernel >= positive stride",
            ));
        }
        Ok(Self {
            weight: parameter(
                "weight",
                &[
                    spec.output_channels,
                    spec.kernel_size,
                    spec.input_channels / spec.groups,
                ],
                context,
            )?,
            bias: spec
                .bias
                .then(|| parameter("bias", &[spec.output_channels], context))
                .transpose()?,
            spec,
            carry: None,
            phase: StreamPhase::AcceptingInput,
        })
    }

    /// Current finite-stream lifecycle phase.
    pub const fn phase(&self) -> StreamPhase {
        self.phase
    }

    /// Number of retained output samples per channel.
    pub fn retained_samples(&self) -> i32 {
        self.carry.as_ref().map_or(0, |y| y.dim(2))
    }

    /// Discards prior overlap and opens a new finite stream.
    pub fn reset_state(&mut self) {
        self.carry = None;
        self.phase = StreamPhase::AcceptingInput;
    }

    /// Evaluates an ordinary finite input under the declared tail policy.
    pub fn forward(&self, x: &T, context: &T::Context) -> Result<T, Error> {
        self.validate_input(x)?;
        if x.dim(2) == 0 {
            return empty_output(x, self.spec.output_channels, context);
        }
        let y = self.forward_untrimmed(x, context)?;
        match self.spec.tail {
            TransposeTail::Emit => Ok(y),
            TransposeTail::Trim => Ok(y.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(0, y.dim(2) - (self.spec.kernel_size - self.spec.stride)),
                ],
                context,
            )?),
        }
    }

    /// Consumes one chunk and emits its stable output prefix.
    /// Failed computations preserve overlap; empty chunks do not advance it.
    pub fn step(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        self.phase.require_input()?;
        self.validate_input(x)?;
        if let Some(carry) = &self.carry {
            if carry.dim(0) != x.dim(0) {
                return Err(invalid("transposed convolution batch changed before reset"));
            }
        }
        if x.dim(2) == 0 {
            return empty_output(x, self.spec.output_channels, context);
        }
        let y = self.forward_untrimmed(x, context)?;
        let length = y.dim(2);
        let y = match &self.carry {
            None => y,
            Some(previous) => {
                let previous = match &self.bias {
                    None => previous.clone(),
                    Some(bias) => previous.subtract(
                        &bias
                            .as_ref()
                            .reshape(&[1, self.spec.output_channels, 1], context)?,
                        context,
                    )?,
                };
                let overlap = previous.dim(2);
                let first = y
                    .index(
                        &[Index::Full, Index::Full, Index::Range(0, overlap)],
                        context,
                    )?
                    .add(&previous, context)?;
                let rest = y.index(
                    &[Index::Full, Index::Full, Index::Range(overlap, length)],
                    context,
                )?;
                T::concatenate(&[first, rest], 2, context)?
            }
        };
        let split = length - (self.spec.kernel_size - self.spec.stride);
        let output = y.index(&[Index::Full, Index::Full, Index::Range(0, split)], context)?;
        let carry = y
            .index(
                &[Index::Full, Index::Full, Index::Range(split, length)],
                context,
            )?
            .compact(context)?;
        self.carry = Some(carry);
        Ok(output)
    }

    /// Handles the final overlap once according to the architecture's tail policy.
    pub fn finish(&mut self, _: &T::Context) -> Result<Option<T>, Error> {
        if self.phase == StreamPhase::Finished {
            return Ok(None);
        }
        let tail = self
            .carry
            .take()
            .filter(|tail| self.spec.tail == TransposeTail::Emit && tail.dim(2) > 0);
        self.phase = StreamPhase::Finished;
        Ok(tail)
    }

    fn validate_input(&self, input: &T) -> Result<(), Error> {
        validate_input(input, self.spec.input_channels)?;
        if self.weight.as_ref().shape()
            != [
                self.spec.output_channels,
                self.spec.kernel_size,
                self.spec.input_channels / self.spec.groups,
            ]
            || self.bias.is_some() != self.spec.bias
            || self
                .bias
                .as_ref()
                .is_some_and(|bias| bias.as_ref().shape() != [self.spec.output_channels])
        {
            return Err(invalid(
                "transposed convolution parameters differ from declared geometry",
            ));
        }
        Ok(())
    }

    fn forward_untrimmed(&self, x: &T, context: &T::Context) -> Result<T, Error> {
        let x = x.swap_axes(1, 2, context)?;
        let mut y = T::conv_transpose1d(
            &x,
            self.weight.as_ref(),
            self.spec.stride,
            0,
            1,
            0,
            self.spec.groups,
            context,
        )?;
        if let Some(bias) = &self.bias {
            y = y.add(bias.as_ref(), context)?;
        }
        Ok(y.swap_axes(1, 2, context)?)
    }
}

fn invalid(message: &str) -> Error {
    Error::InvalidShape(message.into())
}

fn validate_channels(input: i32, output: i32, groups: i32) -> Result<(), Error> {
    if input <= 0 || output <= 0 || groups <= 0 || input % groups != 0 || output % groups != 0 {
        return Err(invalid(
            "convolution channels must be positive multiples of positive groups",
        ));
    }
    Ok(())
}

fn validate_input<T: Tensor>(input: &T, channels: i32) -> Result<(), Error> {
    if !matches!(input.shape(), [batch, width, time] if *batch > 0 && *width == channels && *time >= 0)
    {
        return Err(invalid(
            "streaming convolution expects [positive batch, declared channels, nonnegative time]",
        ));
    }
    Ok(())
}

fn parameter<T: Tensor>(
    name: &str,
    shape: &[i32],
    context: &T::Context,
) -> Result<Parameter<T>, eredu_nn::Error> {
    Parameter::unloaded(
        ParameterSpec::trainable(name).expect("codec parameter names are non-empty"),
        shape,
        context,
    )
}

fn empty_output<T: Tensor>(input: &T, channels: i32, context: &T::Context) -> Result<T, Error> {
    // Every requested shape has zero elements. Reshaping the prototype retains
    // its dtype without requiring an additional broadcasting mechanism.
    Ok(input.reshape(&[input.dim(0), channels, 0], context)?)
}

fn pad_bct<T: Tensor>(
    input: &T,
    left: i32,
    right: i32,
    mode: PadMode,
    context: &T::Context,
) -> Result<T, Error> {
    Ok(T::pad(
        input,
        &[(0, 0), (0, 0), (left, right)],
        mode,
        context,
    )?)
}
