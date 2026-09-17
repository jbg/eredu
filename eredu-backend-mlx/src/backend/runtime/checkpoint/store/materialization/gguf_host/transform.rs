//! Reached typed transformations with exact input/prefix failure ownership.
use super::*;

impl PreparedGgufHostCopy {
    pub(super) fn bind_plan(&mut self, descriptor: &eredu_gguf::TensorDescriptor) {
        self.bind_plan_view(descriptor.view());
    }
    pub(super) fn bind_plan_view(&mut self, descriptor: eredu_gguf::TensorDescriptorView<'_>) {
        let lease = self.source.lease();
        let plan = lease.conversion_plan();
        self.source_plan_matches = plan.as_ref().is_ok_and(|plan| {
            plan.matches_lease(lease)
                && plan.conversion().descriptor().view() == descriptor
                && self
                    .allocation_source
                    .is_none_or(|p| p == std::ptr::from_ref(lease) as usize)
        });
        // Do not return this error ahead of the existing name, shape or payload
        // width validation. The first reached transform/producer consumes it.
        self.source_plan = Some(plan);
    }
    pub(super) fn output_request(
        &mut self,
        dtype: LogicalDtype,
    ) -> Result<TypedOutputRequest, GgufHostCopyCause> {
        if self.source_plan.as_ref().is_some_and(Result::is_err) {
            let Some(Err(cause)) = self.source_plan.take() else {
                unreachable!()
            };
            return Err(GgufHostCopyCause::SourcePlan(cause));
        }
        let Some(Ok(plan)) = self.source_plan.as_ref() else {
            return Err(GgufHostCopyCause::TypedBinding);
        };
        if !self.source_plan_matches {
            return Err(GgufHostCopyCause::TypedBinding);
        }
        let request = TypedOutputRequest::for_output(plan.conversion(), self.produced)
            .ok_or(GgufHostCopyCause::TypedBinding)?;
        if request.dtype != dtype {
            return Err(GgufHostCopyCause::TypedBinding);
        }
        Ok(request)
    }

    pub(super) fn transform_bytes<
        T: safemlx::ArrayElement + Copy + Send + 'static,
        const N: usize,
    >(
        &mut self,
        bytes: impl Into<TypedValues<u8>>,
        dtype: LogicalDtype,
        decode: impl Fn([u8; N]) -> T,
        retain: fn(TypedFailure<T>) -> FailedCopy,
    ) -> Result<TypedValues<T>, GgufHostCopyCause> {
        let bytes = bytes.into();
        let mut destination = None;
        let result = (|| {
            // Same decoder validation and original precedence. A plan failure
            // must not replace a malformed dense payload's actual first cause.
            let chunks = gguf::native_chunks::<N>(bytes.as_slice())?;
            let request = self.output_request(dtype)?;
            if request.kind != (TransformKind::NativeBytes { width: N })
                || !request.accepts_input(bytes.len())
            {
                return Err(GgufHostCopyCause::TypedBinding);
            }
            match self.host_destinations.as_mut().map_or_else(
                || TypedDestination::<T>::try_new(request.output_elements),
                |bank| TypedDestination::<T>::try_new_admitted(request.output_elements, bank),
            ) {
                Ok(value) => destination = Some(value),
                Err((value, cause)) => {
                    destination = Some(value);
                    return Err(cause.into());
                }
            }
            let values = destination.as_mut().expect("reached destination");
            values.fill(chunks.len(), chunks.iter().map(|chunk| decode(*chunk)))?;
            self.transforms[self.produced] = Some(TypedTransformFacts {
                input_elements: bytes.len(),
                input_capacity: bytes.capacity(),
                requested_output_elements: request.output_elements,
                output_elements: values.values().len(),
                output_capacity: values.values().capacity(),
                input_element_bytes: 1,
                output_element_bytes: size_of::<T>(),
                separate_destination: true,
            });
            Ok(())
        })();
        if let Err(cause) = result {
            self.failed = Some(retain(TypedFailure::Transform {
                input: TransformInput::Bytes(bytes),
                destination,
            }));
            return Err(cause);
        }
        // Same success boundary as ordinary decode_native: the old input retires
        // before the native copy sees the final typed Vec. Both were live during fill.
        drop(bytes);
        Ok(destination.expect("completed destination").into_values())
    }

    pub(super) fn transform_half_bits(
        &mut self,
        bits: impl Into<TypedValues<u16>>,
    ) -> Result<TypedValues<half::f16>, GgufHostCopyCause> {
        let bits = bits.into();
        let mut destination = None;
        let result = (|| {
            let request = self.output_request(LogicalDtype::F16)?;
            if request.kind != TransformKind::HalfBits || !request.accepts_input(bits.len()) {
                return Err(GgufHostCopyCause::TypedBinding);
            }
            match self.host_destinations.as_mut().map_or_else(
                || TypedDestination::<half::f16>::try_new(request.output_elements),
                |bank| {
                    TypedDestination::<half::f16>::try_new_admitted(request.output_elements, bank)
                },
            ) {
                Ok(value) => destination = Some(value),
                Err((value, cause)) => {
                    destination = Some(value);
                    return Err(cause.into());
                }
            }
            let values = destination.as_mut().expect("reached destination");
            values.fill(
                bits.len(),
                bits.as_slice().iter().copied().map(half::f16::from_bits),
            )?;
            self.transforms[self.produced] = Some(TypedTransformFacts {
                input_elements: bits.len(),
                input_capacity: bits.capacity(),
                requested_output_elements: request.output_elements,
                output_elements: values.values().len(),
                output_capacity: values.values().capacity(),
                input_element_bytes: size_of::<u16>(),
                output_element_bytes: size_of::<half::f16>(),
                separate_destination: true,
            });
            Ok(())
        })();
        if let Err(cause) = result {
            self.failed = Some(FailedCopy::F16(TypedFailure::Transform {
                input: TransformInput::HalfBits(bits),
                destination,
            }));
            return Err(cause);
        }
        drop(bits);
        Ok(destination.expect("completed destination").into_values())
    }
}
