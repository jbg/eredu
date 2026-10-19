//! Bounded static-contribution transport inside the shared prediction transaction.
use eredu_nn::{Error, NeuralBackend, Tensor, TensorElementType};
use sha2::{Digest, Sha256};

/// The retained session's rank authority, agreement, publication and completion.
/// Implementations must preserve exact bounded completion and poison their
/// submission authority on indeterminate transport failures.
pub trait PredictionTargetTransport<B: NeuralBackend> {
    /// Actual world rank bound by the selected communication manifest.
    fn rank(&self) -> usize;
    /// Agrees readiness and the exact ordered operation before any data wave.
    fn agree(&mut self, ready: bool, descriptor: [u64; 8]) -> Result<bool, Error>;
    /// Publishes from one selected world rank, completing source dependencies.
    fn publish(&mut self, value: B::Tensor, owner: usize) -> Result<B::Tensor, Error>;
    /// Completes a derived value through the same bounded submission authority.
    fn complete(&mut self, value: &B::Tensor) -> Result<(), Error>;
}

/// How ordered, disjointly owned local results form the complete static value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionTensorAssembly {
    /// Add complete-width contributions before nonlinear finishing equations.
    Sum,
    /// Concatenate contiguous shards in the declared order along this axis.
    Concatenate {
        /// Zero-based tensor axis whose contiguous shards are joined.
        axis: usize,
    },
}

/// One authoritative static contribution. Replicated storage is not another source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionTensorSource {
    owner: usize,
    shape: Vec<i32>,
}
impl PredictionTensorSource {
    /// Retains a source with exact invocation geometry.
    pub fn new(owner: usize, shape: Vec<i32>) -> Result<Self, Error> {
        checked_shape(&shape)?;
        Ok(Self { owner, shape })
    }
    /// Exact world rank owning this contribution.
    pub const fn owner(&self) -> usize {
        self.owner
    }
    /// Exact local geometry, including any vocabulary shard width.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
}

/// Architecture-authored sources and combination for one static operation.
/// No artifact, native tensor, mutable state or device is retained here.
#[derive(Debug, Clone)]
pub struct PredictionTensorAssemblyPlan {
    sources: Vec<PredictionTensorSource>,
    assembly: PredictionTensorAssembly,
    shape: Vec<i32>,
    descriptor: [u64; 8],
}
impl PredictionTensorAssemblyPlan {
    /// Validates exact geometry before a contribution or native transfer is made.
    /// Identity includes the retained execution and the semantic operation.
    pub fn new(
        identity: &str,
        sources: Vec<PredictionTensorSource>,
        assembly: PredictionTensorAssembly,
    ) -> Result<Self, Error> {
        if identity.trim().is_empty() || sources.is_empty() {
            return Err(Error::backend("static assembly has no identity or sources"));
        }
        let mut owners = std::collections::BTreeSet::new();
        let mut shape = sources[0].shape.clone();
        for (index, source) in sources.iter().enumerate() {
            checked_shape(&source.shape)?;
            if !owners.insert(source.owner) {
                return Err(Error::backend("static assembly repeats a source rank"));
            }
            match assembly {
                PredictionTensorAssembly::Sum if source.shape != shape => {
                    return Err(Error::backend("static sum contribution geometry differs"))
                }
                PredictionTensorAssembly::Concatenate { axis } => {
                    if axis >= shape.len()
                        || source.shape.len() != shape.len()
                        || source
                            .shape
                            .iter()
                            .zip(&shape)
                            .enumerate()
                            .any(|(i, (a, b))| i != axis && a != b)
                    {
                        return Err(Error::backend(
                            "static concatenation contribution geometry differs",
                        ));
                    }
                    if index > 0 {
                        shape[axis] =
                            shape[axis].checked_add(source.shape[axis]).ok_or_else(|| {
                                Error::backend("static concatenation width overflows")
                            })?;
                    }
                }
                _ => {}
            }
        }
        checked_shape(&shape)?;
        let mut hash = Sha256::new();
        hash.update(b"eredu-prediction-static-assembly-v1\0");
        hash.update((identity.len() as u64).to_le_bytes());
        hash.update(identity.as_bytes());
        hash.update(
            match assembly {
                PredictionTensorAssembly::Sum => 0u64,
                PredictionTensorAssembly::Concatenate { axis } => axis as u64 + 1,
            }
            .to_le_bytes(),
        );
        hash.update((sources.len() as u64).to_le_bytes());
        for source in &sources {
            hash.update((source.owner as u64).to_le_bytes());
            hash.update((source.shape.len() as u64).to_le_bytes());
            for d in &source.shape {
                hash.update(d.to_le_bytes());
            }
        }
        let mut descriptor = [0; 8];
        descriptor[0] = 0x4552_5354;
        descriptor[1] = sources.len() as u64;
        for (word, bytes) in descriptor[2..6]
            .iter_mut()
            .zip(hash.finalize().chunks_exact(8))
        {
            *word = u64::from_le_bytes(bytes.try_into().unwrap());
        }
        Ok(Self {
            sources,
            assembly,
            shape,
            descriptor,
        })
    }
    /// Ordered authoritative producers, independent of replicated storage.
    pub fn sources(&self) -> &[PredictionTensorSource] {
        &self.sources
    }
    /// Complete geometry before architecture finishing.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    fn phase(&self, step: u64, source: usize) -> [u64; 8] {
        let mut descriptor = self.descriptor;
        descriptor[6] = step;
        descriptor[7] = source as u64;
        descriptor
    }

    /// Completes each rank's retained predecessors before discarding nonproducer
    /// values. Lazy captured branches can still contain collective dependencies.
    pub fn complete_dependencies<B: NeuralBackend, T: PredictionTargetTransport<B>>(
        &self,
        values: &[&B::Tensor],
        transport: &mut T,
    ) -> Result<(), Error> {
        let completed = values
            .iter()
            .try_for_each(|value| transport.complete(value));
        agreed(completed, transport, self.phase(7, 0))
    }

    /// Publishes local contributions without invoking a private TP protocol.
    /// Every rank, including inactive pipeline ranks, enters the same bounded
    /// world publications. Scalar dtype metadata is published first so receivers
    /// preserve actual compute precision rather than guessing from token input
    /// or pipeline-wire precision. Preparation errors reach agreement before data.
    pub fn assemble<B: NeuralBackend, T: PredictionTargetTransport<B>>(
        &self,
        local: Result<Option<B::Tensor>, Error>,
        transport: &mut T,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = local.and_then(|value| {
            match (
                self.sources.iter().find(|s| s.owner == transport.rank()),
                value.as_ref(),
            ) {
                (Some(source), Some(value))
                    if value.shape() == source.shape
                        && element_code(value.element_type()).is_ok() => {}
                (None, None) => {}
                _ => {
                    return Err(Error::backend(
                        "static contribution ownership, shape or precision differs",
                    ))
                }
            }
            Ok(value)
        });
        let local = agreed(local, transport, self.phase(0, 0))?;
        let mut combined: Option<B::Tensor> = None;
        let mut shards = Vec::new();
        let mut element = None;
        for (index, source) in self.sources.iter().enumerate() {
            let metadata = (|| {
                let code = if source.owner == transport.rank() {
                    element_code(local.as_ref().unwrap().element_type())?
                } else {
                    0
                };
                // Four exact F32 enum values use the already-selected publication
                // scalar kind; no integer broadcast capability is added.
                B::Tensor::full_f32(code as f32, &[1], context)
            })();
            let metadata = agreed(metadata, transport, self.phase(1, index))?;
            let metadata = transport.publish(metadata, source.owner)?;
            let prepared = (|| {
                let code = metadata.to_f32_vec(context)?;
                if code.len() != 1 {
                    return Err(Error::backend("static precision metadata is not scalar"));
                }
                let current = code_element(code[0])?;
                if element.is_some_and(|previous| previous != current) {
                    return Err(Error::backend(
                        "static contributions have different physical precision",
                    ));
                }
                element = Some(current);
                let value = if source.owner == transport.rank() {
                    local.as_ref().unwrap().clone()
                } else {
                    B::Tensor::full_f32(0., &source.shape, context)?.cast_float(current, context)?
                };
                if value.shape() != source.shape || value.element_type() != Some(current) {
                    return Err(Error::backend(
                        "static contribution destination geometry or precision differs",
                    ));
                }
                Ok(value)
            })();
            let value = agreed(prepared, transport, self.phase(2, index))?;
            let value = transport.publish(value, source.owner)?;
            match self.assembly {
                PredictionTensorAssembly::Sum => {
                    let next = match combined.take() {
                        None => Ok(value),
                        Some(previous) => previous.add(&value, context),
                    }
                    .and_then(|value| value.compact(context));
                    let next = agreed(next, transport, self.phase(3, index))?;
                    // Do not retain a lazy chain of complete-width contributions
                    // across the next source publication.
                    let completion = transport.complete(&next);
                    agreed(completion, transport, self.phase(5, index))?;
                    combined = Some(next);
                }
                PredictionTensorAssembly::Concatenate { .. } => shards.push(value),
            }
        }
        let value = match self.assembly {
            PredictionTensorAssembly::Sum => Ok(combined.unwrap()),
            // One concatenation avoids retaining every growing prefix.
            PredictionTensorAssembly::Concatenate { axis } => {
                B::Tensor::concatenate(&shards, axis as i32, context)
                    .and_then(|value| value.compact(context))
            }
        };
        let value = agreed(value, transport, self.phase(6, 0))?;
        let complete = if value.shape() != self.shape || value.element_type() != element {
            Err(Error::backend(
                "assembled static value geometry or precision differs",
            ))
        } else {
            transport.complete(&value)
        };
        agreed(complete, transport, self.phase(4, 0))?;
        Ok(value)
    }
}

fn agreed<B: NeuralBackend, T: PredictionTargetTransport<B>, V>(
    value: Result<V, Error>,
    transport: &mut T,
    descriptor: [u64; 8],
) -> Result<V, Error> {
    let accepted = transport.agree(value.is_ok(), descriptor)?;
    match value {
        Err(error) => Err(error),
        Ok(value) if accepted => Ok(value),
        Ok(_) => Err(Error::backend(
            "another rank rejected static prediction preparation or identity",
        )),
    }
}
fn checked_shape(shape: &[i32]) -> Result<(), Error> {
    if shape.is_empty() || shape.iter().any(|d| *d <= 0) {
        return Err(Error::backend(
            "static contribution geometry is empty or nonpositive",
        ));
    }
    shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
        .ok_or_else(|| Error::backend("static contribution element count overflows"))?;
    Ok(())
}
fn element_code(element: Option<TensorElementType>) -> Result<i32, Error> {
    match element {
        Some(TensorElementType::F16) => Ok(1),
        Some(TensorElementType::Bf16) => Ok(2),
        Some(TensorElementType::F32) => Ok(3),
        Some(TensorElementType::F64) => Ok(4),
        _ => Err(Error::backend(
            "static contribution precision is unknown or nonfloating",
        )),
    }
}
fn code_element(code: f32) -> Result<TensorElementType, Error> {
    match code {
        1. => Ok(TensorElementType::F16),
        2. => Ok(TensorElementType::Bf16),
        3. => Ok(TensorElementType::F32),
        4. => Ok(TensorElementType::F64),
        _ => Err(Error::backend(
            "static contribution precision metadata is invalid",
        )),
    }
}
