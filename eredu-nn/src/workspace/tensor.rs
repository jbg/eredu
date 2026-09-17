use super::*;
use crate::{AttentionMask, Index, PadMode, Tensor};

/// Metadata tensor implementing the ordinary portable tensor contract. Native
/// storage and numerical values are never created or read by this type.
#[derive(Debug)]
pub struct WorkspaceTensor {
    pub(super) layout: WorkspaceLayout,
    pub(super) storage: Rc<Storage>,
    pub(super) context: Rc<WorkspaceIdentity>,
    pub(super) imported_existing: bool,
}

impl Clone for WorkspaceTensor {
    fn clone(&self) -> Self {
        self.context.record_tensor_clone();
        Self {
            layout: self.layout.clone(),
            storage: self.storage.clone(),
            context: self.context.clone(),
            imported_existing: self.imported_existing,
        }
    }
}

/// One existing backing allocation shared by any number of logical views.
/// The provider supplies its complete capacity; absent native evidence stays
/// unknown. This contains no native address, device, tensor or allocation.
#[derive(Clone, Debug)]
pub struct WorkspaceExistingStorage {
    pub(super) storage: Rc<Storage>,
    pub(super) context: Rc<WorkspaceIdentity>,
}
impl WorkspaceExistingStorage {
    /// Records an existing allocation bound in this metadata context.
    pub fn new(bytes: Option<u64>, context: &WorkspaceContext) -> Self {
        Self {
            storage: context.new_storage(bytes, Vec::new()),
            context: context.identity.clone(),
        }
    }

    /// Fallible metadata construction using the context's recorded or finite
    /// constructor allowance before creating the shared backing-root owner.
    pub fn try_new(bytes: Option<u64>, context: &WorkspaceContext) -> Result<Self, Error> {
        Ok(Self {
            storage: context.try_new_storage(bytes, Vec::new())?,
            context: context.identity.clone(),
        })
    }

    /// Imports an existing source's complete possible backing population. This
    /// supports collapsed source envelopes without losing per-backing rounding
    /// counts; it creates only metadata and grants no native capacity or credit.
    pub fn try_new_population(
        population: WorkspaceStoragePopulation,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let frames = [
            std::mem::size_of::<WorkspaceStoragePopulation>(),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<&mut Storage>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        if population.bytes.is_some_and(|bytes| bytes > 0) && population.maximum_allocations == 0 {
            return Err(WorkspaceMetadataError::Report(WorkspaceReportError::Source).into());
        }
        let mut value = Self::try_new(population.bytes, context)?;
        Rc::get_mut(&mut value.storage)
            .expect("unpublished metadata root")
            .maximum_allocations = population.maximum_allocations;
        Ok(value)
    }

    /// Exact provider-described backing capacity, never a logical view size.
    pub fn capacity_bytes(&self) -> Option<u64> {
        self.storage.bytes
    }

    /// Whether two retained descriptors name the same metadata backing root.
    /// Equal capacities alone do not establish storage identity.
    pub fn same_storage(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.storage, &other.storage)
    }
}

impl WorkspaceTensor {
    /// Records the existing initialization primitive with the actual source
    /// dtype. The shared producer pays shape/output metadata before allocation;
    /// it creates no native tensor and supplies no execution permission.
    pub fn initialized(
        shape: &[i32],
        dtype: WorkspaceDtype,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Initialize,
            &[],
            shape,
            dtype,
            context,
        )
    }

    /// Declares the integer matrix supplied by the existing prepared text-input
    /// producer. This is a cold storage equation, not native initialization or
    /// permission to create token data. The enclosing driver retains its exact
    /// ordinal and replaces it with the independently admitted input receipt.
    pub fn prepared_token_input(
        shape: &[i32], dtype: WorkspaceDtype, context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if !matches!(shape, [batch, positions] if *batch > 0 && *positions > 0)
            || !matches!(dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
        {
            return Err(context.metadata_error(format_args!("prepared token input requires a nonempty I32/U32 matrix")));
        }
        Self::operation(WorkspaceOperationKind::Elementwise("prepared_token_input"),
            &[], shape, dtype, context)
    }

    /// Traces the actual scalar/host constructor with exact floating precision.
    /// The selected mechanism supplies physical facts; logical storage stays F32.
    pub fn initialized_floating(
        shape: &[i32],
        dtype: WorkspaceFloatingType,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::InitializeFloating(dtype),
            &[],
            shape,
            WorkspaceDtype::Float32,
            context,
        )
    }

    /// Traces an actual floating conversion, including possible new storage.
    /// It preserves geometry and the source backing; it does not reinterpret bits.
    pub fn cast_floating(
        &self,
        dtype: WorkspaceFloatingType,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if self.layout.dtype != WorkspaceDtype::Float32 {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Self::operation(
            WorkspaceOperationKind::CastFloating(dtype),
            &[self],
            self.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }

    /// Trace a completed independent host-to-device transfer. The caller must
    /// retain the actual immutable host source and supply native authorization;
    /// an ordinary symbolic tensor is never a transfer permission.
    pub fn transfer_host_floating(
        &self,
        dtype: WorkspaceFloatingType,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            WorkspaceFloatingType,
            &WorkspaceContext,
            [&Self; 1],
            Result<Self, Error>,
        )>())?;
        if self.layout.dtype != WorkspaceDtype::Float32
            || self.shape().is_empty()
            || self.shape().len() > 4
            || self.shape().iter().any(|n| *n <= 0)
            || self
                .layout
                .representation()
                .is_some_and(|r| r.dtype() != dtype)
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Self::operation(
            WorkspaceOperationKind::HostTransferFloating(dtype),
            &[self],
            self.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }

    /// Trace construction separately from the parameter's logical geometry.
    /// The scalar seed remains charged for this complete span even when binding
    /// immediately replaces the placeholder; no logical weight buffer is made.
    pub(super) fn parameter_placeholder(
        layout: WorkspaceLayout,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let value = Self::existing(layout, context)?;
        let mut outputs = context.metadata_vec(1)?;
        outputs.push(context.layout(&[], value.layout.dtype())?);
        context.execute(WorkspaceOperationKind::ParameterPlaceholder, &[], outputs)?;
        Ok(value)
    }

    /// Introduces existing storage. Its residency is priced separately from
    /// this span, including when subsequent views alias its backing allocation.
    pub fn existing(layout: WorkspaceLayout, context: &WorkspaceContext) -> Result<Self, Error> {
        Ok(Self {
            storage: context.try_new_storage(
                Some(
                    layout
                        .as_view()
                        .bytes()
                        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?,
                ),
                Vec::new(),
            )?,
            layout,
            context: context.identity.clone(),
            imported_existing: false,
        })
    }

    /// Imports a logical view of provider-described existing storage. Broadcast
    /// views may have more logical bytes than their backing allocation. Shape
    /// and backing identity must come from the same retained native state.
    pub fn existing_with_storage(
        layout: WorkspaceLayout,
        storage: &WorkspaceExistingStorage,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::import_owned(layout, storage, context).map_err(|cause| match cause {
            WorkspaceImportError::Context => context.metadata_error(format_args!(
                "existing storage belongs to another workspace trace"
            )),
            WorkspaceImportError::Reserve(cause) => context.metadata_source(cause),
            WorkspaceImportError::Metadata(cause) => cause,
        })
    }
    /// Exact logical geometry carried by this metadata value.
    pub fn layout(&self) -> &WorkspaceLayout {
        &self.layout
    }
    /// Whether two tensors belong to the same metadata execution context.
    /// This compares context identity only; it creates no storage and provides
    /// no native allocation, device or submission authority.
    pub fn same_context(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.context, &other.context)
    }
    /// Traces an exact unit-stride rectangular update used by a selected native
    /// storage mechanism. Source and update ranks/dtypes must match; broadcasting
    /// or conversion requires its own operation. No in-place donation is assumed.
    pub fn update_slice(
        &self,
        update: &Self,
        starts: &[i32],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if starts.len() != self.shape().len()
            || update.shape().len() != self.shape().len()
            || self.layout.dtype != update.layout.dtype
            || starts.iter().zip(update.shape()).zip(self.shape()).any(
                |((&start, &width), &limit)| {
                    start < 0 || start.checked_add(width).is_none_or(|end| end > limit)
                },
            )
        {
            return Err(context.metadata_error(format_args!(
                "workspace slice update exceeds its destination or changes rank/dtype"
            )));
        }
        let mut owned_starts = context.metadata_vec(starts.len())?;
        owned_starts.extend_from_slice(starts);
        Self::operation(
            WorkspaceOperationKind::SliceUpdate {
                starts: owned_starts,
            },
            &[self, update],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    /// Describes an exact typed zero-fill constructor. The prototype lends only
    /// its scalar representation; the result has independent storage and shape.
    /// Concrete mechanisms still qualify the actual scalar seed and fill worker.
    pub fn zeros_from_prototype(shape: &[i32], prototype: WorkspaceLayoutView<'_>,
        context: &WorkspaceContext) -> Result<Self, Error> {
    let name=match prototype.dtype() {
        WorkspaceDtype::Float32=>match prototype.representation().ok_or_else(||
            context.metadata_error(format_args!("zero fill lacks exact floating source dtype")))?.dtype() {
            WorkspaceFloatingType::Float32=>"zeros_f32",WorkspaceFloatingType::Float16=>"zeros_f16",
            WorkspaceFloatingType::Bfloat16=>"zeros_bf16",
        },
        WorkspaceDtype::Int32=>"zeros_i32",WorkspaceDtype::Uint32=>"zeros_u32",
        WorkspaceDtype::Uint8=>"zeros_u8",WorkspaceDtype::Bool=>"zeros_bool",
    };
    let mut outputs=context.metadata_vec(1)?;outputs.push(context.layout(shape,prototype.dtype())?);
    let mut values=context.execute(WorkspaceOperationKind::Elementwise(name),&[],outputs)?;
    Ok(values.pop().expect("one declared zero output"))
    }
    /// Traces the existing rank-preserving positive-stride Slice worker with
    /// its exact coordinates. Empty/scalar geometry remains descriptive; each
    /// mechanism separately qualifies the actual native source it supports.
    pub fn static_slice(&self, starts:&[i32], ends:&[i32], strides:&[i32],
        context:&WorkspaceContext) -> Result<Self,Error> {
        use std::mem::{size_of,size_of_val};
        let frames=[size_of::<(&Self,&[i32],&[i32],&[i32],&WorkspaceContext)>(),
            size_of::<[Vec<i32>;4]>(),size_of::<Result<Self,Error>>(),
            size_of::<[i32;4]>(),size_of::<Option<i32>>(),size_of::<usize>(),
            size_of::<std::ops::Range<usize>>()];
        context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||context.metadata_error(format_args!("static slice controls overflow")))?)?;
        self.validate_context(context)?;
        let rank=self.shape().len();
        if starts.len()!=rank || ends.len()!=rank || strides.len()!=rank {
            return Err(context.metadata_error(format_args!("static slice coordinate rank differs")));
        }
        let mut shape=context.metadata_vec(rank)?;
        for axis in 0..rank {
            let (a,b,step,n)=(starts[axis],ends[axis],strides[axis],self.shape()[axis]);
            if a<0 || b<a || b>n || step<=0 {
                return Err(context.metadata_error(format_args!("invalid static slice geometry")));
            }
            let width=b.checked_sub(a).and_then(|d|d.checked_add(step-1))
                .map(|d|d/step).ok_or_else(||context.metadata_error(format_args!("static slice extent overflow")))?;
            shape.push(width);
        }
        let mut a=context.metadata_vec(rank)?;a.extend_from_slice(starts);
        let mut b=context.metadata_vec(rank)?;b.extend_from_slice(ends);
        let mut c=context.metadata_vec(rank)?;c.extend_from_slice(strides);
        Self::operation(WorkspaceOperationKind::StaticSlice {starts:a,ends:b,strides:c},
            &[self],&shape,self.layout.dtype,context)
    }
    /// Traces an exact positive-stride replacement without broadcasting or
    /// rank changes. Native storage facts still account for a full source copy.
    pub fn update_static_slice(
        &self,
        update: &Self,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if starts.len() != self.shape().len()
            || ends.len() != starts.len()
            || strides.len() != starts.len()
            || update.shape().len() != starts.len()
            || update.layout.dtype != self.layout.dtype
            || starts
                .iter()
                .zip(ends)
                .zip(strides)
                .zip(self.shape())
                .zip(update.shape())
                .any(|((((&a, &b), &step), &n), &width)| {
                    a < 0
                        || b <= a
                        || b > n
                        || step <= 0
                        || b.checked_sub(a)
                            .and_then(|d| d.checked_add(step - 1))
                            .map(|d| d / step)
                            != Some(width)
                })
        {
            return Err(
                context.metadata_error(format_args!("invalid static slice update geometry"))
            );
        }
        let mut a = context.metadata_vec(starts.len())?;
        a.extend_from_slice(starts);
        let mut b = context.metadata_vec(ends.len())?;
        b.extend_from_slice(ends);
        let mut c = context.metadata_vec(strides.len())?;
        c.extend_from_slice(strides);
        Self::operation(
            WorkspaceOperationKind::StaticSliceUpdate {
                starts: a,
                ends: b,
                strides: c,
            },
            &[self, update],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    /// Traces row-major materialization. Selected facts decide whether aliasing
    /// or a copy can occur; this does not drop an existing backing identity.
    pub fn contiguous(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Contiguous,
            &[self],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    /// Traces an independent logical data copy through the selected mechanism.
    /// The containing trace retains its allocation and all transfer workspace.
    pub fn deep_copy(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::DeepCopy,
            &[self],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    pub(super) fn validate_context(&self, context: &WorkspaceContext) -> Result<(), Error> {
        if Rc::ptr_eq(&self.context, &context.identity) {
            Ok(())
        } else {
            Err(context.metadata_error(format_args!(
                "workspace tensor belongs to a different selected mechanism"
            )))
        }
    }
    pub(super) fn operation(
        kind: WorkspaceOperationKind,
        inputs: &[&Self],
        shape: &[i32],
        dtype: WorkspaceDtype,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let mut outputs = context.metadata_vec(1)?;
        outputs.push(context.layout(shape, dtype)?);
        let mut values = context.execute(kind, inputs, outputs)?;
        Ok(values.pop().expect("one declared output"))
    }
    pub(super) fn unary(
        &self,
        name: &'static str,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Elementwise(name),
            &[self],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    pub(super) fn floating_unary(
        &self,
        name: &'static str,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Elementwise(name),
            &[self],
            self.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn binary(
        &self,
        rhs: &Self,
        name: &'static str,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Elementwise(name),
            &[self, rhs],
            &broadcast(self.shape(), rhs.shape(), context)?,
            promote(self.layout.dtype, rhs.layout.dtype),
            context,
        )
    }
    fn view(
        &self,
        shape: &[i32],
        name: &'static str,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::View(name),
            &[self],
            shape,
            self.layout.dtype,
            context,
        )
    }
}

pub(super) fn axis(axis: i32, rank: usize, context: &WorkspaceContext) -> Result<usize, Error> {
    let rank = i64::try_from(rank)
        .map_err(|_| context.metadata_error(format_args!("workspace rank overflow")))?;
    let normalized = if axis < 0 {
        rank + i64::from(axis)
    } else {
        i64::from(axis)
    };
    if normalized < 0 || normalized >= rank {
        Err(context.metadata_error(format_args!("workspace axis outside tensor rank")))
    } else {
        Ok(normalized as usize)
    }
}
pub(super) fn broadcast(
    a: &[i32],
    b: &[i32],
    context: &WorkspaceContext,
) -> Result<Vec<i32>, Error> {
    let plan = super::WorkspaceBroadcastShape::new(a, b)
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    let mut shape = context.metadata_vec(plan.rank())?;
    shape.extend(plan.dimensions());
    Ok(shape)
}
fn promote(a: WorkspaceDtype, b: WorkspaceDtype) -> WorkspaceDtype {
    if a == WorkspaceDtype::Float32 || b == WorkspaceDtype::Float32 {
        WorkspaceDtype::Float32
    } else if a == WorkspaceDtype::Int32 || b == WorkspaceDtype::Int32 {
        WorkspaceDtype::Int32
    } else if a == WorkspaceDtype::Uint32 || b == WorkspaceDtype::Uint32 {
        WorkspaceDtype::Uint32
    } else if a == WorkspaceDtype::Uint8 || b == WorkspaceDtype::Uint8 {
        WorkspaceDtype::Uint8
    } else {
        WorkspaceDtype::Bool
    }
}
fn extent(value: i64, context: &WorkspaceContext) -> Result<i32, Error> {
    i32::try_from(value)
        .ok()
        .filter(|value| *value >= 0)
        .ok_or_else(|| {
            context.metadata_error(format_args!("invalid or overflowing workspace extent"))
        })
}

macro_rules! unary {
    ($($name:ident),* $(,)?) => {$(fn $name(&self, context: &WorkspaceContext) -> Result<Self, Error> { self.unary(stringify!($name), context) })*};
}
macro_rules! binary {
    ($($name:ident),* $(,)?) => {$(fn $name(&self, rhs: &Self, context: &WorkspaceContext) -> Result<Self, Error> { self.binary(rhs, stringify!($name), context) })*};
}
impl Tensor for WorkspaceTensor {
    type Context = WorkspaceContext;
    fn shape(&self) -> &[i32] {
        self.layout.shape()
    }
    fn unloaded_f32(shape: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        Self::parameter_placeholder(context.layout(shape, WorkspaceDtype::Float32)?, context)
    }
    fn unloaded_parameter_f32(
        parameter: &crate::ParameterSpec,
        shape: &[i32],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::parameter_placeholder(
            context.parameter_layout(parameter, shape, WorkspaceDtype::Float32)?,
            context,
        )
    }
    fn unloaded_i32(shape: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        Self::parameter_placeholder(context.layout(shape, WorkspaceDtype::Int32)?, context)
    }
    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if WorkspaceLayoutView::new(shape, WorkspaceDtype::Float32)
            .and_then(WorkspaceLayoutView::elements)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
            != values.len() as u64
        {
            return Err(context.metadata_error(format_args!(
                "workspace host data length differs from shape"
            )));
        }
        Self::full_f32(0.0, shape, context)
    }
    fn from_f32_fn(
        shape: &[i32],
        _: impl FnMut(usize) -> f32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        crate::F32InitializationPlan::new(shape).map_err(|cause| context.metadata_source(cause))?;
        Self::operation(
            WorkspaceOperationKind::GeneratedF32Initialization,
            &[],
            shape,
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn from_i32_slice(
        values: &[i32],
        shape: &[i32],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if WorkspaceLayoutView::new(shape, WorkspaceDtype::Int32)
            .and_then(WorkspaceLayoutView::elements)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
            != values.len() as u64
        {
            return Err(context.metadata_error(format_args!(
                "workspace host data length differs from shape"
            )));
        }
        Self::full_i32(0, shape, context)
    }
    fn full_f32(_: f32, shape: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        Self::initialized(shape, WorkspaceDtype::Float32, context)
    }
    fn full_i32(_: i32, shape: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        Self::initialized(shape, WorkspaceDtype::Int32, context)
    }
    binary!(add, subtract, multiply);
    unary!(square);
    fn divide(&self, rhs: &Self, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Elementwise("divide"),
            &[self, rhs],
            &broadcast(self.shape(), rhs.shape(), context)?,
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn tanh(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.floating_unary("tanh", context)
    }
    fn multiply_scalar(&self, _: f32, context: &WorkspaceContext) -> Result<Self, Error> {
        self.floating_unary("multiply_scalar", context)
    }
    fn maximum_scalar(&self, _: f32, context: &WorkspaceContext) -> Result<Self, Error> {
        self.floating_unary("maximum_scalar", context)
    }
    fn maximum_i32(&self, _: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        // This constructor has an actual I32 scalar operand. In particular,
        // sanitizing signed coordinates must keep them usable as gather indices.
        Self::operation(
            WorkspaceOperationKind::Elementwise("maximum_i32"),
            &[self],
            self.shape(),
            promote(self.layout.dtype, WorkspaceDtype::Int32),
            context,
        )
    }
    fn clip(
        &self,
        minimum: &Self,
        maximum: &Self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let shape = broadcast(
            &broadcast(self.shape(), minimum.shape(), context)?,
            maximum.shape(),
            context,
        )?;
        Self::operation(
            WorkspaceOperationKind::Elementwise("clip"),
            &[self, minimum, maximum],
            &shape,
            promote(
                self.layout.dtype,
                promote(minimum.layout.dtype, maximum.layout.dtype),
            ),
            context,
        )
    }
    fn softmax_axis(
        &self,
        selected: i32,
        precise: bool,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let selected = axis(selected, self.shape().len(), context)?;
        Self::operation(
            WorkspaceOperationKind::Reduction("softmax", selected as i32, precise),
            &[self],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    fn reshape(&self, requested: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        let mut shape = context.metadata_vec(requested.len())?;
        shape.extend_from_slice(requested);
        let mut unknown = shape
            .iter()
            .enumerate()
            .filter_map(|(i, n)| (*n == -1).then_some(i));
        let inferred = unknown.next();
        if unknown.next().is_some() {
            return Err(context.metadata_error(format_args!(
                "workspace reshape has multiple inferred dimensions"
            )));
        }
        if let Some(index) = inferred {
            shape[index] = 1;
            let known = WorkspaceLayoutView::new(&shape, self.layout.dtype)
                .and_then(WorkspaceLayoutView::elements)
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
            let count = self
                .layout
                .as_view()
                .elements()
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
            if known == 0 || count % known != 0 {
                return Err(context.metadata_error(format_args!(
                    "workspace reshape cannot infer an exact dimension"
                )));
            }
            shape[index] = i32::try_from(count / known).map_err(|_| {
                context.metadata_error(format_args!("workspace reshape inferred extent overflow"))
            })?;
        }
        if WorkspaceLayoutView::new(&shape, self.layout.dtype)
            .and_then(WorkspaceLayoutView::elements)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
            != self
                .layout
                .as_view()
                .elements()
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
        {
            return Err(
                context.metadata_error(format_args!("workspace reshape changes element count"))
            );
        }
        self.view(&shape, "reshape", context)
    }
    fn broadcast_to(&self, shape: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        if !super::WorkspaceBroadcastShape::new(self.shape(), shape)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
            .dimensions()
            .eq(shape.iter().copied())
        {
            return Err(
                context.metadata_error(format_args!("workspace broadcast target is incompatible"))
            );
        }
        self.view(shape, "broadcast", context)
    }
    fn transpose_axes(&self, axes: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        if axes.len() != self.shape().len() {
            return Err(context.metadata_error(format_args!("workspace transpose rank mismatch")));
        }
        let mut normalized = context.metadata_vec(axes.len())?;
        for selected in axes {
            normalized.push(axis(*selected, self.shape().len(), context)?);
        }
        // Validate every axis before duplicate refusal, as in the shared
        // ordinary contract. A borrowed prefix scan needs no tree nodes.
        if normalized
            .iter()
            .enumerate()
            .any(|(i, a)| normalized[..i].contains(a))
        {
            return Err(context.metadata_error(format_args!("workspace transpose repeats an axis")));
        }
        let mut shape = context.metadata_vec(normalized.len())?;
        shape.extend(normalized.iter().map(|a| self.shape()[*a]));
        Self::operation(WorkspaceOperationKind::Transpose(normalized),
            &[self], &shape, self.layout.dtype, context)
    }
    fn swap_axes(&self, left: i32, right: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        let mut axes = context.metadata_vec(self.shape().len())?;
        axes.extend(0..self.shape().len() as i32);
        let rank = axes.len();
        axes.swap(axis(left, rank, context)?, axis(right, rank, context)?);
        self.transpose_axes(&axes, context)
    }
    fn transpose(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        let mut axes = context.metadata_vec(self.shape().len())?;
        axes.extend((0..self.shape().len() as i32).rev());
        self.transpose_axes(&axes, context)
    }
    fn expand_dims(&self, selected: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        let rank = self
            .shape()
            .len()
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut shape = context.metadata_vec(rank)?;
        shape.extend_from_slice(self.shape());
        shape.insert(axis(selected, rank, context)?, 1);
        self.view(&shape, "expand_dims", context)
    }
    fn squeeze_axes(&self, axes: &[i32], context: &WorkspaceContext) -> Result<Self, Error> {
        let mut selected = context.metadata_vec(axes.len())?;
        for a in axes {
            selected.push(axis(*a, self.shape().len(), context)?);
        }
        if selected
            .iter()
            .enumerate()
            .any(|(i, a)| selected[..i].contains(a))
            || selected.iter().any(|i| self.shape()[*i] != 1)
        {
            return Err(context.metadata_error(format_args!(
                "workspace squeeze requires distinct unit axes"
            )));
        }
        let mut shape = context.metadata_vec(self.shape().len() - selected.len())?;
        shape.extend(
            self.shape()
                .iter()
                .enumerate()
                .filter_map(|(i, n)| (!selected.contains(&i)).then_some(*n)),
        );
        self.view(&shape, "squeeze", context)
    }
    fn index(&self, indexes: &[Index], context: &WorkspaceContext) -> Result<Self, Error> {
        if indexes.len() > self.shape().len() {
            return Err(context.metadata_error(format_args!("too many workspace indexes")));
        }
        let selected_axes = indexes
            .iter()
            .filter(|index| matches!(index, Index::At(_)))
            .count();
        let mut shape = context.metadata_vec(self.shape().len() - selected_axes)?;
        for (i, size) in self.shape().iter().copied().enumerate() {
            let position = |n: i32| {
                if n < 0 {
                    i64::from(size) + i64::from(n)
                } else {
                    i64::from(n)
                }
            };
            match indexes.get(i).copied().unwrap_or(Index::Full) {
                Index::Full => shape.push(size),
                Index::At(n) => {
                    let n = position(n);
                    if n < 0 || n >= i64::from(size) {
                        return Err(
                            context.metadata_error(format_args!("workspace index out of bounds"))
                        );
                    }
                }
                Index::Range(start, end) => {
                    let start = position(start);
                    let end = position(end);
                    if start < 0 || end < start || end > i64::from(size) {
                        return Err(
                            context.metadata_error(format_args!("workspace range out of bounds"))
                        );
                    }
                    shape.push(extent(end - start, context)?);
                }
            }
        }
        Self::operation(
            WorkspaceOperationKind::Index { selected_axes },
            &[self],
            &shape,
            self.layout.dtype,
            context,
        )
    }
    fn narrow_axis(&self, axis: usize, start: i32, end: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        self.validate_context(context)?;
        let range = crate::TensorAxisRange::new(self.shape(), axis, start, end)
            .map_err(|cause|context.metadata_error(format_args!("{cause}")))?;
        let frames = [std::mem::size_of::<crate::TensorAxisRange<'_>>(),
            std::mem::size_of::<[Vec<i32>;3]>(), std::mem::size_of::<(&Self,usize,i32,i32,&WorkspaceContext)>(),
            std::mem::size_of::<Result<Self,Error>>(), std::mem::size_of::<Result<crate::TensorAxisRange<'_>,crate::TensorAxisRangeError>>()];
        context.charge_metadata(frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        let mut starts=context.metadata_vec(range.shape().len())?; starts.resize(range.shape().len(),0);
        let mut ends=context.metadata_vec(range.shape().len())?; ends.extend_from_slice(range.shape());
        let mut strides=context.metadata_vec(range.shape().len())?; strides.resize(range.shape().len(),1);
        starts[range.axis()]=range.start(); ends[range.axis()]=range.end();
        self.static_slice(&starts,&ends,&strides,context)
    }
    fn take_axis(
        &self,
        indexes: &Self,
        selected: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let selected = axis(selected, self.shape().len(), context)?;
        if !matches!(
            indexes.layout.dtype,
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32 | WorkspaceDtype::Uint8
        ) {
            return Err(
                context.metadata_error(format_args!("workspace gather requires integer indices"))
            );
        }
        let rank = (self.shape().len() - 1)
            .checked_add(indexes.shape().len())
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut shape = context.metadata_vec(rank)?;
        shape.extend_from_slice(&self.shape()[..selected]);
        shape.extend_from_slice(indexes.shape());
        shape.extend_from_slice(&self.shape()[selected + 1..]);
        Self::operation(
            WorkspaceOperationKind::Gather { axis: selected },
            &[self, indexes],
            &shape,
            self.layout.dtype,
            context,
        )
    }
    fn zeros_like(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Initialize,
            &[self],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    fn equal_i32(&self, _: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::Elementwise("equal_i32"),
            &[self],
            self.shape(),
            WorkspaceDtype::Bool,
            context,
        )
    }
    fn logical_or(&self, rhs: &Self, context: &WorkspaceContext) -> Result<Self, Error> {
        if self.layout.dtype != WorkspaceDtype::Bool || rhs.layout.dtype != WorkspaceDtype::Bool {
            return Err(
                context.metadata_error(format_args!("workspace logical input must be boolean"))
            );
        }
        self.binary(rhs, "logical_or", context)
    }
    fn where_condition(
        condition: &Self,
        when_true: &Self,
        when_false: &Self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if condition.layout.dtype != WorkspaceDtype::Bool {
            return Err(context.metadata_error(format_args!("workspace condition must be boolean")));
        }
        let shape = broadcast(
            condition.shape(),
            &broadcast(when_true.shape(), when_false.shape(), context)?,
            context,
        )?;
        Self::operation(
            WorkspaceOperationKind::Elementwise("where"),
            &[condition, when_true, when_false],
            &shape,
            promote(when_true.layout.dtype, when_false.layout.dtype),
            context,
        )
    }
    fn masked_scatter(
        &self,
        mask: &Self,
        source: &Self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if mask.layout.dtype != WorkspaceDtype::Bool
            || !super::WorkspaceBroadcastShape::new(self.shape(), mask.shape())
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
                .dimensions()
                .eq(self.shape().iter().copied())
        {
            return Err(context.metadata_error(format_args!("invalid workspace scatter mask")));
        }
        Self::operation(
            WorkspaceOperationKind::Elementwise("masked_scatter"),
            &[self, mask, source],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    fn rope_with_frequencies(
        &self,
        dimensions: i32,
        traditional: bool,
        offset: i32,
        frequencies: &Self,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::RotaryFrequencies(dimensions, traditional, offset),
            &[self, frequencies],
            self.shape(),
            self.layout.dtype,
            context,
        )
    }
    fn concatenate(
        values: &[Self],
        selected: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| context.metadata_error(format_args!("empty workspace concatenation")))?;
        let selected = axis(selected, first.shape().len(), context)?;
        let mut shape = context.metadata_vec(first.shape().len())?;
        shape.extend_from_slice(first.shape());
        shape[selected] = 0;
        for value in values {
            if value.shape().len() != shape.len()
                || value.layout.dtype != first.layout.dtype
                || value
                    .shape()
                    .iter()
                    .enumerate()
                    .any(|(i, n)| i != selected && *n != shape[i])
            {
                return Err(
                    context.metadata_error(format_args!("incompatible workspace concatenation"))
                );
            }
            shape[selected] = shape[selected]
                .checked_add(value.shape()[selected])
                .ok_or_else(|| {
                    context.metadata_error(format_args!("workspace concatenation extent overflow"))
                })?;
        }
        let mut inputs = context.metadata_vec(values.len())?;
        inputs.extend(values.iter());
        Self::operation(
            WorkspaceOperationKind::Concatenate,
            &inputs,
            &shape,
            first.layout.dtype,
            context,
        )
    }
    fn stack(values: &[Self], selected: i32, context: &WorkspaceContext) -> Result<Self, Error> {
        let mut expanded = context.metadata_vec(values.len())?;
        for value in values {
            expanded.push(value.expand_dims(selected, context)?);
        }
        Self::concatenate(&expanded, selected, context)
    }
    fn matmul(lhs: &Self, rhs: &Self, context: &WorkspaceContext) -> Result<Self, Error> {
        let plan = super::WorkspaceMatmulShape::new(lhs.shape(), rhs.shape())
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        let mut shape = context.metadata_vec(plan.rank())?;
        shape.extend(plan.dimensions());
        Self::operation(
            WorkspaceOperationKind::Matmul,
            &[lhs, rhs],
            &shape,
            promote(lhs.layout.dtype, rhs.layout.dtype),
            context,
        )
    }
    fn sum_axis(
        value: &Self,
        selected: i32,
        keep_dims: bool,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        reduce(
            value,
            selected,
            keep_dims,
            "sum",
            match value.layout.dtype {
                WorkspaceDtype::Bool => WorkspaceDtype::Int32,
                WorkspaceDtype::Uint8 => WorkspaceDtype::Uint32,
                dtype => dtype,
            },
            context,
        )
    }
    fn mean_axis(
        value: &Self,
        selected: i32,
        keep_dims: bool,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        reduce(
            value,
            selected,
            keep_dims,
            "mean",
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn argmin_axis(
        value: &Self,
        selected: i32,
        keep_dims: bool,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        reduce(
            value,
            selected,
            keep_dims,
            "argmin",
            WorkspaceDtype::Uint32,
            context,
        )
    }
    fn pad(
        value: &Self,
        widths: &[(i32, i32)],
        mode: PadMode,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if widths.len() != value.shape().len() || widths.iter().any(|(a, b)| *a < 0 || *b < 0) {
            return Err(context.metadata_error(format_args!("invalid workspace padding")));
        }
        let mut shape = context.metadata_vec(value.shape().len())?;
        for (n, (a, b)) in value.shape().iter().zip(widths) {
            shape.push(extent(
                i64::from(*n) + i64::from(*a) + i64::from(*b),
                context,
            )?);
        }
        Self::operation(
            WorkspaceOperationKind::Pad(mode),
            &[value],
            &shape,
            value.layout.dtype,
            context,
        )
    }
    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        convolution(
            input,
            weight,
            &[stride],
            &[padding],
            &[dilation],
            groups,
            None,
            context,
        )
    }
    fn conv2d(
        input: &Self,
        weight: &Self,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        convolution(
            input,
            weight,
            &[stride.0, stride.1],
            &[padding.0, padding.1],
            &[dilation.0, dilation.1],
            groups,
            None,
            context,
        )
    }
    fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        convolution(
            input,
            weight,
            &[stride],
            &[padding],
            &[dilation],
            groups,
            Some(&[output_padding]),
            context,
        )
    }
    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if weight.shape().len() != 2 || input.shape().last() != weight.shape().get(1) {
            return Err(
                context.metadata_error(format_args!("workspace linear dimensions disagree"))
            );
        }
        let mut shape = context.metadata_vec(input.shape().len())?;
        shape.extend_from_slice(input.shape());
        *shape.last_mut().ok_or_else(|| {
            context.metadata_error(format_args!("workspace linear input is scalar"))
        })? = weight.shape()[0];
        if bias.is_some_and(|bias| bias.shape() != [weight.shape()[0]]) {
            return Err(
                context.metadata_error(format_args!("workspace linear bias dimensions disagree"))
            );
        }
        let mut inputs = context.metadata_vec(2 + usize::from(bias.is_some()))?;
        inputs.extend([input, weight]);
        inputs.extend(bias);
        Self::operation(
            WorkspaceOperationKind::DenseLinear,
            &inputs,
            &shape,
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        crate::operation_geometry::NormalizationGeometry::new_fixed(input.shape(), epsilon)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        let mut inputs = context
            .metadata_vec(1 + usize::from(weight.is_some()) + usize::from(bias.is_some()))?;
        inputs.push(input);
        inputs.extend(weight);
        inputs.extend(bias);
        Self::operation(
            WorkspaceOperationKind::Normalization("layer_norm", None),
            &inputs,
            input.shape(),
            input.layout.dtype,
            context,
        )
    }
    fn gelu(input: &Self, context: &WorkspaceContext) -> Result<Self, Error> {
        input.floating_unary("gelu", context)
    }
    fn elu(input: &Self, _: f32, context: &WorkspaceContext) -> Result<Self, Error> {
        input.floating_unary("elu", context)
    }
    fn masked_output_projection(
        input: crate::multimodal::MaskedOutputProjectionInput<'_, Self>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        input.validate()?;
        Self::operation(
            WorkspaceOperationKind::MaskedOutputProjection {
                top_centroids: input.top_centroids,
                mask_margin: input.mask_margin,
            },
            &[
                input.hidden,
                input.output_weight,
                input.centroid_logits,
                input.token_ordering,
            ],
            &[
                input.hidden.shape()[0],
                input.hidden.shape()[1],
                input.output_weight.shape()[0],
            ],
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn multi_axis_rotary_embeddings(
        position_ids: &Self,
        spec: &crate::multimodal::MultiAxisRotarySpec,
        context: &WorkspaceContext,
    ) -> Result<(Self, Self), Error> {
        trace_multi_axis_rotary(position_ids, spec.as_ref(), false, context)
    }
    fn multi_axis_rotary_embeddings_prepared(
        position_ids: &Self,
        prepared: crate::multimodal::PreparedMultiAxisRotary<'_>,
        context: &WorkspaceContext,
    ) -> Result<(Self, Self), Error> {
        trace_multi_axis_rotary(position_ids, prepared.spec(), true, context)
    }
    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::operation(
            WorkspaceOperationKind::TensorRotary(dimensions, traditional, base, scale, offset),
            &[input],
            input.shape(),
            input.layout.dtype,
            context,
        )
    }
    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        _: f32,
        mask: AttentionMask<'_, Self>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let mut inputs =
            context.metadata_vec(3 + usize::from(matches!(mask, AttentionMask::Tensor(_))))?;
        inputs.extend([queries, keys, values]);
        if let AttentionMask::Tensor(mask) = mask {
            inputs.push(mask);
        }
        attention(
            &inputs,
            WorkspaceOperationKind::Attention {
                causal: matches!(mask, AttentionMask::Causal),
                window: None,
                sinks: false,
                softcap: false,
                arithmetic: crate::AttentionArithmetic::Fused,
            },
            context,
        )
    }
}

fn reduce(
    value: &WorkspaceTensor,
    selected: i32,
    keep: bool,
    name: &'static str,
    dtype: WorkspaceDtype,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let selected = axis(selected, value.shape().len(), context)?;
    let mut shape = context.metadata_vec(value.shape().len())?;
    shape.extend_from_slice(value.shape());
    if keep {
        shape[selected] = 1;
    } else {
        shape.remove(selected);
    }
    WorkspaceTensor::operation(
        WorkspaceOperationKind::Reduction(name, selected as i32, keep),
        &[value],
        &shape,
        dtype,
        context,
    )
}

pub(super) fn attention(
    inputs: &[&WorkspaceTensor],
    kind: WorkspaceOperationKind,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let q = inputs[0].shape();
    let k = inputs[1].shape();
    let v = inputs[2].shape();
    if q.len() != 4
        || k.len() != 4
        || v.len() != 4
        || q[0] != k[0]
        || k[0] != v[0]
        || k[1] != v[1]
        || k[2] != v[2]
        || q[3] != k[3]
        || k[1] <= 0
        || q[1] % k[1] != 0
    {
        return Err(context.metadata_error(format_args!("invalid workspace attention geometry")));
    }
    let mut output_shape = [q[0], q[1], q[2], v[3]];
    let output_rank = if let WorkspaceOperationKind::Attention {
        window: Some((window, offset)),
        ..
    } = kind
    {
        crate::operation_geometry::SlidingAttentionGeometry::new_fixed(q[2], k[2], window, offset)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        output_shape[1] = q[2];
        output_shape[2] = q[1].checked_mul(v[3]).ok_or_else(|| {
            context.metadata_error(format_args!("sliding attention output width overflows"))
        })?;
        3
    } else {
        4
    };
    WorkspaceTensor::operation(
        kind,
        inputs,
        &output_shape[..output_rank],
        inputs[0].layout.dtype,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn convolution(
    input: &WorkspaceTensor,
    weight: &WorkspaceTensor,
    stride: &[i32],
    padding: &[i32],
    dilation: &[i32],
    groups: i32,
    transposed: Option<&[i32]>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let rank = stride
        .len()
        .checked_add(2)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    if input.shape().len() != rank
        || weight.shape().len() != rank
        || groups <= 0
        || stride.iter().any(|n| *n <= 0)
        || dilation.iter().any(|n| *n <= 0)
        || padding.iter().any(|n| *n < 0)
        || transposed.is_some_and(|values| values.iter().any(|n| *n < 0))
    {
        return Err(context.metadata_error(format_args!("invalid workspace convolution geometry")));
    }
    let channels = input.shape()[rank - 1];
    if i64::from(weight.shape()[rank - 1]) * i64::from(groups) != i64::from(channels)
        || weight.shape()[0] % groups != 0
    {
        return Err(context.metadata_error(format_args!("workspace convolution channels disagree")));
    }
    let mut shape = context.metadata_vec(rank)?;
    shape.push(input.shape()[0]);
    for i in 0..stride.len() {
        let n = i64::from(input.shape()[i + 1]);
        let kernel = i64::from(weight.shape()[i + 1]);
        let stride_value = i64::from(stride[i]);
        let pad = i64::from(padding[i]);
        let dil = i64::from(dilation[i]);
        if kernel <= 0 {
            return Err(context.metadata_error(format_args!("empty workspace convolution kernel")));
        }
        let output = if let Some(extra) = transposed {
            (n - 1) * stride_value - 2 * pad + dil * (kernel - 1) + i64::from(extra[i]) + 1
        } else {
            (n + 2 * pad - dil * (kernel - 1) - 1).div_euclid(stride_value) + 1
        };
        shape.push(extent(output, context)?);
    }
    shape.push(weight.shape()[0]);
    let mut owned_stride = context.metadata_vec(stride.len())?;
    owned_stride.extend_from_slice(stride);
    let mut owned_padding = context.metadata_vec(padding.len())?;
    owned_padding.extend_from_slice(padding);
    let mut owned_dilation = context.metadata_vec(dilation.len())?;
    owned_dilation.extend_from_slice(dilation);
    let owned_transposed = transposed
        .map(|values| {
            let mut owned = context.metadata_vec(values.len())?;
            owned.extend_from_slice(values);
            Ok::<_, Error>(owned)
        })
        .transpose()?;
    WorkspaceTensor::operation(
        WorkspaceOperationKind::Convolution {
            stride: owned_stride,
            padding: owned_padding,
            dilation: owned_dilation,
            groups,
            transposed: owned_transposed,
        },
        &[input, weight],
        &shape,
        promote(input.layout.dtype, weight.layout.dtype),
        context,
    )
}

// The retained policy uses the same metadata producer as its source census.
fn trace_multi_axis_rotary(
    position_ids: &WorkspaceTensor,
    spec: crate::multimodal::MultiAxisRotarySpecRef<'_>,
    prepared: bool,
    context: &WorkspaceContext,
) -> Result<(WorkspaceTensor, WorkspaceTensor), Error> {
    let mut axes = context.metadata_vec(spec.axes.len())?;
    axes.extend_from_slice(spec.axes);
    let spec = crate::multimodal::MultiAxisRotarySpec {
        axes,
        base: spec.base,
        minimum_position: spec.minimum_position,
        layout: spec.layout,
    };
    let dimensions = spec.dimensions_with_diagnostic(
        |message| context.metadata_error(message),
        |cause| context.metadata_source(cause),
    )?;
    let mut shape = context.metadata_vec(position_ids.shape().len())?;
    shape.extend_from_slice(position_ids.shape());
    if shape.len() < 2 || shape.last().copied() != Some(spec.axes.len() as i32) {
        return Err(context.metadata_error(format_args!(
            "multi-axis rotary position geometry differs from its axes"
        )));
    }
    *shape.last_mut().unwrap() = dimensions;
    let layout = context.layout(&shape, WorkspaceDtype::Float32)?;
    let mut outputs = context.metadata_vec(2)?;
    outputs.extend([layout.clone(), layout]);
    let mut output = context.execute(
        if prepared {
            WorkspaceOperationKind::PreparedMultiAxisRotary(spec)
        } else {
            WorkspaceOperationKind::MultiAxisRotary(spec)
        },
        &[position_ids],
        outputs,
    )?;
    let sine = output.pop().unwrap();
    Ok((output.pop().unwrap(), sine))
}

impl WorkspaceTensor {
    /// Trace the ordinary owner contribution and selected publication Sum.
    /// Native source identity, completion and transport remain independently
    /// required by the mechanism that supplies this context's collective facts.
    pub fn broadcast_publication(&self, group:eredu_core::CollectiveGroupId,
        root:usize, rank:usize, partitions:usize, context:&WorkspaceContext)
        ->Result<Self,Error> {
        if root>=partitions || rank>=partitions {
            return Err(context.metadata_error(format_args!("invalid selected publication rank")));
        }
        let contribution=if rank==root {self.clone()} else {self.multiply_scalar(0.0,context)?};
        Self::operation(WorkspaceOperationKind::Collective(super::WorkspaceCollective::Broadcast {
            group,root,rank,partitions,
        }),&[&contribution],contribution.shape(),contribution.layout.dtype,context)
    }
}

impl WorkspaceTensor {
    /// Metadata for the ordinary rank-two additive indexed update. The supplied
    /// IDs remain a borrowed tensor source; this does not claim their values.
    pub fn indexed_row_add(&self, indices:&Self, updates:&Self, context:&WorkspaceContext)->Result<Self,Error> {
        self.validate_context(context)?;indices.validate_context(context)?;updates.validate_context(context)?;
        if self.shape().len()!=2 || indices.shape().len()!=2 || updates.shape().len()!=2
            || indices.shape()[1]!=1 || indices.shape()[0]!=updates.shape()[0]
            || self.shape()[1]!=updates.shape()[1]
            || !matches!(indices.layout.dtype,WorkspaceDtype::Int32|WorkspaceDtype::Uint32)
            || self.layout.dtype!=updates.layout.dtype {
            return Err(context.metadata_error(format_args!("indexed row-add differs from its retained rank-two geometry")));
        }
        Self::operation(WorkspaceOperationKind::IndexedRowAdd,&[self,indices,updates],self.shape(),self.layout.dtype,context)
    }
}

impl WorkspaceTensor {
    /// Trace the existing one-index Gather from an independently validated
    /// native I32 source. This geometry does not authenticate index values.
    pub fn select_elements_with_indices(&self, indices: &Self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.validate_context(context)?;
        indices.validate_context(context)?;
        if self.shape().len() != 1 || self.shape()[0] <= 0
            || indices.shape().len() != 1 || indices.layout.dtype != WorkspaceDtype::Int32 {
            return Err(context.metadata_error(format_args!("indexed element selection requires rank-one values and I32 indices")));
        }
        Self::operation(WorkspaceOperationKind::IndexedElementSelect, &[self, indices], indices.shape(), self.layout.dtype, context)
    }
    /// Trace exact-shape general Scatter overwrite. Distinct in-range index
    /// values stay with the original sparse producer, outside this metadata.
    pub fn update_elements_with_indices(&self, indices: &Self, updates: &Self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.validate_context(context)?;
        indices.validate_context(context)?;
        updates.validate_context(context)?;
        if self.shape().len() != 1 || self.shape()[0] <= 0
            || indices.shape().len() != 1 || indices.layout.dtype != WorkspaceDtype::Int32
            || updates.shape() != indices.shape() || updates.layout.dtype != self.layout.dtype {
            return Err(context.metadata_error(format_args!("indexed element overwrite differs from its rank-one source geometry")));
        }
        Self::operation(WorkspaceOperationKind::IndexedElementUpdate, &[self, indices, updates], self.shape(), self.layout.dtype, context)
    }
}
