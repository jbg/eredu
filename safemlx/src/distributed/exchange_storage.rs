//! Actual pair source for the ordinary logical exchange's two native edges.
use super::{GroupCpuOperationStorage,GroupCpuLayoutStorage,GroupStorageUnavailable,GroupWorkerOperation};
use crate::{Array,OperationEvent,OperationEvalTraversalLayout,OperationEvalTraversalLimits,
    OriginalScopeObserver,Stream,error::Result};
use std::mem::{size_of,size_of_val};

/// Same native group and input, one send and one receive, followed by the exact
/// two-root CPU completion. The caller separately supplies backing, persistent
/// communicator custody and admission. No logical membership is inferred here.
#[derive(Debug)]
pub struct GroupCpuExchangeStorage<'a> {
    send:GroupCpuOperationStorage<'a>,
    receive:GroupCpuOperationStorage<'a>,
    traversal:OperationEvalTraversalLayout,
    native:safemlx_sys::mlx_distributed_cpu_completion_storage,
}
fn exchange_control_bytes(group:&super::Group) -> Option<usize> {
        // SAFETY: immutable actual group selects the existing pure producer.
        let native=unsafe{safemlx_sys::mlx_distributed_cpu_exchange_storage_controls(group.native.c_group)};
        if native==usize::MAX{return None;}
        let parts=[size_of::<GroupCpuExchangeStorage<'_>>(),size_of::<[GroupCpuOperationStorage<'_>;2]>(),
            size_of::<std::result::Result<GroupCpuExchangeStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&GroupCpuOperationStorage<'_>,&GroupCpuOperationStorage<'_>)>(),size_of::<(i32,i32)>(),
            size_of::<(&super::Group,&Array,&super::Group,&Array,GroupWorkerOperation,GroupWorkerOperation)>(),
            size_of::<OperationEvalTraversalLayout>(),size_of::<OperationEvalTraversalLimits>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>(),
            size_of::<Option<OperationEvalTraversalLayout>>(),size_of::<bool>(),
            size_of::<GroupStorageUnavailable>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
}
impl<'a> GroupCpuOperationStorage<'a> {
    /// Fixed shared query and owning transports, payable before pair selection.
    pub fn exchange_storage_control_bytes(&self)->Option<usize> {
        exchange_control_bytes(self.constructor().source_parts().0)?
            .checked_add(size_of::<&Self>())?.checked_add(size_of::<Option<usize>>())
    }
    /// Join exactly the actual send and receive source. Equal-sized arrays or
    /// distinct wrappers for the same native group are not substitute sources.
    pub fn with_exchange_storage(self,receive:Self)
        ->std::result::Result<GroupCpuExchangeStorage<'a>,GroupStorageUnavailable>
    {
        if !std::ptr::eq(self.constructor().source_parts().1, receive.constructor().source_parts().1) {
            return Err(GroupStorageUnavailable);
        }
        self.with_asymmetric_exchange_storage(receive)
    }
    /// Join two actual completed prototypes on the identical retained native
    /// Group. Only Send retains a DAG input; Receive owns its exact output shape.
    pub fn with_asymmetric_exchange_storage(self, receive: Self)
        -> std::result::Result<GroupCpuExchangeStorage<'a>, GroupStorageUnavailable> {
        let (group,input,operation)=self.constructor().source_parts();
        let (other_group,other_input,other_operation)=receive.constructor().source_parts();
        let (GroupWorkerOperation::Send{peer:destination},GroupWorkerOperation::Receive{peer:source})
            =(operation,other_operation) else {return Err(GroupStorageUnavailable);};
        if !std::ptr::eq(group,other_group) || input.dtype()!=other_input.dtype(){return Err(GroupStorageUnavailable);}
        let mut native=safemlx_sys::mlx_distributed_cpu_completion_storage::default();
        // SAFETY: closed actual group/input loans and initialized scalar result.
        if !unsafe{safemlx_sys::mlx_distributed_query_cpu_exchange_sources_storage(&mut native,
            group.native.c_group,input.as_ptr(),other_input.as_ptr(),destination,source)} {return Err(GroupStorageUnavailable);}
        let n=native.traversal.limits;
        let traversal=OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits{
            roots:n.root_count,arrays:n.array_nodes,tape_entries:n.tape_entries,input_edges:n.input_edges,
            output_slots:n.output_slots,streams:n.stream_count,captures:n.capture_slots,
        }).ok_or(GroupStorageUnavailable)?;
        Ok(GroupCpuExchangeStorage{send:self,receive,traversal,native})
    }
}
impl GroupCpuExchangeStorage<'_> {
    /// Exact shared completion traversal including both native output roots.
    pub fn traversal(&self)->OperationEvalTraversalLayout{self.traversal}
    /// Fresh Graph capacity for both native operations and one completion.
    pub fn graph_capacity(&self)->usize{self.native.graph_capacity}
    /// Fresh Record capacity for the one shared traversal.
    pub fn record_capacity(&self)->usize{self.native.record_capacity}
    /// Platform Event population of the same final completion.
    pub fn platform_events(&self)->usize{self.native.platform_events}
    /// Actual send source, including any contiguity copy producer.
    pub fn send(&self)->&GroupCpuOperationStorage<'_>{&self.send}
    /// Actual receive source and its output allocation producer.
    pub fn receive(&self)->&GroupCpuOperationStorage<'_>{&self.receive}
    /// Fixed construction, validation, root-array and failure transports.
    pub fn construction_control_bytes(&self)->Option<usize>{
        let parts=[size_of::<Self>(),size_of::<[Array;2]>(),size_of::<Result<[Array;2]>>(),
            size_of::<(&Self,&OriginalScopeObserver,&Stream)>(),
            self.send.control_bytes()?,self.receive.control_bytes()?,self.traversal.query_control_bytes()?,
            OperationEvent::traversal_leaf_control_bytes()?.checked_mul(2)?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Construct both existing operations before either is submitted. A failed
    /// second constructor retires the first lazy result without submitting it.
    pub fn construct_original(self,observer:&OriginalScopeObserver,stream:&Stream)->Result<[Array;2]>{
        OperationEvent::validate_traversal_context(observer)?;
        OperationEvent::validate_traversal_leaf(self.send.constructor().source_parts().1,observer)?;
        if !std::ptr::eq(self.send.constructor().source_parts().1, self.receive.constructor().source_parts().1) {
            OperationEvent::validate_traversal_leaf(self.receive.constructor().source_parts().1,observer)?;
        }
        let send=self.send.construct_original(observer,stream)?;
        let receive=self.receive.construct_original(observer,stream)?;
        Ok([send,receive])
    }
}

/// Exact Send/Receive constructor, CPU and paired completion facts sourced from
/// one immutable layout and actual retained Group. This owns no native input,
/// allocation permission, event or successful-completion claim.
#[derive(Debug)]
pub struct GroupCpuExchangeLayoutStorage<'a> {
    send: GroupCpuLayoutStorage<'a>,
    receive: GroupCpuLayoutStorage<'a>,
    traversal: OperationEvalTraversalLayout,
    native: safemlx_sys::mlx_distributed_cpu_completion_storage,
}
impl<'a> GroupCpuLayoutStorage<'a> {
    /// Fixed controls before the same paired worker is queried with a layout.
    pub fn exchange_layout_control_bytes(&self) -> Option<usize> {
        let (group,_,_,_)=self.source_parts();
        // SAFETY: an actual immutable group selects the shared source query.
        let native=unsafe{safemlx_sys::mlx_distributed_cpu_exchange_layout_storage_controls(group.native.c_group)};
        if native==usize::MAX { return None; }
        let parts=[size_of::<GroupCpuExchangeLayoutStorage<'a>>(),size_of::<[Self;2]>(),
            size_of::<std::result::Result<GroupCpuExchangeLayoutStorage<'a>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&Self)>(),size_of::<(i32,i32)>(),
            size_of::<(&super::Group,&[i32],crate::Dtype,GroupWorkerOperation)>(),
            size_of::<(&super::Group,&[i32],crate::Dtype,GroupWorkerOperation)>(),
            size_of::<OperationEvalTraversalLayout>(),size_of::<OperationEvalTraversalLimits>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>(),
            size_of::<Option<OperationEvalTraversalLayout>>(),size_of::<bool>(),
            size_of::<GroupStorageUnavailable>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
    /// Joins the exact same borrowed Group/layout as one ordered Send/Receive
    /// pair. Equal scalar geometry cannot substitute either source identity.
    pub fn with_exchange_layout(self,receive:Self)
        ->std::result::Result<GroupCpuExchangeLayoutStorage<'a>,GroupStorageUnavailable> {
        if !std::ptr::eq(self.source_parts().1, receive.source_parts().1) {
            return Err(GroupStorageUnavailable);
        }
        self.with_asymmetric_exchange_layout(receive)
    }
    /// Same native pair with two independently retained immutable layouts. The
    /// selected Group and scalar representation must still be identical.
    pub fn with_asymmetric_exchange_layout(self, receive: Self)
        -> std::result::Result<GroupCpuExchangeLayoutStorage<'a>, GroupStorageUnavailable> {
        let (group,shape,dtype,operation)=self.source_parts();
        let (other_group,other_shape,other_dtype,other_operation)=receive.source_parts();
        let (GroupWorkerOperation::Send{peer:destination},GroupWorkerOperation::Receive{peer:source})
            =(operation,other_operation) else { return Err(GroupStorageUnavailable); };
        if !std::ptr::eq(group,other_group) || dtype!=other_dtype {
            return Err(GroupStorageUnavailable);
        }
        let mut native=safemlx_sys::mlx_distributed_cpu_completion_storage::default();
        // SAFETY: closed Group and bounded immutable shape loans; the native
        // shared worker writes initialized scalar facts only after validation.
        if !unsafe{safemlx_sys::mlx_distributed_query_cpu_exchange_layouts_storage(&mut native,
            group.native.c_group,shape.as_ptr(),shape.len(),other_shape.as_ptr(),other_shape.len(),dtype.into(),destination,source)} {
            return Err(GroupStorageUnavailable);
        }
        let n=native.traversal.limits;
        let traversal=OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits{
            roots:n.root_count,arrays:n.array_nodes,tape_entries:n.tape_entries,input_edges:n.input_edges,
            output_slots:n.output_slots,streams:n.stream_count,captures:n.capture_slots,
        }).ok_or(GroupStorageUnavailable)?;
        Ok(GroupCpuExchangeLayoutStorage{send:self,receive,traversal,native})
    }
}
impl GroupCpuExchangeLayoutStorage<'_> {
    /// Same two-output completion traversal as the executing pair.
    pub fn traversal(&self) -> OperationEvalTraversalLayout { self.traversal }
    /// Actual shared constructor/Eval/worker and completion Graph capacity.
    pub fn graph_capacity(&self) -> usize { self.native.graph_capacity }
    /// Record capacity for the one paired completion.
    pub fn record_capacity(&self) -> usize { self.native.record_capacity }
    /// Fixed shared paired-completion worker controls.
    pub fn execution_control_bytes(&self) -> usize { self.native.named_control_bytes }
    /// Exact Send source, including its possible contiguous copy.
    pub fn send(&self) -> &GroupCpuLayoutStorage<'_> { &self.send }
    /// Exact Receive source and output backing population.
    pub fn receive(&self) -> &GroupCpuLayoutStorage<'_> { &self.receive }
    /// Source recomputation, complete result and comparison transports payable
    /// before an actual completed input replaces this immutable layout loan.
    pub fn binding_control_bytes(&self) -> Option<usize> {
        let parts=[size_of::<Self>(),size_of::<GroupCpuExchangeStorage<'_>>(),
            size_of::<[GroupCpuOperationStorage<'_>;2]>(),size_of::<(&Self,&Array,&Array)>(),
            size_of::<std::result::Result<GroupCpuExchangeStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<bool>(),size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>(),
            size_of::<&GroupCpuOperationStorage<'_>>(),size_of::<Option<usize>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?
            .checked_add(self.send.binding_control_bytes()?)?
            .checked_add(self.receive.binding_control_bytes()?)?
            .checked_add(exchange_control_bytes(self.send.source_parts().0)?)
    }
}
impl<'a> GroupCpuExchangeLayoutStorage<'a> {
    /// Recomputes both real input sources and their paired completion before
    /// accepting a constructor. Lazy or foreign input completion still refuses.
    pub fn bind_actual<'input>(self,input:&'input Array)
        ->std::result::Result<GroupCpuExchangeStorage<'input>,GroupStorageUnavailable> where 'a:'input {
        self.bind_actual_pair(input, input)
    }
    /// Bind each quoted layout to its actual completed prototype before using
    /// the unchanged constructors and shared two-root completion source.
    pub fn bind_actual_pair<'input>(self, input: &'input Array, receive_like: &'input Array)
        -> std::result::Result<GroupCpuExchangeStorage<'input>, GroupStorageUnavailable> where 'a: 'input {
        let send=self.send.bind_actual(input).map_err(|_|GroupStorageUnavailable)?;
        let receive=self.receive.bind_actual(receive_like).map_err(|_|GroupStorageUnavailable)?;
        let actual=send.with_asymmetric_exchange_storage(receive)?;
        if actual.traversal!=self.traversal || actual.native.graph_allocation_extents!=self.native.graph_allocation_extents
            || actual.native.graph_capacity!=self.native.graph_capacity || actual.native.record_allocation_extents!=self.native.record_allocation_extents
            || actual.native.record_capacity!=self.native.record_capacity || actual.native.synchronizer_graph_extent!=self.native.synchronizer_graph_extent
            || actual.native.signal_graph_extent!=self.native.signal_graph_extent || actual.native.platform_events!=self.native.platform_events {
            return Err(GroupStorageUnavailable);
        }
        Ok(actual)
    }
}

/// Retained pair of the exact cold send/receive sources and shared completion.
/// Both handles retain their actual native incarnation. This carries no input,
/// Graph/Data grant, or completed-operation claim.
#[derive(Debug)]
pub struct OwnedGroupCpuExchangeLayoutStorage {
    send: super::OwnedGroupCpuLayoutStorage,
    receive: super::OwnedGroupCpuLayoutStorage,
    traversal: OperationEvalTraversalLayout,
    native: safemlx_sys::mlx_distributed_cpu_completion_storage,
}
impl GroupCpuExchangeLayoutStorage<'_> {
    /// All fixed and shape/native-handle destinations before retaining this pair.
    pub fn ownership_control_bytes(&self) -> Option<usize> {
        let parts=[size_of::<OwnedGroupCpuExchangeLayoutStorage>(),size_of::<Self>(),
            size_of::<std::result::Result<OwnedGroupCpuExchangeLayoutStorage,std::collections::TryReserveError>>(),
            self.send.ownership_control_bytes()?,self.receive.ownership_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Moves these already-queried facts into paid retained owners. No native
    /// availability query is repeated after this source has been prepared.
    pub fn try_into_owned(self)
        -> std::result::Result<OwnedGroupCpuExchangeLayoutStorage,std::collections::TryReserveError> {
        Ok(OwnedGroupCpuExchangeLayoutStorage{send:self.send.try_into_owned()?,
            receive:self.receive.try_into_owned()?,traversal:self.traversal,native:self.native})
    }
}
impl OwnedGroupCpuExchangeLayoutStorage {
    fn view(&self) -> std::result::Result<GroupCpuExchangeLayoutStorage<'_>,GroupStorageUnavailable> {
        let send=self.send.view();
        // The two owned aliases refer to the same actual native source. Lend
        // one common wrapper to the unchanged pointer-exact pair binder.
        let receive=self.receive.view_with_group(send.source_parts().0)?;
        Ok(GroupCpuExchangeLayoutStorage{send,receive,traversal:self.traversal,native:self.native})
    }
    /// Exact Graph capacity queried before the input was constructed.
    pub fn graph_capacity(&self)->usize{self.native.graph_capacity}
    /// Exact shared completion Record capacity.
    pub fn record_capacity(&self)->usize{self.native.record_capacity}
    /// Actual cold send-copy and receive-output allocation populations.
    pub fn maximum_backing_births(&self)->Option<usize>{
        self.send.evaluation().logical_backing_population().0
            .checked_add(self.receive.evaluation().logical_backing_population().0)
    }
    /// Actual physical scalar representation of the quoted input.
    pub fn dtype(&self)->crate::Dtype{self.send.dtype()}
    /// The exact quoted input geometry.
    pub fn shape(&self)->&[i32]{self.send.shape()}
    /// The independently retained Receive prototype geometry.
    pub fn receive_shape(&self)->&[i32]{self.receive.shape()}
    /// Both source handles retain this same native incarnation.
    pub fn is_for_group(&self,group:&super::Group)->bool{
        self.send.is_for_group(group)&&self.receive.is_for_group(group)
    }
    /// Controls for exact input revalidation and the common accepted binder.
    pub fn binding_control_bytes(&self)->Option<usize>{
        self.view().ok()?.binding_control_bytes()?.checked_add(size_of::<Self>())?
            .checked_add(size_of::<std::result::Result<GroupCpuExchangeLayoutStorage<'_>,GroupStorageUnavailable>>())
    }
    /// The common binder still recomputes both actual input producers and the
    /// paired completion, comparing them with these retained immutable facts.
    pub fn bind_actual<'a>(&'a self,input:&'a Array)
        -> std::result::Result<GroupCpuExchangeStorage<'a>,GroupStorageUnavailable>{
        self.view()?.bind_actual(input)
    }
    /// Bind independently authenticated prototypes to this exact retained pair.
    pub fn bind_actual_pair<'a>(&'a self, input: &'a Array, receive_like: &'a Array)
        -> std::result::Result<GroupCpuExchangeStorage<'a>, GroupStorageUnavailable> {
        self.view()?.bind_actual_pair(input, receive_like)
    }
}
