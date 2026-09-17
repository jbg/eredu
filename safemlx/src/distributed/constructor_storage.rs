//! Actual lazy graph constructor; no communication/Eval allowance is inferred.
use super::{Group,GroupWorkerOperation,GroupStorageUnavailable};
use crate::{Array,Stream,OriginalScopeObserver,error::{Exception,Result},
    utils::{guard::{Guard,MaybeUninitArray},runtime_lock}};
use std::{fmt,mem::{size_of,size_of_val}};

/// Exact retained group/input and lazy constructor recipe. Preparation borrows
/// both native sources and executes one ordinary graph constructor in the given
/// original Scope. The result still needs separately admitted evaluation,
/// communication, backing, completion and persistent communicator owners.
pub struct GroupConstructorStorage<'a> {
    group:&'a Group,
    input:&'a Array,
    operation:GroupWorkerOperation,
    value:safemlx_sys::mlx_distributed_constructor_storage,
    matrix: Option<(&'a [usize], bool)>,
}
impl fmt::Debug for GroupConstructorStorage<'_> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("GroupConstructorStorage").field("operation",&self.operation)
            .field("primitives",&self.value.primitives).field("blocks",&self.value.blocks).finish_non_exhaustive()
    }
}
impl Group {
    /// Pure source inspection. Unknown implementation/ABI, terminal groups and
    /// invalid output geometry refuse without changing native state.
    pub fn constructor_storage<'a>(&'a self,input:&'a Array,operation:GroupWorkerOperation)
        ->std::result::Result<GroupConstructorStorage<'a>,GroupStorageUnavailable> {
        let mut value=safemlx_sys::mlx_distributed_constructor_storage::default();
        let (code,peer)=operation.native();
        // SAFETY: retained immutable wrappers, metadata-only query, no error channel.
        if !unsafe {safemlx_sys::mlx_distributed_group_constructor_storage(&mut value,self.native.c_group,input.as_ptr(),code,peer)} {
            return Err(GroupStorageUnavailable);
        }
        Ok(GroupConstructorStorage{group:self,input,operation,value,matrix:None})
    }
    /// Fixed native/C/Rust inspection frames, payable before the source query.
    pub fn constructor_storage_control_bytes()->Option<usize> {
        // SAFETY: only fixed native type layouts are read.
        let native=unsafe{safemlx_sys::mlx_distributed_group_constructor_storage_controls()};
        let controls=[size_of::<GroupConstructorStorage<'_>>(),
            size_of::<std::result::Result<GroupConstructorStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&Array,GroupWorkerOperation)>(),size_of::<(u32,i32)>(),size_of::<GroupStorageUnavailable>()];
        controls.into_iter().try_fold(native.checked_add(size_of_val(&controls))?,usize::checked_add)
    }
}
impl<'a> GroupConstructorStorage<'a> {
    pub(super) fn from_variable(group:&'a Group,input:&'a Array,matrix:&'a [usize],transposed:bool,
        value:safemlx_sys::mlx_distributed_constructor_storage)->Self {
        Self { group,input,operation:GroupWorkerOperation::Variable,value,matrix:Some((matrix,transposed)) }
    }
    pub(super) fn matrix(&self)->Option<(&'a [usize],bool)> {self.matrix}
    /// Exact matrix-backed source identity, including orientation.
    pub fn is_for_variable(&self,group:&Group,input:&Array,matrix:&[usize],transposed:bool)->bool {
        std::ptr::eq(self.group,group) && std::ptr::eq(self.input,input) &&
            self.matrix.is_some_and(|(source,transpose)| std::ptr::eq(source,matrix) && transpose==transposed)
    }

    pub(super) fn native(&self)->&safemlx_sys::mlx_distributed_constructor_storage {&self.value}
    pub(super) fn source_parts(&self)->(&Group,&Array,GroupWorkerOperation) {
        (self.group,self.input,self.operation)
    }
    /// Exact operation retained from the actual group/input source.
    pub fn operation(&self)->GroupWorkerOperation{self.operation}
    /// Output rank and checked element population of the ordinary constructor.
    pub fn output_geometry(&self)->(usize,usize){(self.value.output_rank,self.value.output_elements)}
    /// Exact dtype of the retained operation's output. These communication
    /// constructors preserve their actual input dtype.
    pub fn output_dtype(&self)->crate::Dtype{self.input.dtype()}
    /// Actual primitive and input-edge populations; singleton identities are zero.
    pub fn graph_population(&self)->(usize,usize){(self.value.primitives,self.value.input_edges)}
    /// Number of physical constructor blocks, excluding the bank header/slots.
    pub fn blocks(&self)->usize{self.value.blocks}
    /// Requested payload bytes including bank header/slots and result C handle.
    pub fn requested_bytes(&self)->usize{self.value.requested_bytes}
    /// Per-request native allocator extent sum. Actual successful reservation,
    /// not this sum alone, proves physical fit in the current arena.
    pub fn graph_extent(&self)->usize{self.value.allocation_extents}
    /// Exact payload classes (bytes, alignment, count), separately reserved by
    /// the same GraphConstruction allocator used by other original constructors.
    pub fn classes(&self)->impl ExactSizeIterator<Item=(usize,usize,usize)>+'_ {
        self.value.request_bytes.iter().copied().zip(self.value.request_alignments.iter().copied())
            .zip(self.value.request_counts.iter().copied()).map(|((bytes,align),count)|(bytes,align,count))
    }
    /// Exact borrowed wrapper/input/operation identity.
    pub fn is_for(&self,group:&Group,input:&Array,operation:GroupWorkerOperation)->bool {
        self.matrix.is_none() && std::ptr::eq(self.group,group)&&std::ptr::eq(self.input,input)&&self.operation==operation
    }
    /// Fixed query, constructor, result-guard and error transport controls.
    pub fn construction_control_bytes(&self)->Option<usize> {
        let controls=[size_of::<Self>(),size_of::<Result<Array>>(),size_of::<MaybeUninitArray>(),
            size_of::<Array>(),size_of::<Exception>(),size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<(&OriginalScopeObserver,&Stream)>(),size_of::<Option<(&[usize],bool)>>(),size_of::<(u32,i32)>(),size_of::<u32>()];
        controls.into_iter().try_fold(self.value.named_control_bytes.checked_add(size_of_val(&controls))?.checked_add(OriginalScopeObserver::control_bytes()?)?,usize::checked_add)
    }
    /// Consume one source-bound constructor attempt. Revalidates actual native
    /// source and geometry, physically reserves the same bank, and returns a
    /// lazy ordinary result. No native task or event is submitted here.
    pub fn construct_original(self,observer:&OriginalScopeObserver,stream:&Stream)->Result<Array> {
        let Some(_loan)=runtime_lock::try_enter_for_recovery() else {return Err(observer.error(10));};
        let mut result=MaybeUninitArray::new();
        let (code,peer)=self.operation.native();
        // SAFETY: exclusive fresh result guard, actual retained group/input and
        // stream, exact observer. Native preparation recomputes all scalar facts
        // and retires its unique bank before returning, including every failure.
        let status=if let Some((matrix,transposed))=self.matrix {
            // SAFETY: the complete count matrix is retained by this same source
            // through the constructor, which copies it into its paid owner.
            unsafe{safemlx_sys::mlx_distributed_construct_variable_original(result.as_mut_raw_ptr(),observer.raw,
                self.group.native.c_group,self.input.as_ptr(),matrix.as_ptr(),matrix.len(),transposed,stream.as_ptr())}
        } else {
            unsafe{safemlx_sys::mlx_distributed_construct_original(result.as_mut_raw_ptr(),observer.raw,
                self.group.native.c_group,self.input.as_ptr(),code,peer,stream.as_ptr())}
        };
        if status!=0{return Err(observer.error(status));}
        result.set_init_success(true);
        result.try_into_guarded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_singleton_constructor_is_only_a_source_bound_handle() {
        let group=Group::init(false,super::super::Backend::Ring).unwrap();
        let input=Array::from_slice(&[7_i32,-11,23],&[3]);
        let other=Array::from_slice(&[7_i32,-11,23],&[3]);
        assert!(Group::constructor_storage_control_bytes().is_some_and(|n|n>0));
        for operation in [GroupWorkerOperation::Sum,GroupWorkerOperation::Maximum,GroupWorkerOperation::Minimum,GroupWorkerOperation::Gather] {
            let source=group.constructor_storage(&input,operation).unwrap();
            assert_eq!(source.graph_population(),(0,0));
            assert_eq!(source.output_geometry(),(1,3));
            assert_eq!(source.blocks(),1);
            assert_eq!(source.classes().map(|(_,_,n)|n).sum::<usize>(),1);
            assert!(source.is_for(&group,&input,operation));
            assert!(!source.is_for(&group,&other,operation));
            assert!(source.graph_extent()>source.requested_bytes());
            assert!(source.construction_control_bytes().is_some_and(|n|n>0));
        }
        assert!(matches!(group.constructor_storage(&input,GroupWorkerOperation::Send{peer:0}),Err(GroupStorageUnavailable)));
        assert!(matches!(group.constructor_storage(&input,GroupWorkerOperation::Receive{peer:0}),Err(GroupStorageUnavailable)));
    }
}
