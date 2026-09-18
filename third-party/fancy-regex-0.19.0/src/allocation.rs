//! Prospective storage admission for the original regex workers.
//!
//! The caller retains its actual funding owner through outputs and errors.
//! These primitives do not themselves qualify a complete regex constructor.
use alloc::{string::String, sync::Arc};
use core::{
    alloc::Layout,
    fmt,
    hash::{BuildHasher, Hash},
    sync::atomic::AtomicUsize,
};
use regex_syntax::allocation::Allocator;
pub use regex_syntax::allocation::{Allocation, AllocationError, Unenforced};

#[derive(Clone, Copy)]
pub(crate) struct Context<'a> {
    pub(crate) policy: &'a dyn Allocation,
    pub(crate) storage: Allocator<'a>,
}
impl fmt::Debug for Context<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AllocationContext")
    }
}
impl<'a> Context<'a> {
    pub(crate) fn new(policy: &'a dyn Allocation) -> Self {
        Self {
            policy,
            storage: Allocator::new(policy),
        }
    }
    pub(crate) fn unenforced() -> Self {
        Self::new(&Unenforced)
    }
    pub(crate) fn fixed() -> Self {
        struct Fixed;
        impl Allocation for Fixed {
            fn reserve(&self, _: usize) -> Result<(), AllocationError> {
                Err(AllocationError::Refused)
            }
        }
        Self::new(&Fixed)
    }
    pub(crate) fn arc<T>(self, value: T) -> Result<Arc<T>, AllocationError> {
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<T>())
            .map_err(|_| AllocationError::SizeOverflow)?;
        self.storage.reserve(layout.pad_to_align().size())?;
        Ok(Arc::new(value))
    }
    pub(crate) fn insert<K: Eq + Hash, V, S: BuildHasher>(
        self,
        map: &mut hashbrown::HashMap<K, V, S>,
        key: K,
        value: V,
    ) -> Result<Option<V>, AllocationError> {
        if !map.contains_key(&key) {
            if let Some(layout) = map
                .try_reserve_layout(1)
                .map_err(|_| AllocationError::SizeOverflow)?
            {
                self.storage.reserve(layout.size())?;
            }
            map.try_reserve(1)
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        Ok(map.insert(key, value))
    }
    pub(crate) fn compile_error(self, error: crate::CompileError) -> crate::Error {
        let allocation_error = match &error {
            crate::CompileError::InnerError(error) => error.allocation_error(),
            crate::CompileError::PikeVmBuildError(error) => error.allocation_error(),
            crate::CompileError::WorkspaceDfaBuildError(error) => error.allocation_error(),
            _ => None,
        };
        if let Some(error) = allocation_error {
            return crate::Error::Allocation(error.into());
        }
        match self.storage.boxed(error) {
            Ok(error) => crate::Error::CompileError(error),
            Err(error) => crate::Error::Allocation(error),
        }
    }
    pub(crate) fn format(self, arguments: fmt::Arguments<'_>) -> Result<String, AllocationError> {
        struct Writer<'a> {
            context: Context<'a>,
            output: String,
            failure: Option<AllocationError>,
        }
        impl fmt::Write for Writer<'_> {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                match self.context.storage.push_str(&mut self.output, text) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        self.failure = Some(error);
                        Err(fmt::Error)
                    }
                }
            }
        }
        let mut writer = Writer {
            context: self,
            output: String::new(),
            failure: None,
        };
        fmt::write(&mut writer, arguments)
            .map_err(|_| writer.failure.unwrap_or(AllocationError::SizeOverflow))?;
        Ok(writer.output)
    }
}

impl regex_automata::util::allocation::Allocation for Context<'_> {
    fn reserve(
        &self,
        bytes: usize,
    ) -> Result<(), regex_automata::util::allocation::AllocationError> {
        self.policy.reserve(bytes).map_err(Into::into)
    }
}
