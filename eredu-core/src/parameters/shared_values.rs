//! Shared parameter results retain the producer's custody through every alias.
use super::{ParameterProjectionValues, ParameterValues};
use serde::{Deserialize, Serialize};
use std::{fmt, ops::Deref, sync::Arc};

trait Custody: Send + Sync {
    fn retire(self: Box<Self>);
}
impl<T: Send + Sync> Custody for T {
    fn retire(self: Box<Self>) {
        // Moving out frees the concrete Box before releasing its funding.
        let value = *self;
        drop(value);
    }
}
struct CustodyOwner(Option<Box<dyn Custody>>);
impl Drop for CustodyOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
struct Owned<T> {
    // Declaration order also preserves payload-before-custody on unwinding.
    values: T,
    _custody: CustodyOwner,
}
struct Owner<T>(Option<Arc<Owned<T>>>);
impl<T> Clone for Owner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> Owner<T> {
    fn get(&self) -> &Arc<Owned<T>> {
        self.0.as_ref().expect("live parameter result")
    }
    fn retain(values: T, custody: impl Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(Owned {
            values,
            _custody: CustodyOwner(Some(Box::new(custody))),
        })))
    }
    fn control_bytes<C>() -> Option<u64> {
        let arc = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Owned<T>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        u64::try_from(arc.checked_add(std::mem::size_of::<C>())?).ok()
    }
}
impl<T> Drop for Owner<T> {
    fn drop(&mut self) {
        // No Weak or raw Arc escapes. Every alias consumes its Arc, so the last
        // one frees the shared control before dropping payload and custody.
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

macro_rules! shared_result {
    ($name:ident, $dto:ident, $borrow:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// Cloning shares all strings, geometry and values with the original
        /// producer's retained custody. Serialization preserves the DTO wire
        /// format. This handle alone grants no admission or execution authority.
        #[derive(Clone)]
        pub struct $name(Owner<$dto>);
        impl $name {
            /// Shared control and concrete custody allocation sizes, excluding
            /// the DTO's separately allocated strings, geometry and values.
            #[doc(hidden)]
            pub fn retained_control_bytes<C: Send + Sync + 'static>() -> Option<u64> {
                Owner::<$dto>::control_bytes::<C>()
            }
            /// Retains an already funded result and its actual producer custody.
            /// The caller accounts for payload and controls before construction;
            /// this bridge establishes neither funding nor native completion.
            #[doc(hidden)]
            pub fn retain(values: $dto, custody: impl Send + Sync + 'static) -> Self {
                Self(Owner::retain(values, custody))
            }
            /// Shares a caller-owned DTO without granting admission or funding.
            pub fn from_caller_owned(values: $dto) -> Self {
                Self::retain(values, ())
            }
            /// Borrows the retained DTO. An explicit deep clone creates an
            /// independent caller-owned export with no allocation authority.
            pub fn $borrow(&self) -> &$dto {
                &self.0.get().values
            }
            /// Whether both aliases retain the same result allocation owner.
            pub fn same_storage(&self, other: &Self) -> bool {
                Arc::ptr_eq(self.0.get(), other.0.get())
            }
        }
        impl Deref for $name {
            type Target = $dto;
            fn deref(&self) -> &Self::Target {
                self.$borrow()
            }
        }
        impl AsRef<$dto> for $name {
            fn as_ref(&self) -> &$dto {
                self.$borrow()
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.$borrow() == other.$borrow()
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.$borrow().fmt(f)
            }
        }
        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.$borrow().serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                $dto::deserialize(deserializer).map(Self::from_caller_owned)
            }
        }
    };
}
shared_result!(
    SharedParameterValues,
    ParameterValues,
    as_values,
    "Read-only aliases of one bounded effective parameter result."
);
shared_result!(
    SharedParameterProjectionValues,
    ParameterProjectionValues,
    as_projection,
    "Read-only aliases of one bounded native parameter projection result."
);

#[cfg(test)]
mod tests;
