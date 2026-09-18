use core::ops;

/// A specialized copy-on-write byte string.
///
/// The purpose of this type is to permit usage of a "borrowed or owned
/// byte string" in a way that keeps std/no-std compatibility. That is, in
/// no-std/alloc mode, this type devolves into a simple &[u8] with no owned
/// variant available. We can't just use a plain Cow because Cow is not in
/// core.
#[derive(Clone, Debug)]
pub struct CowBytes<'a>(Imp<'a>);

// N.B. We don't use alloc::borrow::Cow here since we can get away with a
// Box<[u8]> for our use case, which is 1/3 smaller than the Vec<u8> that
// a Cow<[u8]> would use.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
enum Imp<'a> {
    Borrowed(&'a [u8]),
    Owned(alloc::boxed::Box<[u8]>),
}

#[cfg(not(feature = "alloc"))]
#[derive(Clone, Debug)]
struct Imp<'a>(&'a [u8]);

impl<'a> ops::Deref for CowBytes<'a> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<'a> CowBytes<'a> {
    pub(crate) fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(*const (), usize) -> bool,
    ) {
        #[cfg(feature = "alloc")]
        if let Imp::Owned(bytes) = &self.0 {
            if !bytes.is_empty() {
                visitor(bytes.as_ptr().cast::<()>(), bytes.len());
            }
        }
        #[cfg(not(feature = "alloc"))]
        let _ = visitor;
    }

    /// Create a new borrowed CowBytes.
    #[inline(always)]
    pub(crate) fn new<B: ?Sized + AsRef<[u8]>>(bytes: &'a B) -> CowBytes<'a> {
        CowBytes(Imp::new(bytes.as_ref()))
    }

    /// Create a new owned CowBytes.
    #[cfg(feature = "alloc")]
    #[inline(always)]
    pub(crate) fn new_owned(
        bytes: alloc::boxed::Box<[u8]>,
    ) -> CowBytes<'static> {
        CowBytes(Imp::Owned(bytes))
    }

    /// Return a borrowed byte string, regardless of whether this is an owned
    /// or borrowed byte string internally.
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn into_owned_with_allocations(
        self,
        funding: &dyn crate::allocation::Allocation,
    ) -> Result<CowBytes<'static>, crate::allocation::AllocationError> {
        Ok(match self.0 {
            Imp::Borrowed(bytes) => {
                CowBytes::new_owned(crate::allocation::copy(bytes, funding)?)
            }
            Imp::Owned(bytes) => CowBytes::new_owned(bytes),
        })
    }
}

impl<'a> Imp<'a> {
    #[inline(always)]
    pub fn new(bytes: &'a [u8]) -> Imp<'a> {
        #[cfg(feature = "alloc")]
        {
            Imp::Borrowed(bytes)
        }
        #[cfg(not(feature = "alloc"))]
        {
            Imp(bytes)
        }
    }

    #[cfg(feature = "alloc")]
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        #[cfg(feature = "alloc")]
        {
            match self {
                Imp::Owned(ref x) => x,
                Imp::Borrowed(x) => x,
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.0
        }
    }

    #[cfg(not(feature = "alloc"))]
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.0
    }
}
