use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use fluent_uri::Uri;
use hashbrown::{Equivalent, HashMap};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

use crate::{
    allocation::{Allocation, Allocator},
    uri, Error,
};

type CacheMap = HashMap<(String, String), Arc<Uri<String>>>;

struct StrPair<'a>(&'a str, &'a str);

impl Hash for StrPair<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl Equivalent<(String, String)> for StrPair<'_> {
    fn equivalent(&self, (a, b): &(String, String)) -> bool {
        self.0 == a && self.1 == b
    }
}

#[derive(Debug)]
pub(crate) struct UriCache<'a> {
    allocation: Allocator<'a>,
    cache: CacheMap,
}

impl<'a> UriCache<'a> {
    pub(crate) fn with_allocations(allocation: &'a dyn Allocation) -> Self {
        Self {
            allocation: Allocator(allocation),
            cache: HashMap::new(),
        }
    }

    pub(crate) fn allocation(&self) -> Allocator<'a> {
        self.allocation
    }

    pub(crate) fn resolve_against(
        &mut self,
        base: &Uri<&str>,
        uri: impl AsRef<str>,
    ) -> Result<Arc<Uri<String>>, Error> {
        let base_str = base.as_str();
        let reference = uri.as_ref();
        if let Some(cached) = self.cache.get(&StrPair(base_str, reference)) {
            return Ok(Arc::clone(cached));
        }
        let resolved = self.allocation.arc(uri::resolve_against_with_allocations(
            base,
            reference,
            self.allocation.0,
        )?)?;
        self.allocation.insert(
            &mut self.cache,
            (
                self.allocation.copy_str(base_str)?,
                self.allocation.copy_str(reference)?,
            ),
            Arc::clone(&resolved),
        )?;
        Ok(resolved)
    }

    pub(crate) fn into_shared(self) -> SharedUriCache<'a> {
        SharedUriCache {
            allocation: self.allocation,
            cache: RwLock::new(self.cache),
        }
    }
}

/// A dedicated type for URI resolution caching.
#[derive(Debug)]
pub(crate) struct SharedUriCache<'a> {
    allocation: Allocator<'a>,
    cache: RwLock<CacheMap>,
}

impl<'a> SharedUriCache<'a> {
    pub(crate) fn allocation(&self) -> Allocator<'a> {
        self.allocation
    }
    pub(crate) fn resolve_against(
        &self,
        base: &Uri<&str>,
        uri: impl AsRef<str>,
    ) -> Result<Arc<Uri<String>>, Error> {
        let base_str = base.as_str();
        let reference = uri.as_ref();
        let lookup = StrPair(base_str, reference);

        if let Some(cached) = self.cache.read().get(&lookup) {
            return Ok(Arc::clone(cached));
        }

        let cache = self.cache.upgradable_read();
        if let Some(cached) = cache.get(&lookup) {
            return Ok(Arc::clone(cached));
        }

        let resolved = self.allocation.arc(uri::resolve_against_with_allocations(
            base,
            reference,
            self.allocation.0,
        )?)?;
        let mut cache = RwLockUpgradableReadGuard::upgrade(cache);
        self.allocation.insert(
            &mut cache,
            (
                self.allocation.copy_str(base_str)?,
                self.allocation.copy_str(reference)?,
            ),
            Arc::clone(&resolved),
        )?;
        Ok(resolved)
    }
}
