// Test-only pre-refactor workers. Only recompose's module-relative import
// differs from the exact pinned archive; no algorithm or data changes.
#![allow(dead_code)]
pub use ::char;
pub mod decompose;
pub mod recompose;
