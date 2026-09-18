pub mod compiler;
mod formats;
mod numeric;
mod source_text;
mod schema;
mod shared_context;

mod context_ref;
pub mod context { pub use super::context_ref::*; }

use std::{any::type_name_of_val, sync::Arc};

use serde_json::Value;
pub trait Retrieve: Send + Sync {
    fn retrieve(&self, uri: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Clone)]
pub struct RetrieveWrapper(pub Arc<dyn Retrieve>);
impl RetrieveWrapper {
    pub fn new(retrieve: Arc<dyn Retrieve>) -> Self {
        RetrieveWrapper(retrieve)
    }
}

impl std::fmt::Debug for RetrieveWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", type_name_of_val(&self.0))
    }
}
