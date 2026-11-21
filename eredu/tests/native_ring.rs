#![cfg(unix)]

#[path = "native_ring/partition.rs"]
mod partition;
#[path = "native_ring/realtime.rs"]
mod realtime;
#[path = "native_ring/row_parallel.rs"]
mod row_parallel;
#[path = "native_ring/workers.rs"]
mod workers;
