//! Test-owned expert assignment and routing over generic MLX collectives.
//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks. [`crate::tests::support::expert_dispatch::dispatch_replicated`]
//! exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer. Sharded-token dispatch uses one reusable
//! [`AllToAllVPlan`](crate::tests::support::expert_dispatch::AllToAllVPlan)
//! and compact variable-count exchanges in both directions.

use std::{
    cell::Cell,
    time::{Duration, Instant},
};

use eredu_nn::TensorParallelGroupedOutput;
use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp, r#where, zeros_dtype},
    transforms::{depends, eval},
    Array, Dtype, Stream,
};

use crate::{
    backend::compaction::{compact_indices, count_nonzero},
    backend::error::Error,
    backend::nn::grouped::{PackedGatedProductGroups, PackedRelu2Groups},
    backend::nn::grouping::segment_sum_by_index,
    backend::runtime::distributed::{self as distributed, Group},
};

include!("expert_dispatch/provider.rs");
include!("expert_dispatch/timing.rs");
include!("expert_dispatch/assignment.rs");
include!("expert_dispatch/telemetry.rs");
include!("expert_dispatch/tensor_movement.rs");
include!("expert_dispatch/exchange_harness.rs");
include!("expert_dispatch/tests.rs");
