/// Validated bidirectional mapping between checkpoint-global and owner-local ids.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertAssignment {
    global_expert_count: usize,
    group_size: usize,
    rank: usize,
    owners: Vec<usize>,
    owner_local: Vec<usize>,
    local_global: Vec<usize>,
}

impl ExpertAssignment {
    /// Lowers an architecture-derived realization without recomputing ownership.
    pub fn from_realization<S>(
        plan: &eredu_architectures::ExpertRealizationPlan<S>,
    ) -> Result<Self, Error> {
        Self::from_owners(
            plan.owners().to_vec(),
            plan.expert_parallel_size(),
            plan.expert_parallel_rank(),
        )
    }

    fn from_owners(owners: Vec<usize>, group_size: usize, rank: usize) -> Result<Self, Error> {
        validate_dimensions(owners.len(), group_size, rank)?;
        if let Some((expert, owner)) = owners
            .iter()
            .copied()
            .enumerate()
            .find(|(_, owner)| *owner >= group_size)
        {
            return Err(Error::Parallel(format!(
                "global expert {expert} has invalid owner rank {owner} for EP size {group_size}"
            )));
        }
        let mut next_local = vec![0usize; group_size];
        let mut owner_local = Vec::with_capacity(owners.len());
        let mut local_global = Vec::new();
        for (global, owner) in owners.iter().copied().enumerate() {
            owner_local.push(next_local[owner]);
            next_local[owner] = next_local[owner].checked_add(1).ok_or_else(|| {
                Error::Parallel("owner-local expert index overflowed usize".into())
            })?;
            if owner == rank {
                local_global.push(global);
            }
        }
        if next_local.contains(&0) {
            return Err(Error::Parallel(format!(
                "expert assignment creates an empty rank: counts {next_local:?}"
            )));
        }
        if owners.len() > i32::MAX as usize
            || next_local.iter().any(|count| *count > i32::MAX as usize)
        {
            return Err(Error::Parallel(
                "expert assignment exceeds MLX i32 indexing limits".into(),
            ));
        }
        Ok(Self {
            global_expert_count: owners.len(),
            group_size,
            rank,
            owners,
            owner_local,
            local_global,
        })
    }

    /// Total checkpoint-global routed expert count.
    pub const fn global_expert_count(&self) -> usize {
        self.global_expert_count
    }
    /// EP group size.
    pub const fn group_size(&self) -> usize {
        self.group_size
    }
    /// Current rank within the EP group.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Global expert ids owned by this rank, in owner-local order.
    pub fn local_global_group_indices(&self) -> &[usize] {
        &self.local_global
    }
    /// Number of experts owned by this rank.
    pub fn local_expert_count(&self) -> usize {
        self.local_global.len()
    }
    /// Returns the owner rank of a global expert.
    pub fn owner(&self, global: usize) -> Option<usize> {
        self.owners.get(global).copied()
    }
    /// Returns the owner-local id of a global expert.
    pub fn owner_local_id(&self, global: usize) -> Option<usize> {
        self.owner_local.get(global).copied()
    }
    /// Returns the global id corresponding to a local id on this rank.
    pub fn global_id(&self, local: usize) -> Option<usize> {
        self.local_global.get(local).copied()
    }
    /// Complete global-to-owner mapping.
    pub fn owners(&self) -> &[usize] {
        &self.owners
    }
    /// Complete global-to-owner-local mapping.
    pub fn owner_local_ids(&self) -> &[usize] {
        &self.owner_local
    }
}

fn validate_dimensions(global_experts: usize, group_size: usize, rank: usize) -> Result<(), Error> {
    if global_experts == 0 || group_size == 0 {
        return Err(Error::Parallel(
            "expert count and EP size must be nonzero".into(),
        ));
    }
    if rank >= group_size {
        return Err(Error::Parallel(format!(
            "EP rank {rank} is outside size {group_size}"
        )));
    }
    if global_experts < group_size {
        return Err(Error::Parallel(format!(
            "cannot assign {global_experts} experts to {group_size} non-empty ranks"
        )));
    }
    Ok(())
}
