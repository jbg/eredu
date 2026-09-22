//! Physical-domain configuration shared by native application examples.

pub fn parse(value: &str) -> anyhow::Result<eredu_core::MemoryLimitDeclarations> {
    let entries = value
        .split(',')
        .map(|entry| {
            let (domain, value) = entry
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("expected DOMAIN=BYTES|unlimited"))?;
            anyhow::ensure!(!domain.is_empty(), "memory domain is empty");
            let limit = if value == "unlimited" {
                eredu_core::MemoryLimit::Unlimited
            } else {
                eredu_core::MemoryLimit::Finite(value.parse()?)
            };
            Ok((domain.to_owned(), limit))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let limits = eredu_core::MemoryLimitDeclarations::new(entries);
    eredu_backend_mlx::configure_memory_limits(&limits)?;
    Ok(limits)
}

/// Declares the same finite value independently for every reported domain.
/// Callers apply these request limits; this does not reconfigure the runtime.
pub fn finite_each(bytes: u64) -> anyhow::Result<eredu_core::MemoryLimitDeclarations> {
    let topology = eredu_backend_mlx::memory_topology()?;
    Ok(eredu_core::MemoryLimitDeclarations::new(
        topology
            .domains()
            .map(|(_, domain)| (domain.name.clone(), eredu_core::MemoryLimit::Finite(bytes))),
    ))
}
