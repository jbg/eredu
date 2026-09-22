use std::{fmt, str::FromStr};

use eredu_core::{MemoryLimit, MemoryLimitDeclarations};

/// A physical-domain declaration resolved against the selected backend topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CliMemoryLimit {
    pub(crate) domain: String,
    pub(crate) limit: MemoryLimit,
}

impl FromStr for CliMemoryLimit {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (domain, limit) = value
            .split_once('=')
            .ok_or_else(|| "expected <domain>=<bytes|unlimited>".to_owned())?;
        if domain.is_empty()
            || !domain
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err("domain must use letters, digits, '_' or '-'".into());
        }
        let limit = match limit {
            "unlimited" => MemoryLimit::Unlimited,
            value if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => {
                MemoryLimit::Finite(
                    value
                        .parse()
                        .map_err(|_| "byte limit exceeds u64".to_owned())?,
                )
            }
            _ => return Err("limit must be a byte count or 'unlimited'".into()),
        };
        Ok(Self {
            domain: domain.to_owned(),
            limit,
        })
    }
}

impl fmt::Display for CliMemoryLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}=", self.domain)?;
        match self.limit {
            MemoryLimit::Finite(bytes) => write!(f, "{bytes}"),
            MemoryLimit::Unlimited => f.write_str("unlimited"),
        }
    }
}

pub(crate) fn declarations(limits: &[CliMemoryLimit]) -> MemoryLimitDeclarations {
    MemoryLimitDeclarations::new(
        limits
            .iter()
            .map(|entry| (entry.domain.clone(), entry.limit)),
    )
}

/// Writes one coherent ledger observation using backend-reported domain names.
pub(crate) fn report(output: &mut impl std::io::Write) -> anyhow::Result<()> {
    let topology = eredu::api::local_memory_topology()?;
    let snapshot = eredu::api::local_memory_snapshot()?;
    for (row, (_, description)) in snapshot.domains.iter().zip(topology.domains()) {
        let limit = |value| match value {
            MemoryLimit::Finite(bytes) => bytes.to_string(),
            MemoryLimit::Unlimited => "unlimited".to_owned(),
        };
        writeln!(output,
            "memory_domain: {} configured={} effective={} current_charge={} historical_peak={} fixed_baseline={} registered_storage={} outstanding_reservations={} estimated_placement_allowance={} estimated_overhead={} additional_headroom={}",
            description.name, limit(row.configured_limit), limit(row.effective_limit),
            row.current_charge_bytes, row.historical_peak_bytes, row.fixed_baseline.total()?,
            row.registered_storage_bytes, row.outstanding_reservation_bytes,
            row.estimated_placement_allowance_bytes, row.estimated_overhead_bytes,
            row.additional_headroom_bytes,
        )?;
        if let Some(basis) = row.placement_allowance_basis {
            writeln!(output, "memory_placement_basis: {} {basis:?} (conservative allowance, not measured residency)", description.name)?;
        }
    }
    writeln!(
        output,
        "memory_coverage: reservations={} funding_accounts={} unquoted_owners={}",
        snapshot.reservations, snapshot.funding_accounts, snapshot.unquoted_owners
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_zero_maximum_and_explicit_unlimited_without_conflating_them() {
        for (text, limit) in [
            ("host=0", MemoryLimit::Finite(0)),
            ("gpu-1=18446744073709551615", MemoryLimit::Finite(u64::MAX)),
            ("shared=unlimited", MemoryLimit::Unlimited),
        ] {
            let parsed: CliMemoryLimit = text.parse().unwrap();
            assert_eq!(parsed.limit, limit);
            assert_eq!(parsed.to_string(), text);
        }
    }

    #[test]
    fn rejects_malformed_or_overflowing_limits() {
        for text in [
            "host",
            "=0",
            "host=-1",
            "host=",
            "host=+1",
            "host=infinity",
            "host=18446744073709551616",
            "gpu:0=4",
        ] {
            assert!(text.parse::<CliMemoryLimit>().is_err(), "{text}");
        }
    }
}
