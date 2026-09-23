use eredu::api::{GenerationMemoryOptions, GenerationMemoryPlacement};
use eredu_core::{
    BackendId, CapabilityError, DevicePlan, HardwareBackendProfile, HardwareDeviceProfile,
    HardwareMemorySemantics, HardwareProfile, InputTokenCount, Observed,
};

fn hardware(semantics: HardwareMemorySemantics) -> HardwareProfile {
    HardwareProfile::observe_host(
        Observed::exact(1000, "installed host capacity"),
        Observed::exact(100, "host availability"),
        semantics,
        vec![HardwareBackendProfile {
            backend: BackendId::new("fixture").unwrap(),
            available: true,
            detail: None,
            devices: vec![device("first", 10), device("selected", 20)],
        }],
    )
}

fn device(id: &str, available: u64) -> HardwareDeviceProfile {
    HardwareDeviceProfile {
        id: id.into(),
        family: "opaque backend family".into(),
        index: 7,
        total_memory_bytes: Observed::exact(500, "installed device capacity"),
        available_memory_bytes: Observed::exact(available, "device availability"),
    }
}

fn options(
    hardware: &HardwareProfile,
    host_execution: bool,
) -> Result<GenerationMemoryOptions, CapabilityError> {
    GenerationMemoryOptions::for_hardware_device(
        InputTokenCount::text(12),
        hardware,
        &DevicePlan::new("fixture", "selected").unwrap(),
        host_execution,
    )
}

#[test]
fn host_execution_uses_host_capacity_regardless_of_accelerator_relationship() {
    for semantics in [
        HardwareMemorySemantics::Unified,
        HardwareMemorySemantics::SeparateTiers,
        HardwareMemorySemantics::Unknown,
    ] {
        let options = options(&hardware(semantics), true).unwrap();
        assert!(matches!(options.placement, GenerationMemoryPlacement::Host));
        assert_eq!(options.budget.available_bytes, Some(100));
        assert_eq!(options.host_budget.available_bytes, Some(100));
        assert_eq!(options.input, InputTokenCount::text(12));
    }
}

#[test]
fn unified_capacity_is_observed_once_and_application_policy_remains_unset() {
    let options = options(&hardware(HardwareMemorySemantics::Unified), false).unwrap();
    assert!(matches!(
        options.placement,
        GenerationMemoryPlacement::Unified
    ));
    assert_eq!(options.budget.available_bytes, Some(100));
    assert_eq!(options.budget.application_limit_bytes, None);
    assert_eq!(options.budget.reserve_bytes, 0);
    assert_eq!(options.host_budget.application_limit_bytes, None);
    assert_eq!(options.host_budget.reserve_bytes, 0);
    assert_eq!(options.already_resident_bytes, 0);
    assert_eq!(options.backend_overhead.upper_bytes, None);
}

#[test]
fn discrete_capacity_matches_exact_backend_and_device_and_keeps_host_separate() {
    let mut hardware = hardware(HardwareMemorySemantics::SeparateTiers);
    hardware.backends.insert(
        0,
        HardwareBackendProfile {
            backend: BackendId::new("another-backend").unwrap(),
            available: true,
            detail: None,
            devices: vec![device("selected", 999)],
        },
    );
    let mut options = options(&hardware, false).unwrap();
    assert!(
        matches!(&options.placement, GenerationMemoryPlacement::Separate { device } if device == "selected")
    );
    assert_eq!(options.budget.available_bytes, Some(20));
    assert_eq!(options.host_budget.available_bytes, Some(100));
    options.budget.application_limit_bytes = Some(15);
    options.budget.reserve_bytes = 3;
    assert_eq!(options.budget.available_bytes, Some(20));
    assert_eq!(options.host_budget.reserve_bytes, 0);
}

#[test]
fn zero_availability_is_preserved_as_a_successful_observation() {
    let mut hardware = hardware(HardwareMemorySemantics::SeparateTiers);
    hardware.available_memory_bytes = Observed::exact(0, "host exhausted");
    hardware.backends[0].devices[1].available_memory_bytes = Observed::exact(0, "device exhausted");
    let options = options(&hardware, false).unwrap();
    assert_eq!(options.budget.available_bytes, Some(0));
    assert_eq!(options.host_budget.available_bytes, Some(0));
}

#[test]
fn missing_availability_never_falls_back_to_installed_or_other_device_capacity() {
    let mut hardware = hardware(HardwareMemorySemantics::SeparateTiers);
    for missing in [
        Observed::unavailable("not observed"),
        Observed::unsupported("no counter"),
    ] {
        hardware.backends[0].devices[1].available_memory_bytes = missing;
        let options = options(&hardware, false).unwrap();
        assert_eq!(options.budget.available_bytes, None);
        assert_eq!(options.host_budget.available_bytes, Some(100));
    }
    hardware.physical_memory_semantics = HardwareMemorySemantics::Unified;
    hardware.available_memory_bytes = Observed::unavailable("host counter unavailable");
    hardware.backends[0].devices[1].available_memory_bytes =
        Observed::exact(20, "other observation");
    for host in [false, true] {
        let options = options(&hardware, host).unwrap();
        assert_eq!(options.budget.available_bytes, None);
        assert_eq!(options.host_budget.available_bytes, None);
    }
}

#[test]
fn unknown_relationship_does_not_combine_independent_capacity_observations() {
    let options = options(&hardware(HardwareMemorySemantics::Unknown), false).unwrap();
    assert!(matches!(
        options.placement,
        GenerationMemoryPlacement::Unknown
    ));
    assert_eq!(options.budget.available_bytes, None);
    assert_eq!(options.host_budget.available_bytes, Some(100));
}

#[test]
fn invalid_selection_is_rejected_even_on_host_or_unified_memory() {
    for host in [false, true] {
        let mut hardware = hardware(HardwareMemorySemantics::Unified);
        hardware.backends[0].available = false;
        assert!(matches!(
            options(&hardware, host),
            Err(CapabilityError::InvalidConfiguration {
                field: "hardware.backend",
                ..
            })
        ));
        hardware.backends[0].available = true;
        hardware.backends[0].devices.remove(1);
        assert!(matches!(
            options(&hardware, host),
            Err(CapabilityError::InvalidConfiguration {
                field: "hardware.device",
                ..
            })
        ));
        hardware.backends.clear();
        assert!(matches!(
            options(&hardware, host),
            Err(CapabilityError::InvalidConfiguration {
                field: "hardware.backend",
                ..
            })
        ));
    }
}
