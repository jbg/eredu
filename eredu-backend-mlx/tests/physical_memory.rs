//! The process coordinator owns ordinary native backing through cache reuse.
use eredu_backend_mlx::{configure_memory_limits, memory_snapshot, memory_topology};
use eredu_core::{MemoryDomainError, MemoryLimit, MemoryLimitDeclarations};
use eredu_nn::Tensor;
use safemlx::Array;

fn settle() {
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
}

#[test]
fn process_ledger_tracks_native_backing_publication_cache_and_refusal() {
    let topology = memory_topology().unwrap();
    let zero = MemoryLimitDeclarations::new(
        topology
            .domains()
            .map(|(_, description)| (description.name.clone(), MemoryLimit::Finite(0))),
    );
    let rejected = configure_memory_limits(&zero).unwrap_err();
    let mut cause: &dyn std::error::Error = &rejected;
    let baseline_required = loop {
        if let Some(eredu_runtime::working_memory::WorkingMemoryError::Domain(
            MemoryDomainError::BudgetExceeded {
                domain,
                limit_bytes,
                existing_bytes,
                requested_bytes,
            },
        )) = cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>()
        {
            assert_eq!(*domain, topology.host_domain());
            assert_eq!((*limit_bytes, *existing_bytes), (0, 0));
            break *requested_bytes;
        }
        cause = cause
            .source()
            .expect("zero limits reject the explicitly quoted process baseline");
    };
    let payload_budget = 16u64 << 20;
    let limit = baseline_required.checked_add(payload_budget).unwrap();
    configure_memory_limits(&MemoryLimitDeclarations::new(
        topology
            .domains()
            .map(|(_, description)| (description.name.clone(), MemoryLimit::Finite(limit))),
    ))
    .unwrap();
    settle();
    let prior_cache_limit = safemlx::memory::set_cache_limit(1 << 20).unwrap();
    let baseline = memory_snapshot().unwrap();
    let source = Array::try_from_slice(&[2.0f32, -3.0, 5.0, 7.0], &[4]).unwrap();
    let facts = source.allocation_info().unwrap().unwrap();
    let live = memory_snapshot().unwrap();
    assert!(live
        .domains
        .iter()
        .zip(&baseline.domains)
        .any(|(after, before)| after.registered_storage_bytes > before.registered_storage_bytes));
    assert!(facts.host_control_bytes() > 0);
    assert!(live
        .domains
        .iter()
        .zip(&baseline.domains)
        .all(|(after, before)| after.outstanding_reservation_bytes
            == before.outstanding_reservation_bytes));
    drop(source);
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        memory_snapshot().unwrap(),
        live,
        "the native cache owns the same charge"
    );
    let reused = Array::try_from_slice(&[11.0f32, -13.0, 17.0, 19.0], &[4]).unwrap();
    assert_eq!(reused.allocation_info().unwrap(), Some(facts));
    assert_eq!(
        memory_snapshot().unwrap(),
        live,
        "reuse creates no second backing charge"
    );
    assert_eq!(
        reused.evaluated().unwrap().as_slice::<f32>(),
        &[11.0, -13.0, 17.0, 19.0]
    );
    drop(reused);
    settle();
    let freed = memory_snapshot().unwrap();
    for (after, before) in freed.domains.iter().zip(&baseline.domains) {
        assert_eq!(after.current_charge_bytes, before.current_charge_bytes);
    }
    // Transfer controls are admitted before their shared owner and backing are
    // constructed. Array aliases expose the same identity to the coordinator.
    let mut transfer = safemlx::HostTransferBuffer::new(
        &[4],
        safemlx::Dtype::Float32,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap();
    for (slot, value) in transfer
        .as_bytes_mut()
        .unwrap()
        .chunks_exact_mut(4)
        .zip([2.0f32, -3.0, 5.0, 7.0])
    {
        slot.copy_from_slice(&value.to_ne_bytes());
    }
    let transfer = transfer.freeze();
    let transfer_facts = transfer.allocation_info().unwrap();
    let transfer_live = memory_snapshot().unwrap();
    assert!(transfer_facts.host_control_bytes() > 0);
    assert!(transfer_live
        .domains
        .iter()
        .zip(&baseline.domains)
        .any(|(after, before)| after.registered_storage_bytes
            >= before.registered_storage_bytes + transfer_facts.bytes() as u64));
    let devices = safemlx::physical_memory_topology().unwrap();
    let device = devices.first().map_or(
        safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
        |device| safemlx::Device::new(safemlx::DeviceType::Gpu, device.ordinal as i32),
    );
    let transfer_stream = safemlx::Stream::new_with_device(&device);
    let destination = transfer
        .copy_to_array(&transfer_stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let _ = destination.evaluated().unwrap();
    let destination_facts = destination.allocation_info().unwrap().unwrap();
    if destination_facts.identity() == transfer_facts.identity() {
        assert_eq!(destination_facts, transfer_facts);
        assert_eq!(
            memory_snapshot().unwrap(),
            transfer_live,
            "a zero-copy Array alias adds no physical charge"
        );
    }
    drop(transfer);
    safemlx::reclaim_allocation_owners();
    let mut output = [0.0f32; 4];
    destination
        .evaluated()
        .unwrap()
        .try_copy_into(&mut output)
        .unwrap();
    assert_eq!(output, [2.0, -3.0, 5.0, 7.0]);
    drop(destination);
    transfer_stream.synchronize().unwrap();
    settle();
    for (after, before) in memory_snapshot()
        .unwrap()
        .domains
        .iter()
        .zip(&baseline.domains)
    {
        assert_eq!(after.current_charge_bytes, before.current_charge_bytes);
    }
    // Ordinary exports preserve their own host charge after the native source
    // retires. Truncation and iteration cannot detach backing from that charge.
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let tensor = eredu_backend_mlx::MlxTensor::from_array(
        Array::try_from_slice(&[3.5f32, -7.0, 11.25], &[3]).unwrap(),
    );
    let mut exported = tensor.to_f32_vec(&stream).unwrap();
    assert_eq!(exported.as_slice(), &[3.5, -7.0, 11.25]);
    drop(tensor);
    settle();
    let host = topology.host_domain();
    let baseline_host = baseline
        .domains
        .iter()
        .find(|entry| entry.domain == host)
        .unwrap()
        .current_charge_bytes;
    let retained = memory_snapshot()
        .unwrap()
        .domains
        .into_iter()
        .find(|entry| entry.domain == host)
        .unwrap()
        .current_charge_bytes;
    assert!(retained > baseline_host + 12);
    exported.truncate(1);
    assert_eq!(exported.capacity(), 3);
    let mut iter = exported.into_iter();
    assert_eq!(iter.next(), Some(3.5));
    assert_eq!(iter.next(), None);
    assert_eq!(
        memory_snapshot()
            .unwrap()
            .domains
            .into_iter()
            .find(|entry| entry.domain == host)
            .unwrap()
            .current_charge_bytes,
        retained
    );
    drop(iter);
    assert_eq!(
        memory_snapshot()
            .unwrap()
            .domains
            .into_iter()
            .find(|entry| entry.domain == host)
            .unwrap()
            .current_charge_bytes,
        baseline_host
    );

    let converted_source = eredu_backend_mlx::MlxTensor::from_array(
        Array::try_from_slice(
            &[half::bf16::from_f32(1.5), half::bf16::from_f32(-3.25)],
            &[2],
        )
        .unwrap(),
    );
    let converted_facts = converted_source.as_array().allocation_info().unwrap();
    let native_before = safemlx::memory::active_memory().unwrap();
    let converted = converted_source.to_f32_vec(&stream).unwrap();
    assert_eq!(converted.as_slice(), &[1.5, -3.25]);
    assert_eq!(
        safemlx::memory::active_memory().unwrap(),
        native_before,
        "dtype conversion creates no native array or staging backing"
    );
    assert_eq!(
        converted_source.as_array().allocation_info().unwrap(),
        converted_facts
    );
    drop(converted);
    drop(converted_source);
    settle();

    let source_values = vec![7.25f32; usize::try_from(payload_budget / 8).unwrap()];
    let source = eredu_backend_mlx::MlxTensor::from_array(
        Array::try_from_slice(&source_values, &[source_values.len() as i32]).unwrap(),
    );
    let source_facts = source.as_array().allocation_info().unwrap();
    let before_refusal = memory_snapshot().unwrap();
    let error = source.to_f32_vec(&stream).unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    let mut refused = false;
    loop {
        refused |= matches!(
            cause.downcast_ref::<MemoryDomainError>(),
            Some(MemoryDomainError::BudgetExceeded { .. })
        );
        refused |= matches!(
            cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>(),
            Some(eredu_runtime::working_memory::WorkingMemoryError::Domain(
                MemoryDomainError::BudgetExceeded { .. }
            ))
        );
        match cause.source() {
            Some(next) => cause = next,
            None => break,
        }
    }
    assert!(
        refused,
        "a retained source and a new host output overlap under one physical limit: {error:?}"
    );
    assert_eq!(source.as_array().allocation_info().unwrap(), source_facts);
    for (after, before) in memory_snapshot()
        .unwrap()
        .domains
        .iter()
        .zip(&before_refusal.domains)
    {
        assert_eq!(after.current_charge_bytes, before.current_charge_bytes);
    }
    drop(source);
    settle();

    let input = vec![23u8; usize::try_from(payload_budget).unwrap()];
    let error = Array::try_from_slice(&input, &[input.len() as i32]).unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    let mut domain_refusal = false;
    loop {
        let domain = cause.downcast_ref::<MemoryDomainError>().or_else(|| {
            match cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>() {
                Some(eredu_runtime::working_memory::WorkingMemoryError::Domain(domain)) => {
                    Some(domain)
                }
                _ => None,
            }
        });
        if let Some(MemoryDomainError::BudgetExceeded { limit_bytes, .. }) = domain {
            assert_eq!(*limit_bytes, limit);
            domain_refusal = true;
        }
        match cause.source() {
            Some(next) => cause = next,
            None => break,
        }
    }
    assert!(
        domain_refusal,
        "native refusal retains the neutral physical-domain cause: {error:?}"
    );
    settle();
    for (after, before) in memory_snapshot()
        .unwrap()
        .domains
        .iter()
        .zip(&baseline.domains)
    {
        assert_eq!(after.current_charge_bytes, before.current_charge_bytes);
    }
    safemlx::memory::set_cache_limit(prior_cache_limit).unwrap();
}
