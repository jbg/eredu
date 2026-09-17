fn select_preparation(
    preparation: moshi::RealtimePreparationPlan,
    config: &moshi::MoshiConfig,
    residency: ExecutionResidency,
) -> moshi::PreparedMoshiRealtime {
    let normalized = NormalizedLoadRequest::default()
        .with_weight_residency(residency::weight_policy(residency))
        .with_state_residency(CacheResidencyPolicy::Device)
        .with_communication_completion_policy(
            CommunicationCompletionPolicy::new(
                Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let observations = moshi::observation_points(config)
        .into_iter()
        .map(|p| RealtimeIdentity::new(p.path()).unwrap())
        .collect::<Vec<_>>();
    let request = moshi::moshi_realtime_request_from_normalized(
        &normalized,
        RealtimeObservationRequirements::new(true, observations.clone()),
    )
    .unwrap();
    let inspected = moshi::inspect_moshi_realtime(preparation, request).unwrap();
    let facts = eredu_runtime::synthesize_realtime_capabilities(inspected.requirements(), &Support)
        .with_observation_identities(observations);
    moshi::select_inspected_moshi_realtime(inspected, &facts).unwrap()
}
