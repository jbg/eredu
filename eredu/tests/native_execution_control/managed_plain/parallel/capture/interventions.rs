//! The existing loaded intervention driver under actual Ring ownership cuts.
use super::*;

fn source<const EP: usize>() -> Fixture {
    if EP > 1 {
        crate::routed_components::routed_fixture()
    } else {
        fixture(false)
    }
}
fn loaded<const TP: usize, const PP: usize, const EP: usize>(
    mode: &str,
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
    partitioned: bool,
) -> serde_json::Value {
    // The TP case edits actual component columns; the routed family case edits
    // the complete layer output while still traversing the real EP providers.
    let point = if EP > 1 {
        "model.layers.0.output"
    } else {
        "model.layers.0.feed_forward.units"
    };
    super::super::super::capture::run_intervention_loaded(
        mode,
        model,
        root,
        partitioned,
        point,
        TP * PP * EP,
        // The reached TP/EP chunk2 quote is 15,707,141,187 bytes before
        // existing loaded/source reservations (16GiB left 15,345,481,357).
        // Keep the positive test at its original two-position chunk; the
        // recorded smaller-capacity refusal remains separate evidence.
        if TP>1 && EP>1 {24u64<<30}else{16u64<<30},
    )
}
fn ordinary<const TP: usize, const PP: usize, const EP: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    loaded::<TP, PP, EP>("ordinary", model, root, true)
}
fn managed<const TP: usize, const PP: usize, const EP: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    loaded::<TP, PP, EP>("managed", model, root, true)
}
fn controlled<const TP: usize, const PP: usize, const EP: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    loaded::<TP, PP, EP>("controlled", model, root, true)
}
fn run<const TP: usize, const PP: usize, const EP: usize>(mode: &str) -> serde_json::Value {
    if mode == "serial" {
        let root = managed_fixture(source::<EP>());
        let execution =
            ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
        let (model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap_or_else(report_failure)
                .into_parts();
        return loaded::<TP, PP, EP>("ordinary", model, root, false);
    }
    let worker = match mode {
        "ordinary" => ordinary::<TP, PP, EP>,
        "managed" => managed::<TP, PP, EP>,
        "controlled" => controlled::<TP, PP, EP>,
        _ => panic!("intervention mode"),
    };
    run_partitioned_with_fixture(
        mode,
        eredu_core::ParallelTopology::new(TP, PP, EP, 1).unwrap(),
        Some(worker),
        source::<EP>,
    )
}
fn compare(actual: &serde_json::Value, expected: &serde_json::Value, mode: &str, rank: usize) {
    assert_eq!(
        actual["outcomes"], expected["outcomes"],
        "{mode} rank{rank} original operation outcomes"
    );
    super::compare(actual, expected, mode, rank);
}
macro_rules! case {
    ($name:ident,$tp:literal,$pp:literal,$ep:literal,$label:literal) => {
        #[test]
        #[ignore = "requires Metal and local Ring processes"]
        fn $name() {
            compare_selected_modes_by(
                concat!(
                    "managed_plain::parallel::capture::interventions::",
                    stringify!($name)
                ),
                "EREDU_PUBLIC_PARTITION_INTERVENTION_MODE",
                "PUBLIC_PARTITION_INTERVENTION_RESULT:",
                $label,
                $tp * $pp * $ep,
                &["ordinary", "managed", "controlled"],
                run::<$tp, $pp, $ep>,
                compare,
            );
        }
    };
}
case!(
    native_tp_interventions_preserve_component_windows_and_publication_owner,
    2,
    1,
    1,
    "TP interventions"
);
case!(
    native_pp_interventions_preserve_absent_members_and_publication_owner,
    1,
    2,
    1,
    "PP interventions"
);
case!(
    native_combined_interventions_preserve_component_windows_and_publication_owner,
    2,
    2,
    1,
    "TP/PP interventions"
);
case!(
    native_ep_interventions_preserve_routed_traversal_and_publication_owner,
    1,
    1,
    2,
    "EP interventions"
);
case!(
    native_tp_ep_interventions_preserve_routed_traversal_and_publication_owner,
    2,
    1,
    2,
    "TP/EP interventions"
);
case!(
    native_pp_ep_interventions_preserve_routed_traversal_and_publication_owner,
    1,
    2,
    2,
    "PP/EP interventions"
);
case!(
    native_combined_ep_interventions_preserve_routed_traversal_and_publication_owner,
    2,
    2,
    2,
    "TP/PP/EP interventions"
);

#[path = "interventions/evidence.rs"]
mod evidence;
