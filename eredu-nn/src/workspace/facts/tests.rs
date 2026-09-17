use super::*;
use crate::workspace::{
    validate_workspace_output_storage, validate_workspace_tensor_declaration, WorkspaceDtype,
    WorkspaceEffectError, WorkspaceLayoutView,
};

#[test]
fn flat_aliases_preserve_duplicates_order_and_backward_output_identity() {
    let aliases = [99, 2, 0, 2, 99];
    let effect = WorkspaceOutputEffect::AllocateOrAliasInputs {
        bytes: 48,
        alias_start: 1,
        alias_count: 3,
    };
    let view = effect.as_view(&aliases).unwrap();
    assert_eq!(
        view,
        WorkspaceOutputStorageView::AllocateOrAliasInputs {
            bytes: 48,
            inputs: &[2, 0, 2],
        }
    );
    let WorkspaceOutputStorageView::AllocateOrAliasInputs { inputs, .. } = view else {
        unreachable!()
    };
    assert!(std::ptr::eq(inputs.as_ptr(), aliases[1..].as_ptr()));
    let layout = WorkspaceLayoutView::new(&[2, 3], WorkspaceDtype::Float32).unwrap();
    assert_eq!(
        validate_workspace_output_storage(view, layout, 0, 3),
        Ok(())
    );
    let previous = WorkspaceOutputEffect::AliasOutput(0).as_view(&[]).unwrap();
    assert_eq!(
        validate_workspace_output_storage(previous, layout, 1, 0),
        Ok(())
    );
    assert_eq!(
        validate_workspace_output_storage(previous, layout, 0, 0),
        Err(WorkspaceEffectError::OutputAlias)
    );
    for (alias_start, alias_count) in [(usize::MAX, 1), (3, 3), (6, 0)] {
        assert_eq!(
            WorkspaceOutputEffect::AllocateOrAliasInputs {
                bytes: 48,
                alias_start,
                alias_count,
            }
            .as_view(&aliases),
            None
        );
    }
    assert_eq!(
        WorkspaceOutputEffect::AllocateOrAliasInputs {
            bytes: 0,
            alias_start: aliases.len(),
            alias_count: 0,
        }
        .as_view(&aliases),
        Some(WorkspaceOutputStorageView::AllocateOrAliasInputs {
            bytes: 0,
            inputs: &[],
        })
    );
}

#[test]
fn exact_destination_validation_rejects_each_short_and_long_buffer_without_writes() {
    let expected = WorkspaceEffectLayout {
        outputs: 2,
        aliases: 3,
        assumption_bytes: 5,
    };
    for (kind, lengths) in [
        (WorkspaceFactDestinationKind::Outputs, [1, 3, 5]),
        (WorkspaceFactDestinationKind::Outputs, [3, 3, 5]),
        (WorkspaceFactDestinationKind::Aliases, [2, 2, 5]),
        (WorkspaceFactDestinationKind::Aliases, [2, 4, 5]),
        (WorkspaceFactDestinationKind::TensorAssumptions, [2, 3, 4]),
        (WorkspaceFactDestinationKind::TensorAssumptions, [2, 3, 6]),
    ] {
        let mut outputs = [WorkspaceOutputEffect::AliasOutput(123); 3];
        let mut aliases = [456; 4];
        let mut text = [0xA5; 6];
        let error = WorkspaceEffectDestination {
            outputs: &mut outputs[..lengths[0]],
            aliases: &mut aliases[..lengths[1]],
            assumptions: &mut text[..lengths[2]],
        }
        .validate(expected)
        .unwrap_err();
        assert_eq!(error.kind, kind);
        assert_eq!(outputs, [WorkspaceOutputEffect::AliasOutput(123); 3]);
        assert_eq!(aliases, [456; 4]);
        assert_eq!(text, [0xA5; 6]);
    }
    let facts = WorkspaceHostFacts {
        bytes: 97,
        assumption_bytes: 5,
    };
    for actual in [4, 6] {
        let mut text = [0xA5; 6];
        assert_eq!(
            WorkspaceHostDestination {
                assumptions: &mut text[..actual]
            }
            .validate(facts)
            .unwrap_err()
            .kind,
            WorkspaceFactDestinationKind::HostAssumptions
        );
        assert_eq!(text, [0xA5; 6]);
    }
    let mut outputs = [WorkspaceOutputEffect::Allocate(0); 2];
    let mut aliases = [0; 3];
    let mut text = [0; 5];
    assert_eq!(
        WorkspaceEffectDestination {
            outputs: &mut outputs,
            aliases: &mut aliases,
            assumptions: &mut text
        }
        .validate(expected),
        Ok(())
    );
    assert_eq!(
        WorkspaceHostDestination {
            assumptions: &mut text
        }
        .validate(facts),
        Ok(())
    );
}

#[test]
fn shared_effect_validation_preserves_header_byte_and_alias_error_precedence() {
    let layout = WorkspaceLayoutView::new(&[2, 3], WorkspaceDtype::Float32).unwrap();
    // Tensor text is deliberately nonempty-only; host text has the separate
    // Unicode-whitespace rule. Neither can silently inherit the other's rule.
    assert_eq!(
        validate_workspace_tensor_declaration(1, 1, "\u{2003}"),
        Ok(())
    );
    assert_eq!(
        crate::workspace::validate_workspace_host_assumptions("\u{2003}"),
        Err(WorkspaceEffectError::HostAssumptions)
    );
    assert_eq!(
        validate_workspace_tensor_declaration(0, 1, "valid"),
        Err(WorkspaceEffectError::Incomplete)
    );
    assert_eq!(
        validate_workspace_tensor_declaration(1, 1, ""),
        Err(WorkspaceEffectError::Incomplete)
    );
    for (bytes, expected) in [
        (23, WorkspaceEffectError::UndersizedAllocation),
        (24, WorkspaceEffectError::PossibleInputAlias),
    ] {
        assert_eq!(
            validate_workspace_output_storage(
                WorkspaceOutputStorageView::AllocateOrAliasInputs {
                    bytes,
                    inputs: &[1]
                },
                layout,
                0,
                1
            ),
            Err(expected)
        );
    }
    assert_eq!(
        validate_workspace_output_storage(WorkspaceOutputStorageView::AliasInput(0), layout, 0, 1),
        Ok(())
    );
    assert_eq!(
        validate_workspace_output_storage(WorkspaceOutputStorageView::AliasInput(1), layout, 0, 1),
        Err(WorkspaceEffectError::InputAlias)
    );
}
