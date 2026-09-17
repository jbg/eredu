//! Architecture-owned typed dispatch for a selected resident metadata mechanism.

use super::ReplicatedTextStateProfiles;
use eredu_nn::workspace::WorkspaceBackend;
use eredu_runtime::{
    working_memory::{WorkspaceResidentLayerState, WorkspaceResidentStateFactory},
    DeviceState,
};

impl ReplicatedTextStateProfiles<WorkspaceBackend> for WorkspaceResidentStateFactory {
    type StatelessState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
    type AttentionState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
    type ComponentState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
    type AttentionComponentState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
    type CompressedState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
    type CompressedComponentState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
}
