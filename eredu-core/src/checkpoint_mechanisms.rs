//! Native materialization and transfer contracts for prepared checkpoint weights.

use crate::{BackendFailure, Completion, SubmissionBackend};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointLease, CheckpointSource},
};
use eredu_nn::{NeuralBackend, Tensor};

/// Native tile submission and readback for bounded checkpoint conversion.
pub mod bounded_quantization;

/// Native materialization for retained prediction.
pub mod prediction;
/// Shared metadata preflight and bounded execution of checkpoint recipes.
pub mod recipe_execution;
/// Native storage and submission for prepared parameter units.
pub mod unit_binding;

/// Materializes and binds checkpoint data to backend-native parameter slots.
pub trait CheckpointParameterBackend: NeuralBackend {
    /// One backend-native parameter slot.
    type Parameter: 'static;
    /// Materialized backend-native checkpoint weight.
    type MaterializedWeight;
    /// Backend context used only while realizing checkpoint parameters.
    type MaterializationContext: ?Sized;
    /// In-flight guard retaining encoded sources through exact realization completion.
    type Materialization;
    /// Backend-specific loading failure.
    type ParameterError: std::error::Error + Send + Sync + 'static;

    /// Validates that one neutral recipe can be represented by this backend.
    ///
    /// This is a metadata-only selection gate. Implementations must not acquire
    /// payload leases, allocate native tensors, or submit backend work.
    fn preflight_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<(), Self::ParameterError>;

    /// Lowers one format-preserving encoded lease into a native weight.
    fn materialize(
        lease: CheckpointLease,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError>;

    /// Lowers a validated neutral recipe directly into a native weight.
    fn materialize_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError>;

    /// Lowers an optional physical storage preference while preserving the recipe's
    /// logical geometry and values. Backends may retain ordinary canonical storage.
    fn materialize_recipe_with_layout(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        layout: eredu_nn::ParameterStorageLayout,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError> {
        let _ = layout;
        Self::materialize_recipe(recipe, source, context)
    }

    /// Borrows the native weight retained by an in-flight materialization.
    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight;

    /// Waits for this exact realization and releases its encoded source lease.
    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError>;

    /// Creates another native handle to identical materialized storage without
    /// rereading or rematerializing checkpoint data.
    fn share_materialized_weight(
        weight: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError>;

    /// Validates destination shape/storage compatibility without publication.
    fn validate_bind(
        parameter: &Self::Parameter,
        weight: &Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError>;

    /// Binds one materialized weight to its destination parameter.
    ///
    /// This operation is infallible so orchestration can validate an entire
    /// atomic unit before publishing any destination.
    fn bind(parameter: &mut Self::Parameter, weight: Self::MaterializedWeight);
}

/// Backend-described geometry of one statically traversed parameter slot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterBindingTarget {
    /// Logical shape expected by the destination.
    pub shape: Vec<usize>,
    /// Logical dtype expected by the destination.
    pub dtype: eredu_checkpoint::recipe::RecipeDtype,
    /// Source dtypes the backend can bind to this slot in addition to `dtype`.
    ///
    /// Admission preserves the recipe's dtype and byte accounting; it does not
    /// request conversion to the unloaded slot's initial dtype.
    pub permitted_source_dtypes: Vec<eredu_checkpoint::recipe::RecipeDtype>,
}

/// Backend facts and encoding operations used by exact portable binding.
pub trait ExactParameterMaterializer<B: NeuralBackend> {
    /// Describes the native parameter destination without acquiring payloads.
    fn parameter_target(parameter: &B::Tensor) -> Option<ParameterBindingTarget>;

    /// Lowers an admitted derived F4 checkpoint recipe for an MXFP4 executable.
    fn lower_mxfp4(
        recipe: DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<DerivedWeightRecipe, BackendFailure>;

    /// Opens native quantization mechanisms on the selected execution context.
    fn quantizer(
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Box<dyn bounded_quantization::BoundedQuantizationBackend + '_>, BackendFailure>;
}

/// Native encoding and bounded conversion of exact prepared checkpoint weights.
pub trait CheckpointEncodingMaterializer {
    /// Side-effect-free checkpoint parameter support for the native backend.
    type Parameters: CheckpointParameterBackend;

    /// Lowers an intrinsic packed recipe to the backend's native encoding.
    fn lower_mxfp4(
        &self,
        recipe: DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<DerivedWeightRecipe, crate::BackendFailure>;

    /// Opens native quantization mechanisms on the weights context.
    fn weights_quantizer(
        &self,
    ) -> Result<Box<dyn bounded_quantization::BoundedQuantizationBackend + '_>, BackendFailure>;
}

/// Promotes and demotes backend-native storage without changing its semantics.
pub trait ParameterTransferBackend: SubmissionBackend + CheckpointParameterBackend {
    /// Backend-owned host representation.
    type HostBuffer;
    /// In-flight transfer guard retaining all source and destination storage.
    type Transfer: Completion<Error = Self::TransferError>;
    /// Backend-specific transfer failure.
    type TransferError: std::error::Error + Send + Sync + 'static;

    /// Promotes host storage into a materialized execution weight.
    fn promote(
        executor: &Self::Executor,
        host: &Self::HostBuffer,
    ) -> Result<(Self::MaterializedWeight, Self::Transfer), Self::TransferError>;

    /// Demotes a materialized execution weight into backend-owned host storage.
    fn demote(
        executor: &Self::Executor,
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Self::Transfer), Self::TransferError>;
}

/// Native placement and physical facts used while binding prepared model weights.
pub trait ModelBindingResources {
    /// Exact native placement request supplied by loading.
    type Placement;
    /// The native setup transport used to agree on a rank manifest.
    type CommunicationSetup<'a>: crate::communication::CommunicationSetupTransport<
        Error = Self::BindingError,
    >
    where
        Self: 'a;
    /// Realized rank-local communication retained with the target.
    type Communication: Clone;
    /// Exact native values and queues used by text execution.
    type TextValues: crate::backend::NativeTextValues;
    /// Native placement retained by the installed model.
    type Target;
    /// Native binding failure.
    type BindingError;

    /// Reports whether the placement includes a native rank.
    fn has_native_rank(placement: &Self::Placement) -> bool;

    /// Opens the exact native setup transport for the given placement.
    fn open_communication_setup(
        &self,
        placement: &Self::Placement,
    ) -> Result<Self::CommunicationSetup<'_>, Self::BindingError>;

    /// Realizes the native rank resources from an already agreed world.
    fn realize_model_communication<'a>(
        &'a self,
        setup: crate::communication::WorldCommunicationBinding<Self::CommunicationSetup<'a>>,
    ) -> Result<Self::Communication, Self::BindingError>;

    /// Reports the manifest carried by the realized native resources.
    fn bound_communication_manifest(
        communication: &Self::Communication,
    ) -> &crate::communication::CommunicationManifest;

    /// Releases completed native work before binding its replacement.
    fn reclaim_retired_bindings(&self);

    /// Supplies optional observation and initialization of native allocator defaults.
    fn allocator_cache_defaults(
        &self,
    ) -> Option<&dyn crate::capability::NativeAllocatorCacheDefaults> {
        None
    }

    /// Opens a target for the exact selected placement and any realized communication.
    fn open_binding_target(
        &self,
        placement: &Self::Placement,
        communication: Option<&Self::Communication>,
    ) -> Result<Self::Target, Self::BindingError>;

    /// Reports the physical floating representation produced by the selected source encoding.
    fn floating_state_element(
        &self,
        source: &crate::checkpoint::TensorDtype,
    ) -> Option<eredu_nn::TensorElementType>;

    /// Reports the native observation hooks.
    fn observation_mechanisms(&self) -> crate::ObservationMechanisms;

    /// Reports native capture transforms and limits.
    fn capture_capabilities(&self) -> crate::capture::CaptureCapabilities;

    /// Reclaims temporary materialization allocations before returning the model.
    fn finish_binding(&self) -> Result<(), Self::BindingError>;
}

/// Native text resources without rank communication.
pub struct LocalTextPlacement;

/// Native text resources capable of binding a rank.
pub struct RankedTextPlacement;

/// Exact native contexts and resource factories used to install text routes.
pub trait TextBindingResources: ModelBindingResources {
    /// Native neural operators.
    type Neural: eredu_nn::NeuralBackend;
    /// Native tensor and page storage used by portable state owners.
    type CacheStorage;
    /// Native predictor materialization hooks.
    type PredictionMaterializer;
    /// Rank communication compiled into the native binding.
    type RankPlacement;
}

/// Borrows the native inputs on which portable text route assembly operates.
pub trait TextBindingContext<'a>: TextBindingResources {
    /// Native binding resources borrowed for an installation.
    type TextResources: Clone + 'a;

    /// Borrows the exact execution context.
    fn text_execution(
        &'a self,
    ) -> &'a <<Self::Neural as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context;
    /// Borrows the checkpoint materialization context.
    fn text_weights(
        &'a self,
    ) -> &'a <<Self::Neural as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context;
    /// Borrows the local storage and native resource factories.
    fn text_resources(&'a self) -> Self::TextResources;
}

/// Exact queue and storage capabilities retained by the portable text publisher.
pub trait TextPublicationResources {
    /// Native tensor and queue representations.
    type Values: crate::backend::NativeTextValues;
    /// Native storage operations for any retained parameter bank.
    type BankStorage: crate::residency::observations::DenseResidencyMechanisms
        + crate::residency::NativeBankMechanisms<
            Backend: eredu_nn::GroupedNeuralBackend<
                Tensor = <Self::Values as crate::backend::NativeTextValues>::Tensor,
            >,
            BankError = <Self::Values as crate::backend::NativeTextValues>::Error,
        >;

    /// Borrows the exact native queue used for publication.
    fn publication_context(&self) -> &<Self::Values as crate::backend::NativeTextValues>::Context;
}

/// Native parameter and queue inputs for a text route that uses components.
pub trait TextComponentResources {
    /// Native tensor and queue representation.
    type Values: crate::backend::NativeTextValues;
    /// Native resource factory for each admitted component.
    type Components;

    /// Borrows the execution queue retained by the portable executable.
    fn component_execution(&self) -> &<Self::Values as crate::backend::NativeTextValues>::Context;
    /// Borrows the source materialization queue.
    fn component_weights(&self) -> &<Self::Values as crate::backend::NativeTextValues>::Context;
    /// Supplies the physical parameter and queue operators for binding components.
    fn text_components(&self) -> Self::Components;
}

/// Opens an already selected predictor source on the exact native queue.
pub trait NativePredictionRead: crate::backend::NativeTextValues {
    /// Native binding inputs for the retained source.
    type Reader<'a>;
    /// Opens a reader over the selected source and context.
    fn prediction_reader(
        source: eredu_checkpoint::store::SharedCheckpointSource,
        queue: &Self::Context,
    ) -> Self::Reader<'_>;
}

/// Native allocation and indexed movement for a selected rank.
pub trait TextRankResources<B>: TextComponentResources
where
    B: crate::communication::CommunicationBackend,
{
    /// Allocator for exact pipeline boundary tensor shapes.
    type Allocator: crate::communication::PartitionTensorAllocator<B> + 'static;
    /// Native movement of rows for an admitted expert wave.
    type ExpertMovement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor> + 'static;

    /// Supplies the rank's boundary tensor allocator.
    fn rank_allocator(&self) -> Self::Allocator;
    /// Supplies the rank's indexed tensor movement operations.
    fn rank_expert_movement(&self) -> Self::ExpertMovement;
}

/// Native parameter, output tensor, and completion operations for a text route.
pub trait NativeTextWeights: crate::backend::NativeTextValues {
    /// Parameter tensor materialization and installation.
    type Parameters: ExactParameterMaterializer<Self::Neural>;

    /// Slices one completed score row from the supplied tensor.
    fn text_output_row(
        output: Self::Tensor,
        index: i32,
        queue: &Self::Context,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Settles the output and the supplied state roots on the selected queue.
    fn complete_text_output(
        output: &Self::Tensor,
        state_roots: &[&Self::Tensor],
        queue: &Self::Context,
    ) -> Result<(), Self::Error>;

    /// Reads physical memory facts without submitting work.
    fn text_parameter_memory(
        queue: &Self::Context,
    ) -> Result<Option<crate::capability::ParameterMemoryFacts>, Self::Error>;
}

/// Native read, conversion, and completion operations for admitted parameter bindings.
///
/// Abandoned submissions must retain any in-flight inputs and outputs until the
/// native implementation establishes safe completion or teardown.
pub trait ParameterReadScope {
    /// Native description of one byte-preserving destination.
    type DirectRead;
    /// Native parameter owner.
    type Value;
    /// Resources retained from submission through completion.
    type Pending;
    /// Source-read or native operation failure.
    type Error: std::error::Error
        + Send
        + Sync
        + From<crate::BackendFailure>
        + From<eredu_checkpoint::store::StoreError>
        + 'static;

    /// Number of slots available for staging ordinary materialization batches.
    fn slots(&self) -> std::num::NonZeroUsize;

    /// Checks whether an exact binding can be initialized directly.
    fn direct_read(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        binding: &crate::residency::parameter_binding::WeightBinding,
    ) -> Result<Option<Self::DirectRead>, Self::Error>;

    /// Submits byte-preserving reads into the requested layouts, in order.
    fn submit_direct(
        &mut self,
        reads: Vec<(Self::DirectRead, eredu_nn::ParameterStorageLayout)>,
    ) -> Result<Self::Pending, Self::Error>;

    /// Submits conversion or transfer for an exact source lease or derived recipe.
    fn submit_ordinary(
        &mut self,
        name: &str,
        input: ParameterReadInput<'_>,
    ) -> Result<Self::Pending, Self::Error>;

    /// Takes the values of one submission in the requested order. The native
    /// scope may retain resources for a separately settled aggregate transfer.
    fn take_values(&mut self, pending: Self::Pending) -> Result<Vec<Self::Value>, Self::Error>;

    /// Identifies bounded checkpoint capacity exhaustion inside a native conversion.
    fn checkpoint_capacity_exhausted(error: &Self::Error) -> bool;

    /// Releases source leases held by an aggregate submission, if any.
    fn release_checkpoint_capacity(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

/// The exact input of an ordinary parameter conversion or transfer.
pub enum ParameterReadInput<'a> {
    /// Prepared checkpoint bytes for a directly bound source tensor.
    Lease(eredu_checkpoint::store::CheckpointLease),
    /// A derived recipe and the restricted source supplying its inputs.
    Recipe(
        &'a eredu_checkpoint::recipe::DerivedWeightRecipe,
        &'a dyn eredu_checkpoint::store::CheckpointSource,
    ),
}
