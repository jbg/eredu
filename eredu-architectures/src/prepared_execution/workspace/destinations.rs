//! Cold destination geometry from the ordinary selected module constructors.

use super::*;
use crate::{
    composite_execution::CompositeArchitecture,
    replicated_text::{
        CompositeTextArchitectureVisitor, PreparedCompositeTextArchitecture,
        PreparedReplicatedTextModules, PreparedRoutedCompositeTextArchitecture,
    },
};
use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized};
use std::collections::BTreeMap;
mod partitioned;
pub use partitioned::{PartitionedTextBindingDestinations, project_partitioned_text_binding_destinations};
mod prediction;
mod routed;
pub use prediction::PredictionBindingDestination;
use prediction::ProjectionMaterializer;

/// Physical parameter slots produced by the selected replicated architecture.
/// This owns only metadata. In particular, floating metadata denotes an unloaded
/// slot, not a request to cast its eventual checkpoint payload to float32.
pub struct ReplicatedTextBindingDestinations {
    contract: eredu_runtime::PreparedReplicatedTextContract,
    static_parameters: BTreeMap<String, WorkspaceLayout>,
    units: Vec<BTreeMap<String, WorkspaceLayout>>,
    prediction: Vec<PredictionBindingDestination>,
}

impl ReplicatedTextBindingDestinations {
    /// Exact selected tasks, execution layout and source identities.
    pub fn contract(&self) -> &eredu_runtime::PreparedReplicatedTextContract {
        &self.contract
    }

    /// Slots owned by the architecture's static modules.
    pub fn static_parameters(&self) -> &BTreeMap<String, WorkspaceLayout> {
        &self.static_parameters
    }

    /// Exact physical prediction module destinations and selected task rows.
    pub fn prediction_modules(&self) -> &[PredictionBindingDestination] {
        &self.prediction
    }

    /// Slots in canonical execution-unit ordinal order.
    pub fn units(&self) -> &[BTreeMap<String, WorkspaceLayout>] {
        &self.units
    }
}

/// Projects exact selected replicated or composite module destinations before native loading.
///
/// This runs the ordinary architecture dispatch and constructors with metadata
/// tensors. Packed shapes and companion slots therefore come from their real
/// module definitions, not from logical task shapes. It never opens a payload,
/// chooses another source or creates a native handle. The backend still validates
/// its actual destinations against this projection before binding, and supplies
/// its own scalar-representation compatibility and lowering mechanisms.
///
/// This allocating cold plan is not an admission grant. Transform source reads,
/// constructor storage, native materialization and all retained aliases need
/// their own source admission. Other execution classes keep the total driver's
/// typed rejection until their corresponding projection route is supplied.
pub fn project_replicated_text_binding_destinations(
    sources: &PreparedModelSources,
    context: &WorkspaceContext,
) -> Result<ReplicatedTextBindingDestinations, PreparedExecutionError<Error>> {
    let routes = PreparedExecutionRoutes::new()
        .with_replicated(ReplicatedRoute::<WorkspaceBackend, _>::new(
            context,
            context,
            SharedReplicatedTextVisitor::<WorkspaceResidentStateFactory, _>::new(
                DestinationVisitor(context),
            ),
        ).with_prediction::<ProjectionMaterializer, _, _>(
            |prepared, _store| prediction::materialize(prepared, context),
            |_| DestinationVisitor(context),
        ))
        .with_routed(RoutedRoute::<WorkspaceBackend, ResidentState, ResidentState, _, _, _>::new(
            context, context, DestinationVisitor(context), DestinationVisitor(context), DestinationVisitor(context),
        ).with_prediction::<ProjectionMaterializer, _, _>(
            |prepared, _store| prediction::materialize(prepared, context),
            |_| DestinationVisitor(context),
        ))
        .with_composite(CompositeRoute::<WorkspaceBackend, ResidentState, _>::new(
            context,
            context,
            DestinationVisitor(context),
        ).with_prediction::<ProjectionMaterializer, _, _>(
            |prepared, _store| prediction::materialize(prepared, context),
            |_| DestinationVisitor(context),
        ));
    construct_prepared_execution_impl(
        sources.clone(),
        None::<()>,
        routes,
        DestinationAssembler(
            sources
                .selected()
                .text_realization()
                .state()
                .floating_dtype(),
        ),
        ConstructionPurpose::BindingDestinations,
    )
}

struct DestinationVisitor<'a>(&'a WorkspaceContext);

impl ReplicatedTextArchitectureVisitor<WorkspaceBackend, ResidentState> for DestinationVisitor<'_> {
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;

    fn construction_started(&mut self) {}

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        _store: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
        A::StaticModules: Clone,
    {
        self.collect_modules(prepared.into_modules())
    }
}

impl DestinationVisitor<'_> {
    fn collect_modules<A>(
        self,
        mut modules: PreparedReplicatedTextModules<A>,
    ) -> Result<ReplicatedTextBindingDestinations, Error>
    where
        A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
    {
        let contract = modules.take_contract();
        let architecture = modules.take_architecture();
        let static_parameters = collect(architecture.static_modules())?;
        let layout = contract.selected().requirements().execution_units();
        let units = (0..layout.len())
            .map(|ordinal| {
                let address = layout.address(ordinal).expect("validated execution layout");
                let unit = architecture.build_unit(address.group(), address.index(), self.0)?;
                collect(&unit)
            })
            .collect::<Result<_, Error>>()?;
        Ok(ReplicatedTextBindingDestinations {
            contract,
            static_parameters,
            units,
            prediction: Vec::new(),
        })
    }
}

impl CompositeTextArchitectureVisitor<WorkspaceBackend, ResidentState> for DestinationVisitor<'_> {
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;

    fn construction_started(&mut self) {}

    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        let (modules, _admission) = prepared.into_modules();
        self.collect_modules(modules)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        // Only the actual static/unit destinations belong to this source
        // component. Addressable banks keep their selected separate producer.
        let (routed, _processor, _admission) = prepared.into_parts();
        let (modules, _banks, _options) = routed.into_parts();
        self.collect_modules(modules)
    }
}

fn collect(
    module: &(impl Parameterized<WorkspaceTensor> + ?Sized),
) -> Result<BTreeMap<String, WorkspaceLayout>, Error> {
    #[derive(Default)]
    struct Collector {
        values: BTreeMap<String, WorkspaceLayout>,
        duplicate: Option<String>,
    }
    impl<'a> ParameterVisitor<'a, WorkspaceTensor> for Collector {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a WorkspaceTensor) {
            let name = metadata.id.as_str().to_owned();
            if self
                .values
                .insert(name.clone(), value.layout().clone())
                .is_some()
            {
                self.duplicate = Some(name);
            }
        }
    }
    let mut collector = Collector::default();
    module.visit_parameters(&mut collector);
    if let Some(parameter) = collector.duplicate {
        return Err(Error::backend_source(
            eredu_runtime::ModuleBindingPlanError::DuplicateParameter { parameter },
        ));
    }
    Ok(collector.values)
}

struct DestinationAssembler(Option<eredu_runtime::StateStorageDtype>);

impl PreparedExecutableAssembler<()> for DestinationAssembler {
    type Executable = ReplicatedTextBindingDestinations;
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;

    fn floating_state_dtype(
        &mut self,
        _: &FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Error> {
        Ok(self.0.unwrap_or(eredu_runtime::StateStorageDtype::F32))
    }

    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), Error> {
        Err(Error::backend(
            "replicated destination projection has no communication",
        ))
    }

    fn finish(
        self,
        parts: PreparedExecutableParts<Self::Executable, ()>,
    ) -> Result<Self::Output, Error> {
        Ok(parts.into_executable())
    }
}

mod addressable;
pub use addressable::{AddressableBindingDestinations, project_addressable_binding_destinations};
