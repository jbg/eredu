//! Physical member rows authenticated against the ordinary compact constructor.
use super::*;
use crate::backend::error::Error as NativeError;
use crate::backend::runtime::residency::parameter_bank::{
    IndexedBankSource, IndexedBindingIdentity, IndexedBindingStorage,
};
use eredu_nn::{
    GroupedNeuralBackend, ParameterMetadataView, ParameterSourceVisitor, Parameterized,
};

struct Declaration {
    name: String,
    layout: WorkspaceLayout,
}
struct Declarations<'a> {
    context: &'a WorkspaceContext,
    values: Vec<Declaration>,
    error: Option<Error>,
}
impl<'a> ParameterSourceVisitor<'a, WorkspaceTensor> for Declarations<'_> {
    fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a WorkspaceTensor) {
        if self.error.is_some() {
            return;
        }
        let result = (|| {
            self.context.reserve_metadata_vec(&mut self.values, 1)?;
            let name = self
                .context
                .metadata_string(format_args!("{}", metadata.id().as_str()))?;
            let layout = self.context.layout(value.shape(), value.layout().dtype())?;
            self.values.push(Declaration { name, layout });
            Ok(())
        })();
        if let Err(cause) = result {
            self.error = Some(cause);
        }
    }
    fn retained(&mut self, _: &'a WorkspaceTensor) {
        self.error = Some(self.context.metadata_error(format_args!(
            "compact parameter declaration contains unnamed storage"
        )));
    }
}
pub(super) struct ParameterRow {
    pub(super) layout: WorkspaceLayout,
    /// Actual full replacement descriptor and its selected original row.
    replacement: Option<(WorkspaceLayout, usize)>,
    pub(super) slice: Option<SpeculativeNumericalRecipe>,
    pub(super) ordinary_slice_calls: Option<OrdinaryCallControls>,
}
pub(super) struct ParameterRows {
    pub(super) identity: IndexedBindingIdentity,
    pub(super) rows: Vec<ParameterRow>,
    pub(super) fields: usize,
}
impl ParameterRows {
    pub(super) fn prepare(
        bank: &IndexedBankSource,
        source: WorkspaceAddressableRegionView<'_>,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "addressable member row differs from its retained compact parameter declaration"
            ))
        };
        context.charge_metadata(size_of::<(
            Self,
            Declarations<'_>,
            ParameterRow,
            Option<ParameterRow>,
            IndexedBindingIdentity,
            Result<Self, Error>,
            [usize; 6],
        )>())?;
        let mut declarations = Declarations {
            context,
            values: context.metadata_vec(0)?,
            error: None,
        };
        // This is the same portable constructor's physical ordering. Its
        // unloaded values are used only as declarations, never numerical input.
        match source.kernel.compact(1, context)? {
            WorkspaceGroupedBank::Linear(spec) => {
                WorkspaceBackend::grouped_linear_bank(spec, context)?
                    .visit_parameter_sources(&mut declarations)
            }
            WorkspaceGroupedBank::GatedProduct(spec) => {
                WorkspaceBackend::grouped_gated_product(spec, context)?
                    .visit_parameter_sources(&mut declarations)
            }
            WorkspaceGroupedBank::Relu2(spec) => WorkspaceBackend::grouped_relu2(spec, context)?
                .visit_parameter_sources(&mut declarations),
        }
        .map_err(|cause| context.metadata_source(cause))?;
        if let Some(cause) = declarations.error {
            return Err(cause);
        }
        let fields = declarations.values.len();
        if fields == 0 {
            return Err(invalid());
        }
        let count = fields
            .checked_mul(source.chunks.members)
            .ok_or_else(invalid)?;
        let mut slots = context.metadata_vec(count)?;
        slots.resize_with(count, || None);
        let identity = bank
            .with_layouts(source, context, |actual| {
                let row = (|| {
                    let member = match source.local_members {
                        Some(ids) => ids
                            .binary_search(&actual.member.key.member())
                            .map_err(|_| invalid())?,
                        None => actual.member.key.member(),
                    };
                    let field = declarations
                        .values
                        .iter()
                        .position(|p| p.name == actual.member.parameter)
                        .ok_or_else(invalid)?;
                    let index = member
                        .checked_mul(fields)
                        .and_then(|n| n.checked_add(field))
                        .filter(|&n| n < count)
                        .ok_or_else(invalid)?;
                    if slots[index].is_some() {
                        return Err(invalid());
                    }
                    let row = match actual.storage {
                        IndexedBindingStorage::Copy(row) => ParameterRow {
                            layout: context
                                .layout(row.shape, row.dtype)?
                                .with_representation(row.representation),
                            replacement: None,
                            slice: None,
                            ordinary_slice_calls: None,
                        },
                        IndexedBindingStorage::Replacement { value, row } => {
                            // Descriptor projection performs no native tensor work.
                            let mut projection =
                                ExistingArrayProjection::with_source_count(context, 1)
                                    .map_err(|cause| context.metadata_source(cause))?;
                            let full = projection.project(value.as_array())?;
                            if !projection.is_complete() {
                                return Err(invalid());
                            }
                            let layout = context
                                .layout(full.shape(), full.layout().dtype())?
                                .with_representation(full.layout().representation());
                            ParameterRow {
                                layout: context.layout(
                                    declarations.values[field].layout.shape(),
                                    layout.dtype(),
                                )?,
                                replacement: Some((layout, row)),
                                slice: None,
                                ordinary_slice_calls: None,
                            }
                        }
                    };
                    slots[index] = Some(row);
                    Ok(())
                })();
                row.map_err(NativeError::Neural)
            })
            .map_err(|cause| context.metadata_source(cause))?;
        let mut rows = context.metadata_vec(count)?;
        for (index, slot) in slots.into_iter().enumerate() {
            let mut row = slot.ok_or_else(invalid)?;
            if let Some((full, ordinal)) = row.replacement.as_ref() {
                let slice_context = WorkspaceContext::new_with_metadata_funding(
                    mechanism,
                    context.metadata_funding().ok_or_else(invalid)?,
                )?;
                let value = WorkspaceTensor::existing(
                    slice_context
                        .layout(full.shape(), full.dtype())?
                        .with_representation(full.representation()),
                    &slice_context,
                )?;
                let start = i32::try_from(*ordinal).map_err(|_| invalid())?;
                slice_context.begin_span();
                let value = value.narrow_axis(
                    0,
                    start,
                    start.checked_add(1).ok_or_else(invalid)?,
                    &slice_context,
                )?;
                row.layout = context
                    .layout(value.shape(), value.layout().dtype())?
                    .with_representation(value.layout().representation());
                let report = slice_context.finish_report(&[value])?;
                row.ordinary_slice_calls = super::quote::ordinary_report_calls(mechanism, &report)?;
                row.slice = Some(SpeculativeNumericalRecipe::inspect_owned_child(
                    &report,
                    1,
                    mechanism,
                    &slice_context,
                )?);
            }
            let expected = &declarations.values[index % fields].layout;
            if row.layout.shape() != expected.shape() || row.layout.dtype() != expected.dtype() {
                return Err(invalid());
            }
            rows.push(row);
        }
        Ok(Self {
            identity,
            rows,
            fields,
        })
    }
}
