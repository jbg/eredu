//! Optional physical facts carried alongside unchanged logical workspace geometry.
use super::*;
use crate::{ParameterId, ParameterSpec};
use std::mem::{size_of, size_of_val};

/// An exact floating representation observed by the selected implementation.
/// Logical floating workspace storage remains the conservative four-byte type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceFloatingType {
    /// IEEE single precision.
    Float32,
    /// IEEE half precision.
    Float16,
    /// Brain floating point with eight exponent bits.
    Bfloat16,
}

/// Additional descriptive evidence, never a storage credit or native permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceRepresentation {
    dtype: WorkspaceFloatingType,
    row_contiguous: bool,
    last_axis_contiguous: bool,
    dense_axes: u32,
    element_strides: [u32;4],
    stride_rank: u8,
}
impl WorkspaceRepresentation {
    /// The provider must prove both scalar type and any positive layout claim
    /// from the same actual source or the selected operation's semantics.
    pub const fn new(dtype: WorkspaceFloatingType, row_contiguous: bool) -> Self {
        Self {
            dtype,
            row_contiguous,
            last_axis_contiguous: row_contiguous,
            dense_axes: 0,
            element_strides: [0;4],
            stride_rank: 0,
        }
    }
    /// Adds a provider-proved unit stride on the final logical axis whenever
    /// that axis has more than one element. Other strides remain unrestricted.
    /// This does not prove dense backing or grant a storage allowance.
    pub const fn with_last_axis_contiguous(mut self, value: bool) -> Self {
        self.last_axis_contiguous = self.row_contiguous || value;
        self
    }
    /// Whether each final-axis row is proved contiguous. A false value leaves
    /// that stride unrestricted; rank-zero and singleton axes are vacuous.
    pub const fn last_axis_contiguous(self) -> bool {
        self.last_axis_contiguous
    }
    /// Adds a proved dense permutation of the logical axes, from slowest to
    /// fastest. Nonunit strides must equal the products of faster dimensions.
    /// The compact scalar encoding holds at most eight axes; larger layouts
    /// cannot receive this additional permutation. No backing size or
    /// source authority is implied. Row-major layouts need no extra encoding.
    pub fn with_dense_axis_order(mut self, axes: &[usize]) -> Option<Self> {
        if axes.len()>8 {return None;}
        let mut seen=0u16;let mut encoded=(axes.len() as u32)<<24;
        for (position,&axis) in axes.iter().enumerate() {
            if axis>=axes.len() || seen&(1u16<<axis)!=0 {return None;}
            seen|=1u16<<axis;encoded|=(axis as u32)<<(position*3);
        }
        if !self.row_contiguous {self.dense_axes=encoded;}
        Some(self)
    }
    /// One axis of the proved dense order for this exact rank. Row-major
    /// evidence yields identity order; missing/incompatible evidence is None.
    pub fn dense_axis_at(self,rank:usize,position:usize)->Option<usize> {
        if position>=rank {return None;}
        if self.row_contiguous {return Some(position);}
        if rank>8 || (self.dense_axes>>24) as usize!=rank {return None;}
        Some(((self.dense_axes>>(position*3))&7) as usize)
    }
    /// Attach exact positive element strides on each non-singleton axis. Unit
    /// axes may use canonical strides because no selected element observes them.
    /// This fixed metadata slot holds at most four axes and u32 strides. Larger
    /// facts remain unavailable; this neither creates backing nor proves its span.
    /// The provider must authenticate the same source shape and readable storage.
    pub fn with_element_strides(mut self,strides:&[u64])->Option<Self> {
        if strides.is_empty() || strides.len()>4 {return None;}
        let mut values=[0u32;4];
        for (axis,&stride) in strides.iter().enumerate() {
            if stride==0 {return None;}
            values[axis]=u32::try_from(stride).ok()?;
        }
        if !self.row_contiguous {
            self.element_strides=values;self.stride_rank=strides.len() as u8;
        }
        Some(self)
    }
    /// One explicitly retained stride for this exact rank. Missing facts do not
    /// imply dense storage; row-major facts use the caller's existing shape.
    pub const fn element_stride_at(self,rank:usize,axis:usize)->Option<u64> {
        if rank==0 || rank>4 || axis>=rank || self.stride_rank as usize!=rank {return None;}
        Some(self.element_strides[axis] as u64)
    }
    /// Exact scalar representation.
    pub const fn dtype(self) -> WorkspaceFloatingType {
        self.dtype
    }
    /// Whether row contiguity is proved. False leaves layout unrestricted.
    pub const fn row_contiguous(self) -> bool {
        self.row_contiguous
    }
}

// Geometry equality remains the existing logical contract. Physical evidence
// can refine one selected mechanism without changing shape validation or roots.
impl PartialEq for WorkspaceLayout {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.dtype == other.dtype
    }
}
impl Eq for WorkspaceLayout {}
impl PartialEq for WorkspaceLayoutView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.dtype == other.dtype
    }
}
impl Eq for WorkspaceLayoutView<'_> {}
impl WorkspaceLayout {
    /// Optional exact selected physical facts, separate from logical geometry.
    pub const fn representation(&self) -> Option<WorkspaceRepresentation> {
        self.representation
    }
    /// Attaches provider evidence. Integer logical layouts cannot carry floating
    /// evidence. This does not authenticate a source, grant credit, or admit work.
    pub fn with_representation(mut self, value: Option<WorkspaceRepresentation>) -> Self {
        self.representation = value.filter(|_| self.dtype == WorkspaceDtype::Float32);
        self
    }
}
impl WorkspaceLayoutView<'_> {
    /// Attaches descriptive evidence from the same actual provider, without
    /// allocating shape storage or granting any backing/contiguity authority.
    /// Integer logical layouts cannot carry floating representation evidence.
    pub fn with_representation(mut self,value:Option<WorkspaceRepresentation>)->Self {
        self.representation=value.filter(|_|self.dtype==WorkspaceDtype::Float32);
        self
    }
    /// Optional exact selected physical facts, separate from logical geometry.
    pub const fn representation(self) -> Option<WorkspaceRepresentation> {
        self.representation
    }
}

/// One actual retained parameter's immutable descriptive row. The caller owns
/// source/epoch validation and pays the name, layout and table before construction.
#[derive(Debug)]
pub struct WorkspaceParameterRepresentation {
    id: ParameterId,
    layout: WorkspaceLayout,
}
impl WorkspaceParameterRepresentation {
    /// Moves already constructed metadata; no allocation or native owner clone.
    pub fn new(id: ParameterId, layout: WorkspaceLayout) -> Self {
        Self { id, layout }
    }
}

impl WorkspaceContext {
    /// Installs the selected source's descriptive rows once in this context.
    /// Callers must retain the exact source/parameter epoch through their quote
    /// and validate that epoch before executing it. This table cannot grant fit,
    /// source custody or native permission. Unsupported sources leave it absent.
    /// Conflicting duplicate declarations lose evidence rather than choosing one.
    pub fn install_parameter_representations(
        &self,
        mut rows: Vec<WorkspaceParameterRepresentation>,
    ) -> Result<(), Error> {
        let controls = [
            size_of::<Vec<WorkspaceParameterRepresentation>>(),
            size_of::<WorkspaceParameterRepresentation>(),
            size_of::<Option<WorkspaceRepresentation>>(),
            size_of::<Result<(), Error>>(),
        ];
        self.charge_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let mut installed = self.parameter_representations.borrow_mut();
        if installed.is_some() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        normalize_parameter_representations(&mut rows);
        *installed = Some(rows);
        Ok(())
    }

    /// Adds exact independently retained rows before parameter construction
    /// or span starts. The caller retains and validates every contributing source;
    /// matching names or logical shapes do not grant native storage authority.
    /// Conflicts use the same evidence-removal rule as initial installation.
    pub fn extend_parameter_representations(
        &self,
        mut rows: Vec<WorkspaceParameterRepresentation>,
    ) -> Result<(), Error> {
        let controls = [
            size_of::<Vec<WorkspaceParameterRepresentation>>(),
            size_of::<WorkspaceParameterRepresentation>(),
            size_of::<Option<WorkspaceRepresentation>>(),
            size_of::<std::cell::RefMut<'_, Option<Vec<WorkspaceParameterRepresentation>>>>(),
            size_of::<Result<(), Error>>(),
        ];
        self.charge_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        if self.tracing_started.get() || !self.trace.borrow().operations.is_empty() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut installed = self.parameter_representations.borrow_mut();
        match installed.as_mut() {
            Some(current) => {
                self.reserve_metadata_vec(current, rows.len())?;
                current.append(&mut rows);
                normalize_parameter_representations(current);
            }
            None => {
                normalize_parameter_representations(&mut rows);
                *installed = Some(rows);
            }
        }
        Ok(())
    }

    pub(super) fn parameter_layout(
        &self,
        parameter: &ParameterSpec,
        shape: &[i32],
        dtype: WorkspaceDtype,
    ) -> Result<WorkspaceLayout, Error> {
        let mut layout = self.layout(shape, dtype)?;
        if let Some(rows) = self.parameter_representations.borrow().as_ref() {
            if let Ok(index) = rows.binary_search_by(|row| row.id.cmp(&parameter.id)) {
                let source = &rows[index].layout;
                if source == &layout {
                    layout.representation = source.representation;
                }
            }
        }
        Ok(layout)
    }
}

// Shared duplicate policy for initial and additional exact source rows. Sorting
// changes only metadata order; contradictory evidence is never selected by order.
fn normalize_parameter_representations(rows:&mut Vec<WorkspaceParameterRepresentation>) {
    rows.sort_unstable_by(|a,b|a.id.cmp(&b.id));
    rows.dedup_by(|next,previous| {
        if next.id!=previous.id {return false;}
        if next.layout!=previous.layout || next.layout.representation!=previous.layout.representation {
            previous.layout.representation=None;
        }
        true
    });
}

// The optional callback and its fixed neutral return/transport controls are
// paid on the same cumulative ledger before inspecting an output.
pub(super) fn operation_control_bytes() -> Option<usize> {
    let controls = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<usize>()*5,
        size_of::<Option<usize>>()*2,
        size_of::<std::iter::Enumerate<std::slice::Iter<usize>>>(),
        size_of::<(usize,&usize)>(),
        size_of::<bool>()*3,
        size_of::<u32>()*2, size_of::<u16>(), size_of::<&[usize]>(),
        size_of::<[u32;4]>(),size_of::<u8>(),size_of::<&[u64]>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_,u64>>>(),size_of::<(usize,&u64)>(),
        size_of::<Option<u32>>(),size_of::<Option<u64>>(),
        size_of::<Result<(), WorkspaceMetadataError>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}

#[cfg(test)]
mod dense_order_tests {
    use super::*;
    #[test]
    fn element_stride_evidence_preserves_rank_and_does_not_infer_dense_storage() {
        let plain=WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false);
        let slice=plain.with_element_strides(&[4,1,2]).unwrap();
        assert!(!slice.row_contiguous());assert!(!slice.last_axis_contiguous());
        assert_eq!((0..3).map(|axis|slice.element_stride_at(3,axis)).collect::<Vec<_>>(),
            vec![Some(4),Some(1),Some(2)]);
        assert_eq!(slice.element_stride_at(2,0),None);assert_eq!(slice.element_stride_at(3,3),None);
        assert_eq!(slice.dense_axis_at(3,0),None);
        assert!(plain.with_element_strides(&[]).is_none());
        assert!(plain.with_element_strides(&[1,1,1,1,1]).is_none());
        assert!(plain.with_element_strides(&[1,0]).is_none());
        assert!(plain.with_element_strides(&[u32::MAX as u64+1]).is_none());
        let row=WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,true);
        assert_eq!(row.with_element_strides(&[4,1]).unwrap(),row);
    }
    #[test]
    fn dense_axis_evidence_preserves_exact_rank_and_refuses_missing_permutations() {
        let plain=WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)
            .with_last_axis_contiguous(true);
        assert_eq!(plain.dense_axis_at(4,0),None);
        let headed=plain.with_dense_axis_order(&[0,2,1,3]).unwrap();
        assert_eq!((0..4).map(|p|headed.dense_axis_at(4,p)).collect::<Vec<_>>(),
            vec![Some(0),Some(2),Some(1),Some(3)]);
        assert!(!headed.row_contiguous());assert!(headed.last_axis_contiguous());
        assert_eq!(headed.dense_axis_at(3,0),None);assert_eq!(headed.dense_axis_at(4,4),None);
        assert!(plain.with_dense_axis_order(&[0,2,2,3]).is_none());
        assert!(plain.with_dense_axis_order(&[0,1,4,3]).is_none());
        assert!(plain.with_dense_axis_order(&[0,1,2,3,4,5,6,7,8]).is_none());
        let row=WorkspaceRepresentation::new(WorkspaceFloatingType::Bfloat16,true);
        assert_eq!(row.dense_axis_at(4,2),Some(2));assert_eq!(row.dtype(),WorkspaceFloatingType::Bfloat16);
    }
}
