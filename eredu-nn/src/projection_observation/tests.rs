use super::*;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[test]
fn checked_actual_shapes_price_reconstruction_buffers_and_padded_blocks() {
    for (shape, creation_bytes) in [
        (&[2, 259][..], 9288),
        (&[129, 1][..], 67596),
        (&[2, 3, 256][..], 24576),
        (&[7][..], 596),
    ] {
        let plan = BlockFp8InputReconstructionPlan::new(shape).unwrap();
        assert_eq!(
            plan.logical_capture_source().unwrap(),
            GeneratedTensorSource {
                creation_bytes,
                element_type: Some(TensorElementType::F32),
            }
        );
        plan.validate_operands(&plan.values_shape(), &plan.scales_shape())
            .unwrap();
        assert!(
            plan.validate_operands(&[plan.rows(), plan.width() + 1], &plan.scales_shape())
                .is_err()
        );
        assert_eq!(plan.shape().as_ptr(), shape.as_ptr());
    }
    for shape in [&[][..], &[0][..], &[2, -1][..]] {
        assert!(matches!(
            BlockFp8InputReconstructionPlan::new(shape),
            Err(ProjectionObservationError::Geometry)
        ));
    }
    for shape in [&[i32::MAX][..], &[i32::MAX, 2][..], &[65536, 65536][..]] {
        assert!(matches!(
            BlockFp8InputReconstructionPlan::new(shape),
            Err(ProjectionObservationError::Overflow)
        ));
    }
}

struct Node {
    live: Rc<Cell<usize>>,
    ordinal: usize,
}
impl Drop for Node {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
struct Mechanism<'a> {
    live: Rc<Cell<usize>>,
    calls: &'a RefCell<Vec<usize>>,
    fail: Option<usize>,
    source: &'a Cell<bool>,
}
impl Mechanism<'_> {
    fn node(&self, ordinal: usize) -> Result<Node, usize> {
        assert!(self.source.get());
        self.calls.borrow_mut().push(ordinal);
        assert_eq!(self.live.get(), ordinal, "prior intermediates remain live");
        if self.fail == Some(ordinal) {
            return Err(ordinal);
        }
        self.live.set(self.live.get() + 1);
        Ok(Node {
            live: self.live.clone(),
            ordinal,
        })
    }
}
impl BlockFp8InputReconstructionMechanism for Mechanism<'_> {
    type Value = Node;
    type Error = usize;
    fn expand_scales(&self) -> Result<Node, usize> {
        self.node(0)
    }
    fn broadcast_scales(&self, v: &Node) -> Result<Node, usize> {
        assert_eq!(v.ordinal, 0);
        self.node(1)
    }
    fn flatten_scales(&self, v: &Node) -> Result<Node, usize> {
        assert_eq!(v.ordinal, 1);
        self.node(2)
    }
    fn decode_values(&self) -> Result<Node, usize> {
        self.node(3)
    }
    fn trim_scales(&self, v: &Node) -> Result<Node, usize> {
        assert_eq!(v.ordinal, 2);
        self.node(4)
    }
    fn multiply(&self, v: &Node, s: &Node) -> Result<Node, usize> {
        assert_eq!((v.ordinal, s.ordinal), (3, 4));
        self.node(5)
    }
    fn restore_shape(&self, v: &Node) -> Result<Node, usize> {
        assert_eq!(v.ordinal, 5);
        self.node(6)
    }
}
#[test]
fn shared_program_preserves_source_and_all_prior_values_until_final_operation_or_failure() {
    for fail in (0..7).map(Some).chain([None]) {
        let live = Rc::new(Cell::new(0));
        let calls = RefCell::new(Vec::new());
        let source = Cell::new(true);
        let result = reconstruct_block_fp8_input(Mechanism {
            live: live.clone(),
            calls: &calls,
            fail,
            source: &source,
        });
        match result {
            Ok(output) => {
                assert_eq!(live.get(), 1);
                assert_eq!(output.ordinal, 6);
                drop(output)
            }
            Err(e) => assert_eq!(Some(e), fail),
        }
        assert_eq!(live.get(), 0);
        assert!(source.get());
        assert_eq!(*calls.borrow(), (0..=fail.unwrap_or(6)).collect::<Vec<_>>());
    }
}
