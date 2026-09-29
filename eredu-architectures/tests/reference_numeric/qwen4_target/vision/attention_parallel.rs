//! Unequal and repeated attention segments retain numerical TP parity.
use super::*;

struct RankBind<'a> {
    owner: &'a PreparedParameters,
    context: &'a NumericContext,
    rank: usize,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for RankBind<'_> {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = meta.id.as_str();
        let full = super::super::super::payload::recipe_value(
            &self.owner.recipes()[name],
            self.owner.source().as_ref(),
            self.context,
        )
        .unwrap();
        let local = if name.contains(".attn.qkv.") {
            let hidden = full.dim(0) as usize / 3;
            let width = hidden / 2;
            let pieces = (0..3)
                .map(|part| {
                    full.axis_slice(
                        0,
                        part * hidden + self.rank * width,
                        part * hidden + (self.rank + 1) * width,
                    )
                })
                .collect::<Vec<_>>();
            NumericTensor::concatenate(&pieces, 0, self.context).unwrap()
        } else if name.ends_with(".attn.proj.weight") || name.ends_with(".mlp.linear_fc2.weight") {
            let width = full.dim(1) as usize / 2;
            full.axis_slice(1, self.rank * width, (self.rank + 1) * width)
        } else if name.contains(".mlp.linear_fc1.") {
            let width = full.dim(0) as usize / 2;
            full.axis_slice(0, self.rank * width, (self.rank + 1) * width)
        } else {
            full
        };
        assert_eq!(local.shape(), value.shape(), "{name}");
        *value = local;
    }
}

#[test]
fn repeated_unequal_attention_segments_match_tp2() {
    let (_directory, _target, ingress) = setup();
    let config = ingress.vision().config();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut full =
        VisionBlock::<NumericBackend>::new_with_root(config, "model.visual", 0, &context).unwrap();
    full.visit_parameters_mut(&mut Bind(ingress.vision().block(0).unwrap(), &context));
    let hidden = NumericTensor::new(
        [7, 8],
        (0..56)
            .map(|index| ((index * 7 % 31) as f32 - 15.) / 32.)
            .collect(),
    );
    let angles = (0..7)
        .flat_map(|position| {
            [
                position as f32 * 0.11,
                position as f32 * 0.07,
                position as f32 * 0.11,
                position as f32 * 0.07,
            ]
        })
        .collect::<Vec<_>>();
    let cosine = NumericTensor::new([7, 4], angles.iter().map(|angle| angle.cos()).collect());
    let sine = NumericTensor::new([7, 4], angles.iter().map(|angle| angle.sin()).collect());
    let expected = full
        .forward(&hidden, &[2, 3, 2], &cosine, &sine, &context)
        .unwrap();
    assert!(expected.data.iter().any(|value| value.abs() > 1e-6));
    let group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let group = group.clone();
                let (hidden, cosine, sine) = (&hidden, &cosine, &sine);
                let owner = ingress.vision().block(0).unwrap();
                scope.spawn(move || {
                    let context = NumericContext {
                        bind_checkpoint_values: true,
                        ..Default::default()
                    };
                    let parallel = NumericParallelContext::new(rank, group);
                    let mut block = VisionBlock::<NumericBackend>::new_parallel_with_root(
                        config,
                        "model.visual",
                        0,
                        1,
                        config.intermediate_size / 2,
                        &context,
                    )
                    .unwrap();
                    block.visit_parameters_mut(&mut RankBind {
                        owner,
                        context: &context,
                        rank,
                    });
                    let output = block
                        .forward_parallel(hidden, &[2, 3, 2], cosine, sine, &parallel, &context)
                        .unwrap();
                    assert_eq!(parallel.trace().len(), 2);
                    output
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for output in outputs {
        assert_tensor_close(&output, &expected, "TP2 segmented attention");
    }
}
