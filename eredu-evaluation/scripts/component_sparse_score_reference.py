"""Independent float64 score accounting for the released LFM2 reference run."""
import math

import torch


def score_context(model, step, targets):
    residual = torch.tensor(step['readout']['residual'], dtype=torch.float64)
    target, alternative = (model.lm_head.weight[token].detach().double() for token in targets)
    rows = torch.stack([target, target - alternative])
    gain = model.model.embedding_norm.weight.detach().double()
    epsilon = float(torch.tensor(model.config.norm_eps, dtype=torch.float32))
    inverse = (residual.square().mean() + epsilon).sqrt().reciprocal()
    exact = rows * gain * inverse
    # The public projection contract accepts binary32 directions. Preserve that
    # boundary, and report its contribution separately from model arithmetic.
    directions = exact.float().double()
    return dict(targets=targets, rows=rows, exact=exact, directions=directions, residual=residual)


def finish_scores(model, step, groups, context):
    width = len(context['residual'])
    state = torch.tensor(step['readout']['embedding'], dtype=torch.bfloat16)
    directions = context['directions']
    residual_rounding, other_terms = [[], []], [[], []]
    other_count, write_count = 0, 0
    for layer, block in enumerate(model.model.layers):
        mixer = 'attention' if hasattr(block, 'self_attn') else 'convolution'
        ffn = 'routed' if hasattr(block.feed_forward, 'experts') else 'dense'
        for kind in (mixer, ffn):
            write = torch.tensor(step['writes'][f'{kind}:{layer}'][-width:], dtype=torch.float64)
            after = state + write.bfloat16()
            rounding = after.double() - state.double() - write
            for mode in range(2):
                residual_rounding[mode].append(float(rounding @ directions[mode]))
                if kind == 'convolution':
                    other_terms[mode].append(float(write @ directions[mode]))
            state = after
            other_count += kind == 'convolution'
            write_count += 1
    assert torch.equal(state.double(), context['residual']), 'writes do not reproduce actual BF16 residual'
    head = torch.tensor(step['head_input'], dtype=torch.float64)
    embedding = torch.tensor(step['readout']['embedding'], dtype=torch.float64)
    target, alternative = (step['logits'][token] for token in context['targets'])

    def output_step(value):
        return math.ldexp(1., max(-126, math.frexp(abs(value))[1] - 1) - 7) if value else math.ldexp(1., -133)

    scores = []
    for mode in range(2):
        components = math.fsum(group['score_directions'][mode]['signed_component_sum'] for group in groups.values())
        operator_rounding = math.fsum(group['score_directions'][mode]['signed_observed_write'] -
                                     group['score_directions'][mode]['signed_component_sum'] for group in groups.values())
        embedding_sum = float(embedding @ directions[mode])
        other_sum = math.fsum(other_terms[mode])
        residual_sum = math.fsum(residual_rounding[mode])
        direction_rounding = float(context['residual'] @ (context['exact'][mode] - directions[mode]))
        affine = float(head @ context['rows'][mode])
        normalization_rounding = affine - float(context['residual'] @ context['exact'][mode])
        reconstructed = math.fsum([components, operator_rounding, embedding_sum, other_sum,
                                   residual_sum, direction_rounding, normalization_rounding])
        score = target - (alternative if mode else 0.)
        bound = output_step(target) + (output_step(alternative) if mode else 0.)
        assert abs(reconstructed - affine) <= 1e-9 * (1 + abs(affine))
        assert abs(score - reconstructed) <= bound + 1e-9
        scores.append(dict(mode='margin' if mode else 'target', target=context['targets'][0],
                           alternative=context['targets'][1], components=components,
                           operator_rounding=operator_rounding, embedding=embedding_sum, other_writes=other_sum,
                           residual_rounding=residual_sum, direction_rounding=direction_rounding,
                           normalization_rounding=normalization_rounding, normalization_offset=0.,
                           reconstructed=reconstructed, affine_score=affine, score=score,
                           absolute_error=abs(score - reconstructed), head_rounding_bound=bound))
    return dict(prediction=step['prediction'], residual_replay_exact=True,
                residual_writes=write_count, other_write_count=other_count, scores=scores)
