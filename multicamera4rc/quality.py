"""Confidence-independent checkpoint comparison across fixed evaluation suites."""

import math


def error_panel(reports):
    """Use per-source errors, never total loss or confidence, to select a model."""
    panel = {}
    def add(prefix, metrics, names):
        for name in names:
            value = metrics.get(name)
            if value is None:
                raise ValueError(f'missing evaluation metric {prefix}/{name}')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid evaluation metric {prefix}/{name}')
            panel[prefix+'/'+name] = value
    add('kubric', reports['kubric']['mean'], ('depth_abs_rel', 'base_epe_normalized',
        'trajectory_epe_temporal_normalized', 'displacement_epe_dynamic_normalized'))
    c4 = reports['kubric'].get('by_layout', {}).get('[4, 6]')
    if c4 is not None:
        add('kubric_c4t6', c4, ('depth_abs_rel', 'base_epe_normalized',
            'trajectory_epe_temporal_normalized', 'displacement_epe_dynamic_normalized'))
    if 'variable' in reports:
        add('variable', reports['variable']['mean'], ('depth_abs_rel', 'base_epe_normalized',
            'trajectory_epe_temporal_normalized', 'displacement_epe_dynamic_normalized'))
    for source, metrics in reports['static']['by_source'].items():
        add('static/'+source, metrics, ('depth_abs_rel', 'base_epe_normalized'))
    for source, metrics in reports['native_dynamic']['by_source'].items():
        names = ('native_trajectory_epe_temporal_normalized', 'native_displacement_epe_temporal_normalized') if source == 'pointodyssey' else (
            'trajectory_epe_temporal_normalized', 'displacement_epe_dynamic_normalized')
        add('native/'+source, metrics, names)
    return panel


def compare_errors(reference, candidate):
    if set(reference) != set(candidate) or not reference:
        raise ValueError('quality panels must contain the same nonempty metric set')
    ratios = {}
    for key, before in reference.items():
        after = candidate[key]
        if not all(math.isfinite(v) and v >= 0 for v in (before, after)):
            raise ValueError('quality measurements must be finite and nonnegative')
        ratios[key] = max(after, 1e-9)/max(before, 1e-9)
    score = math.exp(sum(math.log(v) for v in ratios.values())/len(ratios))
    worst = max(ratios.values())
    return dict(score_ratio=score, worst_metric_ratio=worst, metric_ratios=ratios,
                improved_metrics=sum(v < 1 for v in ratios.values()), metric_count=len(ratios),
                # A guard against damaging one task while improving an average.
                accepted=score < .995 and worst <= 1.10,
                allow_longer_sequences=score <= 1.02 and worst <= 1.10)
