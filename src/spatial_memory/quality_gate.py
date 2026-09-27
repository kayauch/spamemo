"""Conservative, shared acceptance policy; frame validation is not room validation."""
import math


POLICY_VERSION = 'fixed-camera-v1'


def compare_reports(baseline, candidate):
    reasons = []
    for key in ('image_size', 'alignment_sha256', 'camera_files_sha256', 'geometry_npz_sha256'):
        if not baseline.get(key) or baseline.get(key) != candidate.get(key):
            reasons.append(f'Incomparable {key}')
    tables = []
    for report in (baseline, candidate):
        rows = report.get('test_views', [])
        table = {r['frame']: r for r in rows}
        tables.append(table)
        if len(table) < 3 or len(table) != len(rows):
            reasons.append('Need at least three distinct held-out views')
        values = [r.get(k, float('nan')) for r in rows
                  for k in ('psnr', 'ssim', 'depth_abs_relative', 'alpha_coverage')]
        if not all(math.isfinite(v) for v in values):
            reasons.append('Missing or nonfinite evaluation values')
        if not report.get('finite_parameters') or not 0 <= report.get('max_camera_center_error', float('inf')) <= 1e-4:
            reasons.append('Invalid model or camera alignment')
    before, after = tables
    if before.keys() != after.keys():
        reasons.append('Held-out frame sets differ')
    if reasons:
        return dict(policy=POLICY_VERSION, accepted=False, reasons=reasons)
    deltas = {k: sum(after[f][k] - before[f][k] for f in before)/len(before)
              for k in ('psnr', 'ssim', 'depth_abs_relative', 'alpha_coverage')}
    worst = min(after[f]['psnr'] - before[f]['psnr'] for f in before)
    # Conservative engineering guardrails, not universal perceptual thresholds.
    if deltas['psnr'] < -.05: reasons.append('Mean PSNR regressed')
    if deltas['ssim'] < -.005: reasons.append('Mean SSIM regressed')
    if worst < -.5: reasons.append('A held-out view regressed by more than 0.5 dB')
    if deltas['depth_abs_relative'] > .005: reasons.append('VGGT depth agreement regressed')
    if deltas['alpha_coverage'] < -.01: reasons.append('Depth-mask coverage regressed')
    if not (deltas['psnr'] >= .05 or deltas['ssim'] >= .001 or deltas['depth_abs_relative'] <= -.005):
        reasons.append('No measurable improvement')
    return dict(policy=POLICY_VERSION, accepted=not reasons, reasons=reasons,
                mean_delta=deltas, worst_view_psnr_delta=worst,
                scope='Held-out frames of this scene; VGGT geometry uses all frames. Not unseen-room validation.')
