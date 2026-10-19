"""Audio anchor timing and QC; these checks do not establish physical A/V sync."""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

SYNC_VERSION = 'audio_anchor_qc_v1'


def refine_880_centers(audio, sr, candidates):
    """Locate a 200 ms marker's leading edge, then return onset + 100 ms.

    The maximum of a wide-band Hilbert envelope can select either end of a
    rectangular tone. Use a smoothed 20% leading edge in a bounded neighbourhood
    instead. This remains an acoustic timing estimate, with microphone/room and
    recording latency unverified. Reject clipped or unsupported candidates.
    440/1760 Hz timing is deliberately left unchanged.
    """
    audio = np.asarray(audio, dtype=float)
    if len(audio) < 100 or sr <= 2000 or not np.isfinite(audio).all():
        return []
    envelope = np.abs(hilbert(sosfiltfilt(
        butter(4, [820, 940], fs=sr, btype='band', output='sos'), audio)))
    envelope = sosfiltfilt(butter(2, 100, fs=sr, output='sos'), envelope)
    centers = []
    for candidate in candidates:
        lo = max(0, int((candidate - .35) * sr))
        hi = min(len(audio), int((candidate + .15) * sr))
        if hi - lo < int(.2 * sr):
            continue
        peak = lo + int(np.argmax(envelope[lo:hi]))
        level = envelope[peak]
        if level <= 0:
            continue
        # Estimate the local pre-marker floor; reject a tone already active
        # at the search boundary, rather than inventing an onset there.
        floor = envelope[lo:lo + max(1, int(.03 * sr))]
        baseline = float(np.median(floor))
        noise = float(np.median(np.abs(floor - baseline)))
        threshold = max(.2 * level, baseline + 6 * noise)
        edges = np.flatnonzero(envelope[lo:peak + 1] >= threshold)
        if not len(edges):
            continue
        onset = lo + int(edges[0])
        if onset <= lo + int(.01 * sr) or onset + int(.2 * sr) >= len(audio):
            continue
        support = envelope[onset:onset + int(.1 * sr)]
        if np.mean(support >= threshold) < .8:
            continue
        center = onset / sr + .1
        if abs(center - candidate) <= .15:
            centers.append(float(center))
    return centers


def summarize_anchor_offsets(offsets, frequencies, max_family_difference_ms=20.):
    """Keep observed frequency families separate and flag timing limitations.

    If both trial families disagree, use the previously verified narrow-band
    440 Hz centers rather than averaging incompatible definitions. Never infer
    an additional gaze shift or claim absolute sync from small anchor scatter.
    """
    families = {}
    for frequency in (440, 880, 1760):
        values = np.asarray([o for o, f in zip(offsets, frequencies)
                             if f == frequency and np.isfinite(o)], dtype=float)
        families[str(frequency)] = dict(
            count=int(len(values)),
            median_offset_ms=float(np.median(values)) if len(values) else None,
            std_ms=float(np.std(values)) if len(values) else None)
    warnings = ['absolute_audio_video_latency_unverified',
                'experiment_log_timestamp_semantics_unverified',
                'clock_drift_not_fitted']
    a, b = (families[str(f)]['median_offset_ms'] for f in (440, 880))
    difference = b - a if a is not None and b is not None else None
    if a is not None and b is not None and abs(difference) <= max_family_difference_ms:
        selected = [440, 880]
        reason = 'trial_frequency_medians_agree_within_20ms'
    elif a is not None:
        selected = [440]
        reason = '440hz_only'
        if b is not None:
            warnings.append('trial_frequency_disagreement_880_excluded')
    elif b is not None:
        selected = [880]
        reason = '880hz_only'
    else:
        selected = []
        reason = 'no_observed_trial_anchors'
    values = [o for o, f in zip(offsets, frequencies) if f in selected]
    if len(selected) < 2:
        warnings.append('trial_sync_not_cross_checked_between_frequencies')
    if len(values) < 5:
        warnings.append('insufficient_trial_anchors')
    return dict(
        version=SYNC_VERSION, marker_880_timing='20pct_leading_edge_plus_100ms',
        frequency_families=families, trial_frequency_difference_ms=difference,
        selected_trial_frequencies=selected, selection_reason=reason,
        selected_trial_offset_ms=float(np.median(values)) if values else None,
        anchor_std_ms=float(np.std(values)) if values else None,
        absolute_sync_verified=False, drift_fitted=False, warnings=warnings)
