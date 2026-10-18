"""Blink event comparison on the synchronized gaze clock."""
import numpy as np
import pandas as pd


def synchronized_frames(gaze, frame_log):
    fields = frame_log.drop(columns=['valid_eye_frame'], errors='ignore')
    out = gaze[['frame', 'timestamp_ms_synced', 'trial_number', 'valid_eye_frame']].merge(
        fields, left_on='frame', right_on='frame_number', how='left', validate='one_to_one')
    out = out.sort_values('timestamp_ms_synced').reset_index(drop=True)
    time = out['timestamp_ms_synced'].to_numpy(float)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('Diagnostic gaze timestamps must be strictly increasing')
    # Estimate sample support only on the synchronized clock, never from PTS.
    out['sample_end_ms'] = np.r_[time[1:], time[-1] + np.median(np.diff(time))]
    return out


def predicted_intervals(q, start, end, regions):
    """Map exported invalid regions to synchronized sample support by frame."""
    out = []
    for region in regions.itertuples():
        group = q[q['frame_number'].between(region.start_frame, region.end_frame)]
        if group.empty:
            continue
        a = max(start, float(group['timestamp_ms_synced'].min()))
        b = min(end, float(group['sample_end_ms'].max()))
        if b > a:
            out.append((a-start, b-start, str(region.reason_group),
                        int(group['frame_number'].min()), int(group['frame_number'].max())))
    return out


def metrics_for_trial(vp, trial, predicted, truth, match_events, tolerance_ms=100.0):
    matches = match_events(predicted, truth, tolerance_ms)
    tp = len(matches)
    fp, fn = len(predicted)-tp, len(truth)-tp
    row = dict(vp=vp, trial=trial, tp=tp, fp=fp, fn=fn,
               n_webcam=len(predicted), n_eyelink=len(truth), tolerance_ms=tolerance_ms,
               timestamp_source='timestamp_ms_synced')
    row.update(scores(tp, fp, fn))
    linked = dict(matches)
    records = []
    for i, event in enumerate(predicted):
        j = linked.get(i)
        records.append(dict(vp=vp, trial=trial, pred_index=i,
            pred_start_ms=event[0], pred_end_ms=event[1], reason_group=event[2],
            matched=j is not None, eyelink_index=j,
            eyelink_start_ms=truth[j][0] if j is not None else np.nan,
            eyelink_end_ms=truth[j][1] if j is not None else np.nan))
    return row, records


def scores(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else np.nan
    recall = tp/(tp+fn) if tp+fn else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    return dict(precision=precision, recall=recall, f1=f1)


def save_statistics(extra, vp, rows, records):
    metrics = pd.DataFrame(rows)
    metrics.to_csv(extra / 'blink_diagnostic_metrics.csv', index=False)
    pd.DataFrame(records, columns=['vp', 'trial', 'pred_index', 'pred_start_ms',
        'pred_end_ms', 'reason_group', 'matched', 'eyelink_index',
        'eyelink_start_ms', 'eyelink_end_ms']).to_csv(extra / 'blink_event_matches.csv', index=False)
    tp, fp, fn = (int(metrics[k].sum()) for k in ['tp', 'fp', 'fn'])
    pd.DataFrame([dict(vp=vp, n_trials=len(rows), tp=tp, fp=fp, fn=fn,
        **scores(tp, fp, fn), timestamp_source='timestamp_ms_synced',
        tolerance_ms=100.0)]).to_csv(extra / 'blink_diagnostic_summary.csv', index=False)
