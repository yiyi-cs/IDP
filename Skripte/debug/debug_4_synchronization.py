    # ══════════════════════════════════════════════════════════════════════
    # QUALITY / BLINK MERGE: MediaPipe -> ptgaze (v3.3)
    # ══════════════════════════════════════════════════════════════════════
    #
    # MediaPipe is the source of truth for eye-state / blink quality.
    #
    # In addition to the legacy blink fields, propagate:
    #
    #   valid_eye_frame
    #   invalid_reason
    #
    # These fields are required by downstream fixation detection.
    #
    # IMPORTANT:
    #   - timestamp_ms_synced is NOT modified here.
    #   - PTGaze keeps its own synchronized timestamps.
    #   - MediaPipe quality information is mapped to the temporally
    #     nearest PTGaze sample.
    # ══════════════════════════════════════════════════════════════════════

    print(
        "\n  [QUALITY-MERGE] "
        "Uebertrage MediaPipe Blink-/Quality-Daten auf ptgaze..."
    )

    # ------------------------------------------------------------------
    # Column groups
    # ------------------------------------------------------------------

    # Boolean / binary:
    # nearest-neighbour mapping
    blink_cols_binary = [
        "is_blink",
        "eyes_closed",
        "valid_eye_frame",
    ]

    # Continuous:
    # linear interpolation
    blink_cols_continuous = [
        "left_ear",
        "right_ear",
        "avg_ear",
    ]

    # Counter:
    # previous MediaPipe sample / forward-fill semantics
    blink_cols_counter = [
        "blink_count",
    ]

    # Categorical:
    # nearest-neighbour mapping
    quality_cols_categorical = [
        "invalid_reason",
    ]

    all_quality_cols = (
        blink_cols_binary
        + blink_cols_continuous
        + blink_cols_counter
        + quality_cols_categorical
    )

    available_quality_cols = [
        col
        for col in all_quality_cols
        if col in pupil_data_synced.columns
    ]

    missing_quality_cols = [
        col
        for col in all_quality_cols
        if col not in pupil_data_synced.columns
    ]

    if not available_quality_cols:

        print(
            "    [!] Keine Blink-/Quality-Spalten "
            "in MediaPipe-Daten gefunden"
        )

        print(
            "        Quality-Merge wird uebersprungen"
        )

    else:

        print(
            "    Verfuegbare Spalten: "
            + ", ".join(
                available_quality_cols
            )
        )

        if missing_quality_cols:

            print(
                "    Fehlende Spalten: "
                + ", ".join(
                    missing_quality_cols
                )
            )

        # ==============================================================
        # Sort both signals by the EXISTING synchronized timestamp
        # ==============================================================

        pupil_sorted = (
            pupil_data_synced
            .sort_values(
                "timestamp_ms_synced"
            )
            .reset_index(drop=True)
        )

        ptgaze_sorted = (
            ptgaze_data_synced
            .sort_values(
                "timestamp_ms_synced"
            )
            .reset_index(drop=True)
        )

        mp_times = (
            pupil_sorted[
                "timestamp_ms_synced"
            ]
            .to_numpy()
        )

        pt_times = (
            ptgaze_sorted[
                "timestamp_ms_synced"
            ]
            .to_numpy()
        )

        # ==============================================================
        # Common nearest-neighbour mapping
        #
        # Calculate ONCE and reuse for:
        #   - binary fields
        #   - categorical fields
        # ==============================================================

        indices = np.searchsorted(
            mp_times,
            pt_times,
        )

        indices = np.clip(
            indices,
            0,
            len(mp_times) - 1,
        )

        left_indices = np.clip(
            indices - 1,
            0,
            len(mp_times) - 1,
        )

        right_indices = indices

        left_dist = np.abs(
            pt_times
            - mp_times[left_indices]
        )

        right_dist = np.abs(
            pt_times
            - mp_times[right_indices]
        )

        nearest_indices = np.where(
            left_dist <= right_dist,
            left_indices,
            right_indices,
        )

        nearest_time_diff = np.minimum(
            left_dist,
            right_dist,
        )

        n_merged = 0

        # ==============================================================
        # 1. Binary / boolean fields
        # ==============================================================

        for col in blink_cols_binary:

            if col not in pupil_sorted.columns:
                continue

            mp_values = (
                pupil_sorted[col]
                .to_numpy()
            )

            ptgaze_sorted[col] = (
                mp_values[
                    nearest_indices
                ]
            )

            n_merged += 1

        # ==============================================================
        # 2. Categorical fields
        # ==============================================================

        for col in quality_cols_categorical:

            if col not in pupil_sorted.columns:
                continue

            mp_values = (
                pupil_sorted[col]
                .to_numpy()
            )

            ptgaze_sorted[col] = (
                mp_values[
                    nearest_indices
                ]
            )

            n_merged += 1

        # ==============================================================
        # 3. Continuous EAR fields
        #
        # Keep existing behaviour:
        # linear interpolation on timestamp_ms_synced.
        # ==============================================================

        for col in blink_cols_continuous:

            if col not in pupil_sorted.columns:
                continue

            mp_values = (
                pd.to_numeric(
                    pupil_sorted[col],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            valid_mask = (
                ~np.isnan(
                    mp_values
                )
            )

            if valid_mask.sum() > 1:

                ptgaze_sorted[col] = np.interp(
                    pt_times,
                    mp_times[
                        valid_mask
                    ],
                    mp_values[
                        valid_mask
                    ],
                )

            else:

                ptgaze_sorted[col] = np.nan

            n_merged += 1

        # ==============================================================
        # 4. Blink counter
        #
        # Use last MediaPipe sample that is not in the future.
        # ==============================================================

        for col in blink_cols_counter:

            if col not in pupil_sorted.columns:
                continue

            mp_values = (
                pupil_sorted[col]
                .to_numpy()
            )

            previous_indices = (
                np.searchsorted(
                    mp_times,
                    pt_times,
                    side="right",
                )
                - 1
            )

            previous_indices = np.clip(
                previous_indices,
                0,
                len(mp_times) - 1,
            )

            ptgaze_sorted[col] = (
                mp_values[
                    previous_indices
                ]
            )

            n_merged += 1

        # ==============================================================
        # Metadata
        # ==============================================================

        ptgaze_sorted[
            "blink_source"
        ] = "mediapipe"

        ptgaze_sorted[
            "eye_quality_source"
        ] = "mediapipe"

        ptgaze_sorted[
            "quality_merge_time_diff_ms"
        ] = nearest_time_diff

        # ==============================================================
        # IMPORTANT:
        #
        # PTGaze timestamp_ms_synced is intentionally NOT overwritten.
        # ==============================================================

        ptgaze_data_synced = (
            ptgaze_sorted
        )

        # ==============================================================
        # Diagnostics
        # ==============================================================

        max_time_diff = float(
            np.max(
                nearest_time_diff
            )
        )

        mean_time_diff = float(
            np.mean(
                nearest_time_diff
            )
        )

        print(
            f"    [OK] {n_merged} "
            f"Blink-/Quality-Spalten uebertragen"
        )

        print(
            "    Zeitliche Genauigkeit:"
        )

        print(
            f"      - Mittlere Abweichung: "
            f"{mean_time_diff:.1f} ms"
        )

        print(
            f"      - Maximale Abweichung: "
            f"{max_time_diff:.1f} ms"
        )

        # --------------------------------------------------------------
        # Quality diagnostics
        # --------------------------------------------------------------

        if (
            "valid_eye_frame"
            in ptgaze_sorted.columns
        ):

            valid_count = (
                ptgaze_sorted[
                    "valid_eye_frame"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

            invalid_count = (
                len(ptgaze_sorted)
                - valid_count
            )

            print(
                "    Eye-frame quality:"
            )

            print(
                f"      - Valid: "
                f"{valid_count}/{len(ptgaze_sorted)} "
                f"({100 * valid_count / len(ptgaze_sorted):.1f}%)"
            )

            print(
                f"      - Invalid: "
                f"{invalid_count}"
            )

        if (
            "invalid_reason"
            in ptgaze_sorted.columns
        ):

            invalid_reasons = (
                ptgaze_sorted.loc[
                    ~ptgaze_sorted[
                        "valid_eye_frame"
                    ]
                    .fillna(False)
                    .astype(bool),
                    "invalid_reason",
                ]
                .value_counts(
                    dropna=False
                )
            )

            if len(
                invalid_reasons
            ):

                print(
                    "    Invalid reasons:"
                )

                for (
                    reason,
                    count
                ) in (
                    invalid_reasons.items()
                ):

                    print(
                        f"      - "
                        f"{reason}: {count}"
                    )

        # --------------------------------------------------------------
        # Timing warning
        # --------------------------------------------------------------

        if max_time_diff > 100:

            print(
                "    [!] WARNUNG: "
                "Grosse Zeitdifferenz "
                "bei Quality-Merge!"
            )

            print(
                "        Moeglicherweise "
                "unterschiedliche Frameraten "
                "oder fehlende Samples"
            )

# Verteilung
for block_key in ['experiment_block1', 'experiment_block2']:
    if block_key in phases['phases']:
        trials_in_block = [t['trial_number'] for t in phases['phases'][block_key]['trials']]
        n_samples = int((pupil_data_synced['trial_assignment'].isin(trials_in_block)).sum())  # ← FIX!
        print(f"  {block_key}: {n_samples} Samples (Trials {min(trials_in_block)}-{max(trials_in_block)})")

# ==================== SCHRITT 6: VALIDIERUNG (OPTIONAL) ====================

print(f"\n{'='*70}")
print("VALIDIERUNG (gegen EyeLink-Blöcke)")
print(f"{'='*70}\n")

# Lade EyeLink-Blöcke (falls vorhanden)
blocks_csv = os.path.join(OUTPUT_BASE_DIR, "debug_3_blocks_overview.csv")

if not os.path.exists(blocks_csv):
    print(f" debug_3_blocks_overview.csv nicht gefunden")
    print(f"   Validierung übersprungen (nicht kritisch)")
    blocks_df = None
else:
    blocks_df = pd.read_csv(blocks_csv)
    print(f"Geladen: {len(blocks_df)} EyeLink-Blöcke")
    
    # Spalten-Check
    start_col = 'start_time_ms' if 'start_time_ms' in blocks_df.columns else 'start_time'
    end_col = 'end_time_ms' if 'end_time_ms' in blocks_df.columns else 'end_time'
    
    # Coverage-Check
    n_in_fixation = 0
    for _, block_row in blocks_df[blocks_df['block_type'] == 'fixation'].iterrows():
        block_start = block_row[start_col]
        block_end = block_row[end_col]
        
        pupil_in_block = pupil_data_synced[
            (pupil_data_synced['timestamp_ms_synced'] >= block_start) &
            (pupil_data_synced['timestamp_ms_synced'] <= block_end)
        ]
        n_in_fixation += len(pupil_in_block)
    
    video_fps = len(pupil_data) / ((pupil_data['timestamp_ms'].max() - pupil_data['timestamp_ms'].min()) / 1000)
    if blocks_df is not None and len(blocks_df) > 0:
        # Berechne durchschnittliche Fixations-Dauer aus debug_3
        fixation_blocks = blocks_df[blocks_df['block_type'] == 'fixation']
        avg_fixation_duration_s = fixation_blocks['duration_ms'].mean() / 1000
        
        print(f"    Durchschnittliche Fixations-Dauer (aus debug_3): {avg_fixation_duration_s:.2f}s")
        
        expected_per_fixation = avg_fixation_duration_s * video_fps
    else:
        # Fallback
        expected_per_fixation = EXPECTED_FIXATION_DURATION_S * video_fps
    n_fixation_phases = len(blocks_df[blocks_df['block_type'] == 'fixation'])
    avg_per_fixation = n_in_fixation / n_fixation_phases if n_fixation_phases > 0 else 0
    coverage = (avg_per_fixation / expected_per_fixation * 100) if expected_per_fixation > 0 else 0
    
    print(f"\nFixationsphasen: {n_fixation_phases}")
    print(f"Pupillen-Samples darin: {n_in_fixation}")
    print(f"  Ø {avg_per_fixation:.1f} Samples/Fixation")
    print(f"  Erwartet: {expected_per_fixation:.0f}")
    print(f"\nCoverage: {coverage:.1f}%")
    
    if coverage >= 80:
        print(f" Gute Coverage")
    elif coverage >= 50:
        print(f" Geringe Coverage")
    else:
        print(f" Sehr geringe Coverage")

# ==================== SCHRITT 7: VISUALISIERUNG ====================

print(f"\n{'='*70}")
print("VISUALISIERUNG")
print(f"{'='*70}\n")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# ─────────────────────────────────────────────────────────────────
# Panel 1: Offset-Übersicht (einfach, da konstant!)
# ─────────────────────────────────────────────────────────────────

ax1 = fig.add_subplot(gs[0, :])

# Zeige nur Offset-Linie (da konstant aus debug_0)
ax1.axhline(offset_ms, color='green', linewidth=3, label=f'Offset: {offset_ms:.0f} ms')
ax1.axhline(0, color='gray', linestyle=':', linewidth=1)

# Falls Validierungs-Daten vorhanden (aus debug_0)
if validation and validation.get('mean_deviation_ms'):
    std_dev = validation.get('std_deviation_ms', 0)
    ax1.fill_between([0, len(all_trials)], 
                     offset_ms - std_dev, offset_ms + std_dev,
                     alpha=0.2, color='green', 
                     label=f'±Std ({std_dev:.1f} ms)')

ax1.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Offset (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Synchronisations-Offset (aus debug_0)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(all_trials))

# Info-Text
info_text = (f"Methode: {sync_method}\n"
             f"Confidence: {confidence:.2%}\n"
             f"Audio-Validated: {'Ja' if validation else 'Nein'}")
ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ─────────────────────────────────────────────────────────────────
# Panel 2: Trial-Zuordnungs-Statistik
# ─────────────────────────────────────────────────────────────────

ax2 = fig.add_subplot(gs[1, 0])

# Zähle Samples pro Trial
trial_sample_counts = []
trial_numbers = []

for trial in all_trials:
    trial_num = trial['trial_number']
    n_samples = (pupil_data_synced['trial_assignment'] == trial_num).sum()
    trial_sample_counts.append(n_samples)
    trial_numbers.append(trial_num)

ax2.bar(trial_numbers, trial_sample_counts, color='steelblue', edgecolor='black', linewidth=0.5)
ax2.axhline(np.mean(trial_sample_counts), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(trial_sample_counts):.0f}')
ax2.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('Anzahl Samples', fontsize=11, fontweight='bold')
ax2.set_title('Pupillen-Samples pro Trial', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# ─────────────────────────────────────────────────────────────────
# Panel 3: Coverage-Check (falls EyeLink vorhanden)
# ─────────────────────────────────────────────────────────────────

ax3 = fig.add_subplot(gs[1, 1])

if blocks_df is not None:
    # Berechne Coverage pro Fixationsphase
    coverage_per_phase = []
    
    for _, block_row in blocks_df[blocks_df['block_type'] == 'fixation'].iterrows():
        block_start = block_row[start_col]
        block_end = block_row[end_col]
        
        pupil_in_block = pupil_data_synced[
            (pupil_data_synced['timestamp_ms_synced'] >= block_start) &
            (pupil_data_synced['timestamp_ms_synced'] <= block_end)
        ]
        
        phase_coverage = (len(pupil_in_block) / expected_per_fixation * 100) if expected_per_fixation > 0 else 0
        coverage_per_phase.append(phase_coverage)
    
    trial_nums_for_coverage = range(1, len(coverage_per_phase) + 1)
    
    colors = ['green' if c >= 80 else 'orange' if c >= 50 else 'red' 
             for c in coverage_per_phase]
    
    ax3.bar(trial_nums_for_coverage, coverage_per_phase, color=colors, 
           edgecolor='black', linewidth=0.5)
    ax3.axhline(80, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Gut (≥80%)')
    ax3.axhline(50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='OK (≥50%)')
    ax3.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Fixationsphasen-Coverage', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)
else:
    ax3.text(0.5, 0.5, 'Keine EyeLink-Blöcke\n(debug_3 fehlt)', 
            ha='center', va='center', fontsize=12, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax3.axis('off')

plt.suptitle('DEBUG 4: Synchronisation v3.0 (Vereinfacht mit debug_0)', 
             fontsize=16, fontweight='bold')

plot_path = os.path.join(OUTPUT_BASE_DIR, "debug_4_sync_quality.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f" Plot: {plot_path}")
plt.close()

# ==================== SCHRITT 8: EXPORT ====================

print(f"\n{'='*70}")
print("EXPORT")
print(f"{'='*70}\n")

output_path = os.path.join(OUTPUT_BASE_DIR, "debug_4_pupil_data_synced.csv")
pupil_data_synced.to_csv(output_path, index=False)
print(f" Pupil-Data: {output_path}")

# EXPORT PTGAZE (NEU v3.1)

if ptgaze_available and ptgaze_data_synced is not None:
    ptgaze_output_path = os.path.join(OUTPUT_BASE_DIR, "debug_4_ptgaze_gaze_synced.csv")
    ptgaze_data_synced.to_csv(ptgaze_output_path, index=False)
    print(f" ptgaze-Data: {ptgaze_output_path}")

# Sync-Info (erweitert) - v3.1 mit ptgaze-Support
sync_info_export = make_json_serializable({
    'sync_method': sync_method,
    'offset_ms': float(offset_ms),
    'offset_used_ms': float(offset_used),  # NEU: Tatsaechlich genutzter Offset
    'confidence': float(confidence),
    'source': 'phases_detected.json',
    'debug_0_version': phases['sync_info'].get('method', 'unknown'),
    
    # MediaPipe-Statistiken
    'mediapipe': {
        'n_samples_synced': int(len(pupil_data_synced)),
        'n_samples_assigned_to_trials': int(n_assigned),
        'assignment_rate': float(n_assigned / len(pupil_data_synced)),
        'n_fixation_samples': int(n_fixation),
        'n_stimulus_samples': int(n_stimulus),
        'n_unassigned_samples': int(n_unassigned),
    },
    
    # ptgaze-Statistiken (NEU v3.2 mit Blink-Merge)
    'ptgaze': {
        'available': ptgaze_available,
        'n_samples_synced': int(len(ptgaze_data_synced)) if ptgaze_available else 0,
        'n_samples_assigned_to_trials': int(n_assigned_ptgaze) if ptgaze_available else 0,
        'assignment_rate': float(n_assigned_ptgaze / len(ptgaze_data_synced)) if ptgaze_available and len(ptgaze_data_synced) > 0 else 0.0,
        'n_fixation_samples': int(n_fixation_ptgaze) if ptgaze_available else 0,
        'n_stimulus_samples': int(n_stimulus_ptgaze) if ptgaze_available else 0,
        'n_unassigned_samples': int(n_unassigned_ptgaze) if ptgaze_available else 0,
        'same_offset_as_mediapipe': True,
        'blink_source': 'mediapipe',
        'blink_merge_applied': ptgaze_available,
    },

    'n_trials': int(len(all_trials)),
    'csv_version': '3.3',  
    'validation': {
        'coverage_percent': float(coverage) if blocks_df is not None else None,
        'n_fixation_blocks': int(n_fixation_phases) if blocks_df is not None else None,
        'audio_validated': bool(validation)
    }
})

sync_json = os.path.join(OUTPUT_BASE_DIR, "debug_4_sync_info.json")
with open(sync_json, 'w') as f:
    json.dump(sync_info_export, f, indent=2)
print(f" Sync-Info: {sync_json}")

print(f"\n{'='*70}")
print(" ERFOLGREICH!")
print(f"{'='*70}")
print(f"\n Statistik:")
print(f"   • Samples: {len(pupil_data_synced)}")
print(f"   • Trial-Zuordnung: {n_assigned} ({n_assigned/len(pupil_data_synced)*100:.1f}%)")
print(f"   • Offset: {offset_used:.0f} ms")  # ← FIX: Nutze offset_used statt offset_ms!
print(f"   • Methode: {sync_method} (aus debug_0)")

if blocks_df is not None:
    print(f"   • Coverage: {coverage:.1f}%")

# ptgaze-Statistik (NEU v3.2 mit Blink-Merge)
if ptgaze_available:
    print(f"\n   [ptgaze Sync]")
    print(f"   - Samples: {len(ptgaze_data_synced)}")
    print(f"   - Trial-Zuordnung: {n_assigned_ptgaze} ({n_assigned_ptgaze/len(ptgaze_data_synced)*100:.1f}%)")
    print(f"   - Offset: {offset_used:.0f} ms (identisch zu MediaPipe)")
    print(f"   - Blink-Daten: von MediaPipe (stabiler)")
    print(f"   - Output: debug_4_ptgaze_gaze_synced.csv")
else:
    print(f"\n   [ptgaze Sync]")
    print(f"   - Nicht verfuegbar (debug_1_ptgaze_data.csv fehlt)")

#  NEU: Zeige Dual-Anchor-Info falls aktiv
if sync_method == 'dual_anchor' and 'offset_880hz_ms' in sync_info:
    print(f"\n   💡 Dual-Anchor Details:")
    print(f"      Kalibrierung (1760 Hz): {sync_info['offset_1760hz_ms']:.0f} ms")
    print(f"      Trials (880 Hz):        {sync_info['offset_880hz_ms']:.0f} ms")
    print(f"      Differenz:              {sync_info.get('frequency_difference_ms', 0):+.0f} ms")
    print(f"      >> Daten nutzen Trial-Offset für optimale Korrelation!")
