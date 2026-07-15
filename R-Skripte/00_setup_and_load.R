# =============================================================================
# 00_SETUP_AND_LOAD.R
# =============================================================================
# Validierungsstudie: CV Eye-Tracking vs. EyeLink 1000 Plus
# 
# ZWECK:
#   Zentrale Datenaufbereitung für alle nachfolgenden Analyseskripte.
#   Lädt Rohdaten, definiert Faktoren/Kontraste, wendet Ausschlüsse an.
#
# OUTPUT:
#   - 00_prepared_data.RData (data_analysis, data_fig7, config, contrast_info)
#   - 00_exclusion_summary.csv
#
# DATENSTRUKTUR:
#   data_analysis : Bereinigte Frame-Level-Daten (16 Spalten)
#   data_fig7     : Minimaler Datensatz für Figure 7 (inkl. Blinks/Fixation)
#   config        : Analyse-Parameter
#   contrast_info : Kontrast-Dokumentation
# =============================================================================

# =============================================================================
# 1. PAKETE LADEN
# =============================================================================

message("\n", paste(rep("=", 70), collapse = ""))
message("PAKETE LADEN")
message(paste(rep("=", 70), collapse = ""))

load_packages <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      message(paste("Installiere Paket:", pkg))
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

required_packages <- c(
  "tidyverse",
  "lme4", "lmerTest",
  "emmeans", "effectsize",
  "performance", "see", "MuMIn",
  "TOSTER", "boot",
  "broom.mixed", "knitr", "kableExtra", "flextable", "officer",
  "patchwork", "cowplot", "ggnewscale",
  "moments", "zoo",
  "ez", "rstatix", "afex"
)

load_packages(required_packages)
message("Alle Pakete erfolgreich geladen.")

# =============================================================================
# 2. KONFIGURATION
# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ANPASSUNG FÜR FREMDEN PC: Nur die Pfade ändern!                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

message("\n", paste(rep("=", 70), collapse = ""))
message("KONFIGURATION")
message(paste(rep("=", 70), collapse = ""))

config <- list(
  # =========================================================================
  # PFADE - HIER ANPASSEN
  # =========================================================================
  data_dir = "C:.../Ergebnisse/Statistical_Analysis",
  output_base = "C:.../Ergebnisse/Statistical_Analysis/R_Results",
  
  # =========================================================================
  # DATEINAMEN
  # =========================================================================
  frame_level_file = "analysis_frame_level.csv",
  
  # =========================================================================
  # ANALYSE-PARAMETER
  # =========================================================================
  min_samples_per_trial = 10,
  allowed_calib_configs = c("BegFirst10EndFix", "FullCalib"),
  alpha = 0.05,
  alpha_random_effects = 0.20
)

# Output-Ordner erstellen
get_script_output_dir <- function(script_name) {
  dir_path <- file.path(config$output_base, script_name)
  if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)
  return(dir_path)
}

if (!dir.exists(config$output_base)) dir.create(config$output_base, recursive = TRUE)
config$output_dir <- get_script_output_dir("00_setup")

message("Datenverzeichnis: ", config$data_dir)
message("Ausgabeverzeichnis: ", config$output_base)

# =============================================================================
# 3. DATEN LADEN
# =============================================================================

message("\n", paste(rep("=", 70), collapse = ""))
message("DATEN LADEN")
message(paste(rep("=", 70), collapse = ""))

frame_level_path <- file.path(config$data_dir, config$frame_level_file)

if (file.exists(frame_level_path)) {
  data_raw <- read_csv(frame_level_path, show_col_types = FALSE)
  n_frames_total <- nrow(data_raw)
  message(paste("Frames geladen:", format(n_frames_total, big.mark = ",")))
} else {
  stop(paste("FEHLER: Datei nicht gefunden:", frame_level_path))
}

# =============================================================================
# 4. FAKTOREN UND KONTRASTE
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("FAKTOREN UND KONTRASTE")
message(paste(rep("-", 70), collapse = ""))

data_raw <- data_raw %>%
  mutate(
    vp_id = factor(vp_id),
    trial_id = factor(trial_assignment),
    stimulus_id = factor(stimulus_id),
    method = factor(method, levels = c("mediapipe", "ptgaze")),
    camera = factor(video_fps, levels = c("25hz", "60hz")),
    calibration = factor(calib_config, levels = c("FullCalib", "BegFirst10EndFix")),
    phase_type = factor(phase_type)
  )

contrasts(data_raw$method) <- contr.sum(2) / 2
contrasts(data_raw$camera) <- contr.sum(2) / 2
contrasts(data_raw$calibration) <- contr.sum(2) / 2

message("Sum-Kontraste gesetzt: (-0.5, +0.5)")

# Kontrast-Dokumentation
contrast_info <- list(
  method = list(coding = "Sum (-0.5, +0.5)", levels = "mediapipe = -0.5, ptgaze = +0.5"),
  camera = list(coding = "Sum (-0.5, +0.5)", levels = "25hz = -0.5, 60hz = +0.5"),
  calibration = list(coding = "Sum (-0.5, +0.5)", levels = "FullCalib = -0.5, BegFirst10EndFix = +0.5")
)

# =============================================================================
# 5. KALIBRIERUNGSFILTER
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("KALIBRIERUNGSFILTER (nur K1 und K4)")
message(paste(rep("-", 70), collapse = ""))

data_filtered <- data_raw %>%
  filter(calib_config %in% config$allowed_calib_configs)

n_after_calib <- nrow(data_filtered)
message(paste("Frames nach Kalibrierungsfilter:", format(n_after_calib, big.mark = ",")))

contrasts(data_filtered$calibration) <- contr.sum(2) / 2

# =============================================================================
# 6. PHASEN-STATISTIK (für Arbeit)
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("PHASEN-STATISTIK")
message(paste(rep("-", 70), collapse = ""))

n_stimulus <- sum(data_filtered$phase_type == "stimulus", na.rm = TRUE)
n_fixation <- sum(data_filtered$phase_type == "fixation", na.rm = TRUE)

message(paste("Frames gesamt (nach Kalib-Filter):", format(n_after_calib, big.mark = ",")))
message(paste("  - Stimulus-Phase:", format(n_stimulus, big.mark = ",")))
message(paste("  - Fixations-Phase:", format(n_fixation, big.mark = ",")))

# =============================================================================
# 7. BLINK-STATISTIK KOMBINIERT (Stimulus + Fixation)
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("BLINK-STATISTIK KOMBINIERT (Stimulus + Fixation)")
message(paste(rep("-", 70), collapse = ""))

n_blink_combined <- sum(data_filtered$exclude_blink, na.rm = TRUE)
n_outside_combined <- sum(data_filtered$exclude_outside, na.rm = TRUE)
n_valid_combined <- sum(!data_filtered$exclude_any, na.rm = TRUE)

pct_blink_combined <- 100 * n_blink_combined / n_after_calib
pct_outside_combined <- 100 * n_outside_combined / n_after_calib

message(paste("Frames gesamt (Stimulus + Fixation):", format(n_after_calib, big.mark = ",")))
message(paste("AUSSCHLUSS Blink (kombiniert):", format(n_blink_combined, big.mark = ","), 
              sprintf("(%.1f%%)", pct_blink_combined)))
message(paste("AUSSCHLUSS Outside (kombiniert):", format(n_outside_combined, big.mark = ","), 
              sprintf("(%.1f%%)", pct_outside_combined)))
message(paste("Valide Frames (kombiniert):", format(n_valid_combined, big.mark = ","),
              sprintf("(%.1f%%)", 100 * n_valid_combined / n_after_calib)))

# =============================================================================
# 8. STIMULUS-PHASE EXTRAHIEREN
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("STIMULUS-PHASE EXTRAHIEREN")
message(paste(rep("-", 70), collapse = ""))

data_stimulus <- data_filtered %>% 
  filter(phase_type == "stimulus")

message(paste("Stimulus-Frames:", format(nrow(data_stimulus), big.mark = ",")))

# =============================================================================
# 9. AUSSCHLÜSSE: BLINKS UND OUTSIDE (nur Stimulus-Phase)
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("AUSSCHLÜSSE: Blinks und Outside Monitor (nur Stimulus-Phase)")
message(paste(rep("-", 70), collapse = ""))

n_total_stimulus <- nrow(data_stimulus)
n_blink <- sum(data_stimulus$exclude_blink, na.rm = TRUE)
n_outside <- sum(data_stimulus$exclude_outside, na.rm = TRUE)
n_excluded <- sum(data_stimulus$exclude_any, na.rm = TRUE)

message(paste("Stimulus-Frames gesamt:", format(n_total_stimulus, big.mark = ",")))
message(paste("AUSSCHLUSS Blink (Stimulus):", format(n_blink, big.mark = ","), 
              sprintf("(%.1f%%)", 100 * n_blink / n_total_stimulus)))
message(paste("AUSSCHLUSS Outside (Stimulus):", format(n_outside, big.mark = ","), 
              sprintf("(%.1f%%)", 100 * n_outside / n_total_stimulus)))

# =============================================================================
# 10. FINALER DATENSATZ (data_analysis)
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("FINALER DATENSATZ")
message(paste(rep("-", 70), collapse = ""))

# Ausschlüsse anwenden
data_analysis_full <- data_stimulus %>%
  filter(!exclude_any)

n_final <- nrow(data_analysis_full)
pct_retained <- 100 * n_final / n_total_stimulus

message(paste("Finale Frames für Analyse:", format(n_final, big.mark = ","),
              sprintf("(%.1f%% der Stimulus-Phase)", pct_retained)))

# =============================================================================
# 11. SPALTENBEREINIGUNG (data_analysis)
# =============================================================================
# Entfernt redundante Spalten für übersichtlichen Analysedatensatz.
# Rohdaten bleiben in analysis_frame_level.csv erhalten.

message("\n", paste(rep("-", 70), collapse = ""))
message("SPALTENBEREINIGUNG")
message(paste(rep("-", 70), collapse = ""))

cols_analysis <- c(
  # Identifikatoren
  "vp_id", "trial_id", "stimulus_id", "frame", "timestamp_ms_synced",
  # Faktoren (keine Duplikate wie video_fps, calib_config)
  "method", "camera", "calibration",
  # Blickdaten
  "cv_deg_x", "cv_deg_y", "eyelink_deg_x", "eyelink_deg_y",
  # Ausschluss-Flags
  "is_blink", "outside_monitor", "exclude_any",
  # Phase
  "phase_type"
)

n_cols_before <- ncol(data_analysis_full)
data_analysis <- data_analysis_full %>% select(all_of(cols_analysis))
n_cols_after <- ncol(data_analysis)

message(paste("Spalten reduziert:", n_cols_before, "→", n_cols_after))

# =============================================================================
# 12. DATENSATZ FÜR FIGURE 7 (data_fig7)
# =============================================================================
# Enthält alle Phasen und Blinks für Visualisierung der Zeitreihen.

message("\n", paste(rep("-", 70), collapse = ""))
message("DATENSATZ FÜR FIGURE 7")
message(paste(rep("-", 70), collapse = ""))

cols_fig7 <- c(
  "vp_id", "trial_id", "frame", "timestamp_ms_synced",
  "method", "camera", "calibration",
  "cv_deg_x", "eyelink_deg_x",
  "is_blink", "phase_type"
)

data_fig7 <- data_filtered %>% select(all_of(cols_fig7))

message(paste("data_fig7 erstellt:", format(nrow(data_fig7), big.mark = ","), "Frames,", ncol(data_fig7), "Spalten"))
message("[INFO] Enthält alle Phasen und Blinks für Figure 7")

# =============================================================================
# 13. SPEICHERN
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("SPEICHERN")
message(paste(rep("-", 70), collapse = ""))

save(
  data_analysis,
  data_fig7,
  config,
  contrast_info,
  file = file.path(config$output_dir, "00_prepared_data.RData")
)

message(paste("Gespeichert:", file.path(config$output_dir, "00_prepared_data.RData")))
message("  - data_analysis: Bereinigte Analysedaten (Frame-Level)")
message("  - data_fig7: Daten für Figure 7 (alle Phasen, inkl. Blinks)")
message("  - config: Analyse-Parameter")
message("  - contrast_info: Kontrast-Dokumentation")

# =============================================================================
# 14. ZUSAMMENFASSUNG
# =============================================================================

message("\n", paste(rep("=", 70), collapse = ""))
message("ZUSAMMENFASSUNG DATENAUFBEREITUNG")
message(paste(rep("=", 70), collapse = ""))

summary_stats <- tibble(
  Metrik = c(
    "Frames gesamt (alle Kalibrierungen)",
    "Frames nach Kalibrierungsfilter (K1+K4)",
    "  - davon Stimulus-Phase",
    "  - davon Fixations-Phase",
    "--- KOMBINIERT (Stimulus + Fixation) ---",
    "AUSSCHLUSS: Blinks (kombiniert)",
    "AUSSCHLUSS: Outside Monitor (kombiniert)",
    "--- NUR STIMULUS-PHASE ---",
    "AUSSCHLUSS: Blinks (Stimulus)",
    "AUSSCHLUSS: Outside Monitor (Stimulus)",
    "Frames für Analyse (Stimulus, valide)",
    "Anteil behalten (von Stimulus)",
    "VPs",
    "Trials pro VP×Methode×Kamera×Kalib"
  ),
  Wert = c(
    format(n_frames_total, big.mark = ","),
    format(n_after_calib, big.mark = ","),
    format(n_stimulus, big.mark = ","),
    format(n_fixation, big.mark = ","),
    "---",
    sprintf("%s (%.1f%%)", format(n_blink_combined, big.mark = ","), pct_blink_combined),
    sprintf("%s (%.1f%%)", format(n_outside_combined, big.mark = ","), pct_outside_combined),
    "---",
    sprintf("%s (%.1f%%)", format(n_blink, big.mark = ","), 100 * n_blink / n_total_stimulus),
    sprintf("%s (%.1f%%)", format(n_outside, big.mark = ","), 100 * n_outside / n_total_stimulus),
    format(n_final, big.mark = ","),
    sprintf("%.1f%%", pct_retained),
    as.character(n_distinct(data_analysis$vp_id)),
    "24"
  )
)

print(kable(summary_stats, format = "simple"))
write_csv(summary_stats, file.path(config$output_dir, "00_exclusion_summary.csv"))

# =============================================================================
# 15. CLEANUP: Temporäre Datensätze entfernen
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("CLEANUP")
message(paste(rep("-", 70), collapse = ""))

# Temporäre Datensätze (data.frames/tibbles)
temp_data <- c("data_raw", "data_filtered", "data_stimulus", "data_analysis_full", "summary_stats")

existing_temp <- temp_data[temp_data %in% ls()]
if (length(existing_temp) > 0) {
  rm(list = existing_temp)
  message(paste("Temporäre Datensätze entfernt:", paste(existing_temp, collapse = ", ")))
}

message("Finale Datensätze: data_analysis, data_fig7, config, contrast_info")

message("\n[OK] Setup abgeschlossen.")
message("Weiter mit: 01_descriptives_and_regression.R")
