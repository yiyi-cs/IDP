# =============================================================================
# 03_RM_ANOVA.R - Repeated Measures ANOVA (Robustheitscheck)
# =============================================================================
# Validierungsstudie: CV Eye-Tracking vs. EyeLink 1000 Plus
#
# ZWECK:
#   Robustheitscheck der LMM-Ergebnisse mittels klassischer rm-ANOVA.
#
# DESIGN:
#   2 x 2 x 2 within-subject
#   - Method: MediaPipe vs. PTGaze
#   - Camera: 25Hz vs. 60Hz
#   - Calibration: FullCalib vs. BegFirst10EndFix
#
# AVs: bias_x, slope_x, r_z_x (nur X-Koordinate)
#
# OUTPUT:
#   - Table_5_rmANOVA.docx
#   - Figure_QQ_Plots_rmANOVA.png
#   - 03_rm_anova.RData
# =============================================================================

# =============================================================================
# 1. SETUP
# =============================================================================

library(tidyverse)
library(afex)
library(emmeans)
library(flextable)
library(officer)

# -----------------------------------------------------------------------------
# Daten laden
# -----------------------------------------------------------------------------
if (!exists("data_trial")) {
  load(file.path(config$output_base, "01_descriptives_and_regression", "01_trial_level.RData"))
  message("Daten aus 01_trial_level.RData geladen")
}

# Output-Verzeichnis
output_dir <- file.path(config$output_base, "03_rm_anova")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Primäre AVs
avs <- c("r_z_x", "slope_x", "bias_x")
av_labels <- c(r_z_x = "Korrelation (r_z)", slope_x = "Slope", bias_x = "Bias")

message("\n", paste(rep("=", 70), collapse = ""))
message("REPEATED MEASURES ANOVA (Robustheitscheck)")
message(paste(rep("=", 70), collapse = ""))

# =============================================================================
# 2. HILFSFUNKTIONEN
# =============================================================================

apa_table <- function(ft, note = NULL) {
  ft <- ft %>%
    font(fontname = "Times New Roman", part = "all") %>%
    fontsize(size = 12, part = "all") %>%
    align(align = "center", part = "all") %>%
    align(align = "left", j = 1, part = "body") %>%
    line_spacing(space = 2, part = "all") %>%
    border_remove() %>%
    hline_top(border = fp_border(width = 1), part = "header") %>%
    hline_bottom(border = fp_border(width = 1), part = "header") %>%
    hline_bottom(border = fp_border(width = 1), part = "body")
  
  if (!is.null(note)) {
    ft <- ft %>%
      add_footer_lines(paste("Anmerkung.", note)) %>%
      font(fontname = "Times New Roman", part = "footer") %>%
      fontsize(size = 10, part = "footer") %>%
      italic(part = "footer") %>%
      align(align = "left", part = "footer")
  }
  return(ft)
}

# =============================================================================
# 3. DATENAGGREGATION AUF VP-EBENE
# =============================================================================
# rm-ANOVA erfordert aggregierte Daten auf VP-Ebene (eine Zeile pro VP × Bedingung)

message("\n=== DATENAGGREGATION ===")

data_vp <- data_trial %>%
  group_by(vp_id, method, camera, calibration) %>%
  summarise(
    n_trials = n(),
    bias_x = mean(bias_x, na.rm = TRUE),
    slope_x = mean(slope_x, na.rm = TRUE),
    r_z_x = mean(r_z_x, na.rm = TRUE),
    .groups = "drop"
  )

n_vps <- n_distinct(data_vp$vp_id)
n_conditions <- 8  # 2x2x2

message("N VPs: ", n_vps)
message("Bedingungen pro VP: ", n_conditions)
message("Zeilen gesamt: ", nrow(data_vp), " (erwartet: ", n_vps * n_conditions, ")")

# Prüfe Vollständigkeit
completeness <- data_vp %>% count(vp_id) %>% filter(n != 8)
if (nrow(completeness) > 0) {
  warning("Unvollständige Daten für VPs: ", paste(completeness$vp_id, collapse = ", "))
  complete_vps <- data_vp %>% count(vp_id) %>% filter(n == 8) %>% pull(vp_id)
  data_vp <- data_vp %>% filter(vp_id %in% complete_vps)
  message("Nach Filterung: ", n_distinct(data_vp$vp_id), " VPs")
}

# Faktoren setzen
data_vp <- data_vp %>%
  mutate(
    vp_id = factor(vp_id),
    method = factor(method, levels = c("mediapipe", "ptgaze")),
    camera = factor(camera, levels = c("25hz", "60hz")),
    calibration = factor(calibration, levels = c("FullCalib", "BegFirst10EndFix"))
  )

# =============================================================================
# 4. VORAUSSETZUNGSPRÜFUNG
# =============================================================================

message("\n=== VORAUSSETZUNGSPRÜFUNG ===")

# -----------------------------------------------------------------------------
# 4.1 Ausreißerprüfung (z-Werte > 3.29)
# -----------------------------------------------------------------------------
message("\n--- Ausreißerprüfung ---")

outlier_check <- data_vp %>%
  group_by(method, camera, calibration) %>%
  mutate(bias_z = scale(bias_x), slope_z = scale(slope_x), r_z_z = scale(r_z_x)) %>%
  ungroup()

extreme_outliers <- outlier_check %>%
  filter(abs(bias_z) > 3.29 | abs(slope_z) > 3.29 | abs(r_z_z) > 3.29)

if (nrow(extreme_outliers) > 0) {
  message("[WARNUNG] Extreme Ausreißer gefunden (|z| > 3.29):")
  print(extreme_outliers %>% select(vp_id, method, camera, calibration, bias_z, slope_z, r_z_z))
} else {
  message("Keine extremen Ausreißer gefunden (|z| > 3.29)")
}

# -----------------------------------------------------------------------------
# 4.2 Normalverteilungsprüfung (Shapiro-Wilk pro Zelle)
# -----------------------------------------------------------------------------
message("\n--- Normalverteilungsprüfung (Shapiro-Wilk) ---")

normality_tests <- data_vp %>%
  group_by(method, camera, calibration) %>%
  summarise(
    bias_W = ifelse(n() >= 3, shapiro.test(bias_x)$statistic, NA),
    bias_p = ifelse(n() >= 3, shapiro.test(bias_x)$p.value, NA),
    slope_W = ifelse(n() >= 3, shapiro.test(slope_x)$statistic, NA),
    slope_p = ifelse(n() >= 3, shapiro.test(slope_x)$p.value, NA),
    r_z_W = ifelse(n() >= 3, shapiro.test(r_z_x)$statistic, NA),
    r_z_p = ifelse(n() >= 3, shapiro.test(r_z_x)$p.value, NA),
    n = n(),
    .groups = "drop"
  )

non_normal <- normality_tests %>% filter(bias_p < .05 | slope_p < .05 | r_z_p < .05)

if (nrow(non_normal) > 0) {
  message("[INFO] Zellen mit signifikanter Abweichung von Normalverteilung (p < .05):")
  print(non_normal %>% select(method, camera, calibration, bias_p, slope_p, r_z_p))
  message("Anmerkung: rm-ANOVA ist bei balancierten Designs robust gegenüber moderaten Verletzungen.")
} else {
  message("Keine signifikanten Abweichungen von der Normalverteilung (alle p > .05)")
}

write_csv(normality_tests, file.path(output_dir, "normality_tests.csv"))

# -----------------------------------------------------------------------------
# 4.3 QQ-Plots zur visuellen Inspektion der Normalverteilung
# -----------------------------------------------------------------------------
message("\n--- QQ-Plots ---")

# QQ-Plots für jede AV (über alle Bedingungen gepoolt)
png(file.path(output_dir, "Figure_QQ_Plots_rmANOVA.png"),
    width = 12, height = 4, units = "in", res = 300, bg = "white")

par(mfrow = c(1, 3), family = "serif", mar = c(4, 4, 3, 1))

# Bias
qqnorm(data_vp$bias_x, main = "QQ-Plot: Bias", 
       xlab = "Theoretische Quantile", ylab = "Beobachtete Quantile")
qqline(data_vp$bias_x, col = "red", lwd = 2)

# Slope
qqnorm(data_vp$slope_x, main = "QQ-Plot: Slope",
       xlab = "Theoretische Quantile", ylab = "Beobachtete Quantile")
qqline(data_vp$slope_x, col = "red", lwd = 2)

# Korrelation (Fisher-z)
qqnorm(data_vp$r_z_x, main = "QQ-Plot: Korrelation (r_z)",
       xlab = "Theoretische Quantile", ylab = "Beobachtete Quantile")
qqline(data_vp$r_z_x, col = "red", lwd = 2)

dev.off()

message("-> Figure_QQ_Plots_rmANOVA.png gespeichert")

# -----------------------------------------------------------------------------
# 4.4 Sphärizität
# -----------------------------------------------------------------------------
message("\n--- Sphärizität ---")
message("Bei 2-stufigen Faktoren automatisch erfüllt. Greenhouse-Geisser-Korrektur wird angewendet.")

# =============================================================================
# 5. RM-ANOVA MIT AFEX
# =============================================================================

message("\n=== RM-ANOVA ===")

run_anova <- function(dv, data) {
  message("\n--- ", dv, " ---")
  result <- afex::aov_ez(
    id = "vp_id", dv = dv, data = data,
    within = c("method", "camera", "calibration"),
    type = 3, anova_table = list(es = "pes", correction = "GG")
  )
  print(summary(result))
  return(result)
}

anova_results <- list()
for (av in avs) {
  anova_results[[av]] <- run_anova(av, data_vp)
}

# =============================================================================
# 6. TABELLE 5: RM-ANOVA ERGEBNISSE
# =============================================================================

message("\n=== TABELLE 5: RM-ANOVA ERGEBNISSE ===")

extract_anova <- function(av, result) {
  tbl <- as.data.frame(result$anova_table)
  tbl$Effect <- rownames(tbl)
  tbl$AV <- av
  tbl %>%
    select(AV, Effect, `num Df`, `den Df`, `F`, `Pr(>F)`, pes) %>%
    rename(df1 = `num Df`, df2 = `den Df`, F_value = `F`, p = `Pr(>F)`)
}

anova_summary <- bind_rows(lapply(avs, function(av) extract_anova(av, anova_results[[av]])))

# Formatieren
anova_summary <- anova_summary %>%
  mutate(
    Sig = case_when(p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", p < .10 ~ ".", TRUE ~ ""),
    p_fmt = ifelse(p < .001, "< .001", sprintf("%.3f", p)),
    pes_fmt = sprintf("%.3f", pes),
    Effekt_Typ = case_when(
      Effect %in% c("method", "camera", "calibration") ~ "Haupteffekt",
      str_count(Effect, ":") == 1 ~ "2-Wege",
      str_count(Effect, ":") == 2 ~ "3-Wege"
    )
  )

message("\n--- Alle Effekte ---")
print(anova_summary %>% select(AV, Effect, Effekt_Typ, F_value, df1, df2, p_fmt, Sig, pes_fmt))

# Haupteffekte separat
message("\n--- Haupteffekte ---")
haupteffekte <- anova_summary %>% filter(Effekt_Typ == "Haupteffekt")
print(haupteffekte %>% select(AV, Effect, F_value, df1, df2, p_fmt, Sig, pes_fmt))

# Signifikante Interaktionen
message("\n--- Signifikante Interaktionen (p < .05) ---")
sig_int <- anova_summary %>% filter(Effekt_Typ != "Haupteffekt", p < .05)
if (nrow(sig_int) > 0) {
  print(sig_int %>% select(AV, Effect, F_value, p_fmt, Sig, pes_fmt))
} else {
  message("Keine signifikanten Interaktionen")
}

# Word-Tabelle erstellen
table5_data <- anova_summary %>%
  mutate(
    AV_label = case_when(
      AV == "bias_x" ~ "Bias",
      AV == "slope_x" ~ "Slope",
      AV == "r_z_x" ~ "Korrelation"
    ),
    Effect_label = case_when(
      Effect == "method" ~ "Methode",
      Effect == "camera" ~ "Kamera",
      Effect == "calibration" ~ "Kalibrierung",
      Effect == "method:camera" ~ "Methode × Kamera",
      Effect == "method:calibration" ~ "Methode × Kalibrierung",
      Effect == "camera:calibration" ~ "Kamera × Kalibrierung",
      Effect == "method:camera:calibration" ~ "Methode × Kamera × Kalibrierung"
    ),
    df = sprintf("(%d, %.1f)", df1, df2),
    F_fmt = sprintf("%.2f", F_value)
  ) %>%
  select(AV_label, Effect_label, df, F_fmt, p_fmt, pes_fmt) %>%
  rename(AV = AV_label, Effekt = Effect_label, `df` = df, 
         `F` = F_fmt, `p` = p_fmt, `ηp²` = pes_fmt)

ft_table5 <- flextable(table5_data) %>%
  apa_table("Greenhouse-Geisser korrigierte Freiheitsgrade. ηp² = partielles Eta-Quadrat.") %>%
  autofit()

save_as_docx(ft_table5, path = file.path(output_dir, "Table_5_rmANOVA.docx"))
write_csv(anova_summary, file.path(output_dir, "anova_summary.csv"))

message("-> Tabelle 5 gespeichert")

# =============================================================================
# 7. POST-HOC TESTS
# =============================================================================

message("\n=== POST-HOC TESTS ===")

posthoc_results <- list()

for (av in avs) {
  message("\n--- ", av_labels[av], " ---")
  
  # EMMs berechnen
  emm_method <- emmeans(anova_results[[av]], ~ method)
  emm_camera <- emmeans(anova_results[[av]], ~ camera)
  emm_calib <- emmeans(anova_results[[av]], ~ calibration)
  
  # Paarvergleiche mit Bonferroni-Korrektur
  pairs_method <- pairs(emm_method, adjust = "bonferroni")
  pairs_camera <- pairs(emm_camera, adjust = "bonferroni")
  pairs_calib <- pairs(emm_calib, adjust = "bonferroni")
  
  message("\nMethode:")
  print(pairs_method)
  message("\nKamera:")
  print(pairs_camera)
  message("\nKalibrierung:")
  print(pairs_calib)
  
  posthoc_results[[av]] <- list(
    emm_method = emm_method, emm_camera = emm_camera, emm_calib = emm_calib,
    pairs_method = pairs_method, pairs_camera = pairs_camera, pairs_calib = pairs_calib
  )
}

# =============================================================================
# 8. ZUSAMMENFASSUNG
# =============================================================================

message("\n=== ZUSAMMENFASSUNG ===")

# Signifikante Effekte
sig_effects <- anova_summary %>%
  filter(p < .05) %>%
  select(AV, Effect, F_value, df1, df2, p_fmt, pes_fmt)

message("\n--- Signifikante Effekte (p < .05) ---")
print(sig_effects)

# Effektstärken interpretieren
message("\n--- Effektstärken (partielles Eta-Quadrat) ---")
message("Interpretation: klein >= .01, mittel >= .06, gross >= .14")

for (av in avs) {
  message("\n", av_labels[av], ":")
  av_effects <- anova_summary %>% 
    filter(AV == av, Effekt_Typ == "Haupteffekt") %>%
    mutate(
      interpretation = case_when(
        pes >= .14 ~ "gross",
        pes >= .06 ~ "mittel",
        pes >= .01 ~ "klein",
        TRUE ~ "negligible"
      )
    )
  for (i in 1:nrow(av_effects)) {
    message("  ", av_effects$Effect[i], ": ηp² = ", av_effects$pes_fmt[i], 
            " (", av_effects$interpretation[i], ")")
  }
}

# =============================================================================
# 9. SPEICHERN
# =============================================================================

message("\n=== SPEICHERN ===")

rm_anova_results <- list(
  data_vp = data_vp,
  anova_results = anova_results,
  anova_summary = anova_summary,
  posthoc_results = posthoc_results,
  avs = avs
)

save(rm_anova_results, file = file.path(output_dir, "03_rm_anova.RData"))

message(paste("Gespeichert:", file.path(output_dir, "03_rm_anova.RData")))

# =============================================================================
# 10. GENERIERTE DATEIEN
# =============================================================================

message("\n", paste(rep("=", 70), collapse = ""))
message("GENERIERTE DATEIEN")
message(paste(rep("=", 70), collapse = ""))

message("Tabellen:")
message("  - Table_5_rmANOVA.docx")
message("Abbildungen:")
message("  - Figure_QQ_Plots_rmANOVA.png")
message("CSV:")
message("  - anova_summary.csv")
message("  - normality_tests.csv")
message("Daten:")
message("  - 03_rm_anova.RData")

message("\n[OK] rm-ANOVA (Robustheitscheck) abgeschlossen.")
