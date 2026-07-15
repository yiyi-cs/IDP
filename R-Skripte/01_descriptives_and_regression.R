# =============================================================================
# 01_DESCRIPTIVES_AND_REGRESSION.R
# =============================================================================
# Validierungsstudie: CV Eye-Tracking vs. EyeLink 1000 Plus
#
# ZWECK:
#   Berechnung der Trial-Level Validierungsmetriken und deskriptive Statistiken.
#
# OUTPUT:
#   - Table_1_Deskriptive.docx
#   - Figure_2_AV_Illustration.png/.pdf
#   - Figure_4_GlobalMetrics.png/.pdf
#   - Figure_7_VP_Trials.png/.pdf
#   - 01_trial_level.RData (data_trial, data_vp)
#
# DATENSTRUKTUR:
#   data_trial : Trial-Level Metriken (valide Trials, ≥10 Samples)
#   data_vp    : VP-Level aggregierte Metriken pro Bedingung
# =============================================================================

# =============================================================================
# 1. SETUP
# =============================================================================

library(tidyverse)
library(flextable)
library(officer)
library(patchwork)
library(cowplot)
library(zoo)
library(ggnewscale)

# Daten laden
if (!exists("data_analysis")) {
  load(file.path(config$output_base, "00_setup", "00_prepared_data.RData"))
  message("Daten aus 00_prepared_data.RData geladen")
}

output_dir <- file.path(config$output_base, "01_descriptives_and_regression")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

message("\n", paste(rep("=", 70), collapse = ""))
message("DESKRIPTIVE STATISTIK UND TRIAL-LEVEL REGRESSION")
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

theme_apa_fig <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(family = "sans"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
      axis.title = element_text(face = "bold", size = base_size),
      axis.title.y = element_text(margin = margin(r = 5)),
      axis.title.x = element_text(margin = margin(t = 5)),
      axis.text = element_text(color = "black", size = base_size - 1),
      axis.line = element_line(color = "black", linewidth = 0.4),
      legend.position = "none",
      strip.text = element_text(face = "bold", size = base_size),
      strip.background = element_blank(),
      plot.title = element_text(face = "bold", size = base_size + 1, hjust = 0.5),
      plot.margin = margin(5, 10, 5, 5)
    )
}

# VP-Mapping
vp_mapping <- c(
  "beo7" = "VP 1",  "bjs4" = "VP 2",  "egf5" = "VP 3",  "fbn6" = "VP 4",
  "fgt6" = "VP 5",  "jkl7" = "VP 6",  "kdn8" = "VP 7",  "kro3" = "VP 8",
  "ldj9" = "VP 9",  "mhe9" = "VP 10", "oem4" = "VP 11", "ogt7" = "VP 12"
)
vp_ids_original <- sort(unique(as.character(data_analysis$vp_id)))
vp_labels <- vp_mapping[vp_ids_original]
vp_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
               "#A65628", "#F781BF", "#1B9E77", "#D95F02", "#7570B3", "#E7298A")
names(vp_colors) <- vp_ids_original
vp_shapes <- c(16, 16, 16, 16, 17, 17, 17, 17, 15, 15, 15, 15)
names(vp_shapes) <- vp_ids_original

# =============================================================================
# 3. TRIAL-LEVEL METRIKEN BERECHNEN
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("TRIAL-LEVEL METRIKEN")
message(paste(rep("-", 70), collapse = ""))

compute_trial_metrics <- function(cv_x, eyelink_x, min_samples = 10) {
  valid_x <- !is.na(cv_x) & !is.na(eyelink_x)
  n_valid_x <- sum(valid_x)
  
  result <- c(
    n_samples_total = length(cv_x),
    n_samples_valid = n_valid_x,
    bias_x = NA_real_, slope_x = NA_real_,
    r_x = NA_real_, r_z_x = NA_real_,
    mae_x = NA_real_, cv_M_trial = NA_real_,
    valid_trial = FALSE
  )
  
  if (n_valid_x >= min_samples) {
    cv_x_valid <- cv_x[valid_x]
    el_x_valid <- eyelink_x[valid_x]
    model_x <- lm(cv_x_valid ~ el_x_valid)
    result["bias_x"] <- coef(model_x)[1]
    result["slope_x"] <- coef(model_x)[2]
    r_x <- cor(cv_x_valid, el_x_valid)
    result["r_x"] <- r_x
    r_x_bounded <- max(min(r_x, 0.999), -0.999)
    result["r_z_x"] <- atanh(r_x_bounded)
    result["mae_x"] <- mean(abs(cv_x_valid - el_x_valid), na.rm = TRUE)
    result["cv_M_trial"] <- mean(cv_x_valid, na.rm = TRUE)
    result["valid_trial"] <- TRUE
  }
  return(result)
}

trial_groups <- data_analysis %>%
  group_by(vp_id, trial_id, method, camera, calibration, stimulus_id) %>%
  group_keys()

trial_metrics_list <- data_analysis %>%
  group_by(vp_id, trial_id, method, camera, calibration, stimulus_id) %>%
  group_map(~ {
    metrics <- compute_trial_metrics(.x$cv_deg_x, .x$eyelink_deg_x, config$min_samples_per_trial)
    return(as_tibble_row(metrics))
  }, .keep = TRUE)

data_trial_all <- bind_cols(trial_groups, bind_rows(trial_metrics_list)) %>%
  mutate(valid_trial = as.logical(valid_trial))

# Faktoren und Kontraste
data_trial_all <- data_trial_all %>%
  mutate(
    vp_id = factor(vp_id),
    trial_id = factor(trial_id),
    stimulus_id = factor(stimulus_id),
    method = factor(method, levels = c("mediapipe", "ptgaze")),
    camera = factor(camera, levels = c("25hz", "60hz")),
    calibration = factor(calibration, levels = c("FullCalib", "BegFirst10EndFix"))
  )
contrasts(data_trial_all$method) <- contr.sum(2) / 2
contrasts(data_trial_all$camera) <- contr.sum(2) / 2
contrasts(data_trial_all$calibration) <- contr.sum(2) / 2

# Nur valide Trials → data_trial
data_trial <- data_trial_all %>% filter(valid_trial)

message(paste("Trials berechnet:", nrow(data_trial_all)))
message(paste("Valide Trials (data_trial):", nrow(data_trial)))

# =============================================================================
# 4. AGGREGATION: Trial → VP (data_vp)
# =============================================================================

data_vp <- data_trial %>%
  group_by(vp_id, method, camera, calibration) %>%
  summarise(
    cv_M_vp = mean(cv_M_trial, na.rm = TRUE),  # <-- ERGÄNZEN
    r_vp = mean(r_x, na.rm = TRUE),
    slope_vp = mean(slope_x, na.rm = TRUE),
    bias_vp = mean(bias_x, na.rm = TRUE),
    mae_vp = mean(mae_x, na.rm = TRUE),
    n_trials = n(),
    .groups = "drop"
  )

# Globale Statistiken (für Tabelle 1)
global_stats <- data_vp %>%
  group_by(method, camera, calibration) %>%
  summarise(
    cv_M = mean(cv_M_vp, na.rm = TRUE),   # <-- ERGÄNZEN
    cv_SD = sd(cv_M_vp, na.rm = TRUE),    # <-- ERGÄNZEN
    r_M = mean(r_vp, na.rm = TRUE), r_SD = sd(r_vp, na.rm = TRUE),
    slope_M = mean(slope_vp, na.rm = TRUE), slope_SD = sd(slope_vp, na.rm = TRUE),
    bias_M = mean(bias_vp, na.rm = TRUE), bias_SD = sd(bias_vp, na.rm = TRUE),
    mae_M = mean(mae_vp, na.rm = TRUE), mae_SD = sd(mae_vp, na.rm = TRUE),
    n_vps = n(),
    .groups = "drop"
  )

# =============================================================================
# 5. TABELLE 1: DESKRIPTIVE STATISTIK
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("TABELLE 1")
message(paste(rep("-", 70), collapse = ""))

table1_data <- global_stats %>%
  mutate(
    Methode = ifelse(method == "mediapipe", "MediaPipe", "PTGaze"),
    Kamera = ifelse(camera == "25hz", "25 Hz", "60 Hz"),
    Kalibrierung = ifelse(calibration == "FullCalib", "Voll", "Reduziert"),
    `X_CV (SD)` = sprintf("%.2f (%.2f)", cv_M, cv_SD),  # <-- ERGÄNZEN
    `r (SD)` = sprintf("%.2f (%.2f)", r_M, r_SD),
    `Steigung (SD)` = sprintf("%.2f (%.2f)", slope_M, slope_SD),
    `Bias (SD)` = sprintf("%.2f (%.2f)", bias_M, bias_SD),
    `MAE (SD)` = sprintf("%.2f (%.2f)", mae_M, mae_SD)
  ) %>%
  select(Methode, Kamera, Kalibrierung, `X_CV (SD)`, `r (SD)`, 
         `Steigung (SD)`, `Bias (SD)`, `MAE (SD)`)


ft_table1 <- flextable(table1_data) %>%
  apa_table("Deskriptive Statistiken (Trial → VP → Global). r = Korrelation; Steigung = Slope; Bias = Intercept; MAE = Mean Absolute Error. Alle Werte in ° Sehwinkel (außer r). N = 12 pro Bedingung.") %>%
  autofit()

save_as_docx(ft_table1, path = file.path(output_dir, "Table_1_Deskriptive.docx"))
message("-> Tabelle 1 gespeichert")

# =============================================================================
# 6. ABBILDUNG 2: AV-ILLUSTRATION (Korrelation, Steigung, Bias erklärt)
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("ABBILDUNG 2: AV-ILLUSTRATION")
message(paste(rep("-", 70), collapse = ""))

# Simulierte Beispieldaten für Illustration
set.seed(456)  # Geänderter Seed für niedrigere Korrelation

# Erzeuge realistische Blickdaten (in Grad Sehwinkel)
n_points <- 60
eyelink_base <- seq(-10, 10, length.out = n_points)
eyelink_x <- eyelink_base + rnorm(n_points, 0, 0.2)

# ============================================================================
# EINHEITLICHE PARAMETER FÜR ALLE PANELS
# ============================================================================
arrow_linewidth <- 0.3
arrow_size <- unit(0.07, "cm")
text_size <- 2.2

# --- Panel A: Korrelation ---
cv_ideal_corr <- eyelink_x
noise_corr <- rnorm(n_points, 0, 4.0)
cv_observed_corr <- eyelink_x + noise_corr
r_observed <- round(cor(eyelink_x, cv_observed_corr), 2)

# Labels und Farben definieren
label_ideal_corr <- "Ideal (r = 1)"
label_observed_corr <- paste0("Beobachtet (r = ", r_observed, ")")
colors_corr <- setNames(c("#4DAF4A", "#E41A1C"), c(label_ideal_corr, label_observed_corr))

data_corr <- tibble(
  EyeLink = rep(eyelink_x, 2),
  CV = c(cv_ideal_corr, cv_observed_corr),
  Typ = factor(rep(c(label_ideal_corr, label_observed_corr), each = n_points),
               levels = c(label_ideal_corr, label_observed_corr))
)

# Finde einen Punkt mit großer POSITIVER Abweichung im LINKEN Bereich
left_indices <- which(eyelink_x > -8 & eyelink_x < -4)
positive_deviations <- noise_corr[left_indices]
best_idx <- left_indices[which.max(positive_deviations)]

annot_x_corr <- eyelink_x[best_idx]
annot_y_ideal_corr <- cv_ideal_corr[best_idx]
annot_y_observed_corr <- cv_observed_corr[best_idx]

p_corr_illustration <- ggplot(data_corr, aes(x = EyeLink, y = CV, color = Typ)) +
  geom_point(alpha = 0.7, size = 1.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 0.4) +
  scale_color_manual(values = colors_corr, name = NULL) +
  annotate("segment", 
           x = annot_x_corr, xend = annot_x_corr, 
           y = annot_y_ideal_corr, yend = annot_y_observed_corr,
           arrow = arrow(length = arrow_size, ends = "both", type = "closed"),
           color = "black", linewidth = arrow_linewidth) +
  annotate("text", x = annot_x_corr + 4.5, y = (annot_y_ideal_corr + annot_y_observed_corr) / 2, 
           label = "Rauschen", color = "black", size = text_size, fontface = "bold", hjust = 0) +
  labs(title = "Korrelation", x = "EyeLink (°)", y = "CV (°)") +
  coord_fixed(ratio = 1, xlim = c(-12, 12), ylim = c(-12, 12)) +
  theme_apa_fig(base_size = 9) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.25, "cm"),
        legend.key.spacing.x = unit(0.05, "cm"),
        legend.margin = margin(-2, 0, 0, 0),
        legend.box.margin = margin(-5, 0, 0, 0),
        plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
        axis.title.x = element_text(size = 8),
        axis.title.y = element_text(size = 8),
        plot.margin = margin(5, 2, 2, 2))

# --- Panel B: Steigung ---
slope_observed <- 0.6
cv_ideal_slope <- eyelink_x
cv_observed_slope <- slope_observed * eyelink_x + rnorm(n_points, 0, 0.5)

label_ideal_slope <- "Ideal (Slope = 1)"
label_observed_slope <- paste0("Beobachtet (Slope = ", slope_observed, ")")
colors_slope <- setNames(c("#4DAF4A", "#E41A1C"), c(label_ideal_slope, label_observed_slope))

data_slope <- tibble(
  EyeLink = rep(eyelink_x, 2),
  CV = c(cv_ideal_slope, cv_observed_slope),
  Typ = factor(rep(c(label_ideal_slope, label_observed_slope), each = n_points),
               levels = c(label_ideal_slope, label_observed_slope))
)

annot_x_slope <- 8
annot_y_ideal_slope <- annot_x_slope * 1
annot_y_observed_slope <- annot_x_slope * slope_observed

p_slope_illustration <- ggplot(data_slope, aes(x = EyeLink, y = CV, color = Typ)) +
  geom_point(alpha = 0.7, size = 1.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#4DAF4A", linewidth = 0.4) +
  geom_abline(intercept = 0, slope = slope_observed, linetype = "solid", color = "#E41A1C", linewidth = 0.4) +
  scale_color_manual(values = colors_slope, name = NULL) +
  annotate("segment", 
           x = annot_x_slope, xend = annot_x_slope, 
           y = annot_y_observed_slope, yend = annot_y_ideal_slope,
           arrow = arrow(length = arrow_size, ends = "both", type = "closed"),
           color = "black", linewidth = arrow_linewidth) +
  annotate("text", x = annot_x_slope - 5.5, y = (annot_y_ideal_slope + annot_y_observed_slope) / 2, 
           label = "Kompression", color = "black", size = text_size, fontface = "bold", hjust = 1) +
  labs(title = "Steigung", x = "EyeLink (°)", y = NULL) +
  coord_fixed(ratio = 1, xlim = c(-12, 12), ylim = c(-12, 12)) +
  theme_apa_fig(base_size = 9) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.25, "cm"),
        legend.key.spacing.x = unit(0.05, "cm"),
        legend.margin = margin(-2, 0, 0, 0),
        legend.box.margin = margin(-5, 0, 0, 0),
        plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
        axis.title.x = element_text(size = 8),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        plot.margin = margin(5, 2, 2, 2))

# --- Panel C: Bias ---
bias_observed <- 3
cv_ideal_bias <- eyelink_x
cv_observed_bias <- eyelink_x + bias_observed + rnorm(n_points, 0, 0.5)

label_ideal_bias <- "Ideal (Bias = 0°)"
label_observed_bias <- paste0("Beobachtet (Bias = ", bias_observed, "°)")
colors_bias <- setNames(c("#4DAF4A", "#E41A1C"), c(label_ideal_bias, label_observed_bias))

data_bias <- tibble(
  EyeLink = rep(eyelink_x, 2),
  CV = c(cv_ideal_bias, cv_observed_bias),
  Typ = factor(rep(c(label_ideal_bias, label_observed_bias), each = n_points),
               levels = c(label_ideal_bias, label_observed_bias))
)

annot_x_bias <- 0
annot_y_ideal_bias <- 0
annot_y_observed_bias <- bias_observed

p_bias_illustration <- ggplot(data_bias, aes(x = EyeLink, y = CV, color = Typ)) +
  geom_point(alpha = 0.7, size = 1.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#4DAF4A", linewidth = 0.4) +
  geom_abline(intercept = bias_observed, slope = 1, linetype = "solid", color = "#E41A1C", linewidth = 0.4) +
  scale_color_manual(values = colors_bias, name = NULL) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray50", linewidth = 0.3) +
  annotate("segment", 
           x = annot_x_bias, xend = annot_x_bias, 
           y = annot_y_ideal_bias, yend = annot_y_observed_bias,
           arrow = arrow(length = arrow_size, ends = "both", type = "closed"),
           color = "black", linewidth = arrow_linewidth) +
  annotate("text", x = annot_x_bias + 4.5, y = (annot_y_ideal_bias + annot_y_observed_bias) / 2, 
           label = "Offset", color = "black", size = text_size, fontface = "bold", hjust = 0) +
  labs(title = "Bias", x = "EyeLink (°)", y = NULL) +
  coord_fixed(ratio = 1, xlim = c(-12, 12), ylim = c(-12, 12)) +
  theme_apa_fig(base_size = 9) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 6),
        legend.key.size = unit(0.25, "cm"),
        legend.key.spacing.x = unit(0.05, "cm"),
        legend.margin = margin(-2, 0, 0, 0),
        legend.box.margin = margin(-5, 0, 0, 0),
        plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
        axis.title.x = element_text(size = 8),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        plot.margin = margin(5, 2, 2, 2))

# --- Kombinieren ---
fig_2 <- (p_corr_illustration | p_slope_illustration | p_bias_illustration) +
  plot_layout(widths = c(1, 1, 1))

ggsave(file.path(output_dir, "Figure_2_AV_Illustration.png"),
       fig_2, width = 180, height = 70, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(output_dir, "Figure_2_AV_Illustration.pdf"),
       fig_2, width = 180, height = 70, units = "mm", bg = "white")

message("-> Abbildung 2 gespeichert")
message(paste("   Korrelation im Beispiel: r =", r_observed))

# =============================================================================
# 7. ABBILDUNG 4: GLOBALE METRIKEN
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("ABBILDUNG 4: GLOBALE METRIKEN")
message(paste(rep("-", 70), collapse = ""))

vp_global <- data_vp %>%
  group_by(vp_id) %>%
  summarise(r = mean(r_vp), slope = mean(slope_vp), bias = mean(bias_vp), .groups = "drop") %>%
  mutate(vp_id = factor(vp_id, levels = vp_ids_original))

global_summary <- vp_global %>%
  summarise(
    r_M = mean(r), r_SE = sd(r) / sqrt(n()), r_lower = r_M - 1.96 * r_SE, r_upper = r_M + 1.96 * r_SE,
    slope_M = mean(slope), slope_SE = sd(slope) / sqrt(n()), slope_lower = slope_M - 1.96 * slope_SE, slope_upper = slope_M + 1.96 * slope_SE,
    bias_M = mean(bias), bias_SE = sd(bias) / sqrt(n()), bias_lower = bias_M - 1.96 * bias_SE, bias_upper = bias_M + 1.96 * bias_SE
  )

plot_global_panel <- function(vp_data, y_var, y_label, ref_line, y_limits, title) {
  y_mean <- global_summary[[paste0(y_var, "_M")]]
  y_lower <- global_summary[[paste0(y_var, "_lower")]]
  y_upper <- global_summary[[paste0(y_var, "_upper")]]
  ggplot() +
    geom_point(data = vp_data, aes(x = 1, y = .data[[y_var]], color = vp_id, shape = vp_id),
               position = position_jitter(width = 0.15, seed = 42), size = 2.5, alpha = 0.9) +
    geom_pointrange(aes(x = 1.4, y = y_mean, ymin = y_lower, ymax = y_upper), color = "#D62728", size = 0.5, linewidth = 0.7) +
    geom_hline(yintercept = ref_line, linetype = "dashed", color = "gray40", linewidth = 0.5) +
    scale_x_continuous(limits = c(0.5, 1.8), breaks = NULL) +
    scale_color_manual(values = vp_colors, labels = vp_labels, drop = FALSE) +
    scale_shape_manual(values = vp_shapes, labels = vp_labels, drop = FALSE) +
    labs(x = NULL, y = y_label, title = title) +
    coord_cartesian(ylim = y_limits) +
    theme_apa_fig()
}

p4_r <- plot_global_panel(vp_global, "r", "Korrelation (r)", 1, c(0.35, 1.1), "Korrelation")
p4_slope <- plot_global_panel(vp_global, "slope", "Steigung", 1, c(0.35, 1.1), "Steigung")
p4_bias <- plot_global_panel(vp_global, "bias", "Bias (°)", 0, c(-2.5, 2.5), "Bias")

legend_data <- data.frame(vp_id = factor(vp_ids_original, levels = vp_ids_original), x = 1, y = 1)
p_legend <- ggplot(legend_data, aes(x = x, y = y, color = vp_id, shape = vp_id)) +
  geom_point(size = 2.5) +
  scale_color_manual(values = vp_colors, name = NULL, labels = vp_labels) +
  scale_shape_manual(values = vp_shapes, name = NULL, labels = vp_labels) +
  theme_void() +
  theme(legend.position = "bottom", legend.text = element_text(size = 8)) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE), shape = guide_legend(nrow = 2, byrow = TRUE))
legend_grob <- cowplot::get_legend(p_legend)

fig_4 <- ((p4_r | p4_slope | p4_bias) / wrap_elements(full = legend_grob)) + plot_layout(heights = c(1, 0.15))
ggsave(file.path(output_dir, "Figure_4_GlobalMetrics.png"), fig_4, width = 180, height = 100, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(output_dir, "Figure_4_GlobalMetrics.pdf"), fig_4, width = 180, height = 100, units = "mm", bg = "white")
message("-> Abbildung 4 gespeichert")

# =============================================================================
# 8. ABBILDUNG 7: VP-TRIAL-ZEITREIHEN
# =============================================================================

message("\n", paste(rep("-", 70), collapse = ""))
message("ABBILDUNG 7: VP-TRIAL-ZEITREIHEN")
message(paste(rep("-", 70), collapse = ""))

# Nutze data_fig7 (enthält alle Phasen und Blinks)
if (exists("data_fig7")) {
  message("Nutze data_fig7 für Figure 7")
  
  all_trial_specs <- tribble(
    ~vp_id,  ~trial_nr, ~camera, ~calibration,
    "beo7",  1,         "25hz",  "FullCalib",
    "bjs4",  24,        "25hz",  "BegFirst10EndFix",
    "egf5",  11,        "25hz",  "FullCalib",
    "fbn6",  1,         "60hz",  "BegFirst10EndFix",
    "fgt6",  13,        "25hz",  "FullCalib",
    "jkl7",  1,         "25hz",  "BegFirst10EndFix",
    "kdn8",  13,        "25hz",  "BegFirst10EndFix",
    "kro3",  20,        "25hz",  "FullCalib",
    "ldj9",  8,         "60hz",  "FullCalib",
    "mhe9",  21,        "25hz",  "FullCalib",
    "oem4",  17,        "60hz",  "BegFirst10EndFix",
    "ogt7",  1,         "25hz",  "FullCalib"
  ) %>% mutate(vp_label = vp_mapping[vp_id])
  
  extract_trial_data <- function(spec_row, data) {
    trial_data <- data %>%
      filter(
        as.character(vp_id) == spec_row$vp_id,
        as.numeric(as.character(trial_id)) == spec_row$trial_nr,
        as.character(camera) == spec_row$camera,
        as.character(calibration) == spec_row$calibration
      ) %>%
      arrange(method, timestamp_ms_synced)
    if (nrow(trial_data) == 0) return(NULL)
    trial_data %>%
      group_by(method) %>%
      mutate(time_ms = timestamp_ms_synced - min(timestamp_ms_synced, na.rm = TRUE)) %>%
      ungroup() %>%
      mutate(vp_label = spec_row$vp_label, trial_nr_plot = spec_row$trial_nr)
  }
  
  all_trials_data <- map_dfr(seq_len(nrow(all_trial_specs)), function(i) extract_trial_data(all_trial_specs[i, ], data_fig7))
  
  if (nrow(all_trials_data) > 0) {
    x_values <- c(all_trials_data$cv_deg_x, all_trials_data$eyelink_deg_x)
    y_range <- range(x_values, na.rm = TRUE)
    y_limits <- c(y_range[1] - diff(y_range) * 0.05, y_range[2] + diff(y_range) * 0.05)
    x_limits <- c(0, max(all_trials_data$time_ms, na.rm = TRUE))
    
    identify_blink_regions <- function(data) {
      if (!"is_blink" %in% names(data) || !any(data$is_blink, na.rm = TRUE)) return(tibble(blink_start = numeric(), blink_end = numeric()))
      data %>%
        arrange(time_ms) %>%
        mutate(is_blink_safe = replace_na(is_blink, FALSE), blink_change = is_blink_safe != lag(is_blink_safe, default = FALSE), blink_group = cumsum(blink_change)) %>%
        filter(is_blink_safe) %>%
        group_by(blink_group) %>%
        summarise(blink_start = min(time_ms), blink_end = max(time_ms), .groups = "drop") %>%
        select(-blink_group)
    }
    
    identify_phase_regions <- function(data) {
      if (!"phase_type" %in% names(data)) return(tibble(phase_start = numeric(), phase_end = numeric(), phase_type = character()))
      time_min <- min(data$time_ms, na.rm = TRUE)
      time_max <- max(data$time_ms, na.rm = TRUE)
      has_fix <- any(data$phase_type == "fixation", na.rm = TRUE)
      has_stim <- any(data$phase_type == "stimulus", na.rm = TRUE)
      if (has_fix && has_stim) {
        fix_max <- max(data$time_ms[data$phase_type == "fixation"], na.rm = TRUE)
        stim_min <- min(data$time_ms[data$phase_type == "stimulus"], na.rm = TRUE)
        trans <- (fix_max + stim_min) / 2
        tibble(phase_type = c("fixation", "stimulus"), phase_start = c(time_min, trans), phase_end = c(trans, time_max))
      } else if (has_stim) { tibble(phase_type = "stimulus", phase_start = time_min, phase_end = time_max) }
      else { tibble(phase_start = numeric(), phase_end = numeric(), phase_type = character()) }
    }
    
    blink_regions <- all_trials_data %>% group_by(vp_label) %>% group_modify(~identify_blink_regions(.x)) %>% ungroup()
    phase_regions <- all_trials_data %>% group_by(vp_label) %>% group_modify(~identify_phase_regions(.x)) %>% ungroup()
    phase_colors <- c("fixation" = "#4DAF4A", "stimulus" = "#377EB8")
    
    plot_single_trial <- function(vp_lab, trial_data, blinks, phases, specs, x_lim, y_lim) {
      spec <- specs %>% filter(vp_label == vp_lab)
      eyelink_data <- trial_data %>% filter(method == "mediapipe") %>% select(time_ms, eyelink_deg_x) %>% distinct() %>% mutate(source = "EyeLink", x_pos = eyelink_deg_x)
      cv_data <- trial_data %>% mutate(source = ifelse(method == "mediapipe", "MediaPipe", "PTGaze"), x_pos = cv_deg_x) %>% select(time_ms, source, x_pos)
      plot_data <- bind_rows(eyelink_data %>% select(time_ms, source, x_pos), cv_data) %>% mutate(source = factor(source, levels = c("EyeLink", "MediaPipe", "PTGaze")))
      vp_blinks <- blinks %>% filter(vp_label == vp_lab)
      vp_phases <- phases %>% filter(vp_label == vp_lab)
      camera_display <- ifelse(spec$camera == "25hz", "Systemkamera", "Webcam")
      calib_display <- ifelse(spec$calibration == "FullCalib", "volle Kalib.", "reduz. Kalib.")
      subtitle_text <- paste0(vp_lab, " | Trial ", spec$trial_nr, " | ", camera_display, " | ", calib_display)
      p <- ggplot() +
        {if (nrow(vp_phases) > 0) geom_rect(data = vp_phases, aes(xmin = phase_start, xmax = phase_end, ymin = y_lim[1], ymax = y_lim[2], fill = phase_type), alpha = 0.25)} +
        scale_fill_manual(values = phase_colors, guide = "none") +
        ggnewscale::new_scale_fill() +
        {if (nrow(vp_blinks) > 0) geom_rect(data = vp_blinks, aes(xmin = blink_start, xmax = blink_end, ymin = y_lim[1], ymax = y_lim[2]), fill = "gray70", alpha = 0.6)} +
        geom_line(data = plot_data, aes(x = time_ms, y = x_pos, color = source), linewidth = 0.5, na.rm = TRUE) +
        scale_color_manual(values = c("EyeLink" = "black", "MediaPipe" = "#E41A1C", "PTGaze" = "#377EB8")) +
        scale_x_continuous(limits = x_lim, expand = c(0, 0)) +
        scale_y_continuous(limits = y_lim) +
        labs(subtitle = subtitle_text, x = NULL, y = NULL) +
        theme_apa_fig(base_size = 9) +
        theme(plot.subtitle = element_text(size = 8, face = "bold"), legend.position = "none", plot.margin = margin(2, 5, 2, 5))
      return(p)
    }
    
    vp_order <- paste0("VP ", 1:12)
    plot_list <- map(vp_order, function(vp_lab) {
      vp_data <- all_trials_data %>% filter(vp_label == vp_lab)
      if (nrow(vp_data) == 0) return(ggplot() + annotate("text", x = 0.5, y = 0.5, label = paste(vp_lab, "\nKeine Daten")) + theme_void())
      plot_single_trial(vp_lab, vp_data, blink_regions, phase_regions, all_trial_specs, x_limits, y_limits)
    })
    
    fig7 <- wrap_plots(plot_list, ncol = 2, nrow = 6)
    legend_fig7 <- tibble(x = rep(1:10, 3), y = rep(1:10, 3), source = factor(rep(c("EyeLink", "MediaPipe", "PTGaze"), each = 10)))
    p_legend_fig7 <- ggplot(legend_fig7, aes(x = x, y = y, color = source)) + geom_line(linewidth = 1) +
      scale_color_manual(values = c("EyeLink" = "black", "MediaPipe" = "#E41A1C", "PTGaze" = "#377EB8"), name = "") +
      theme_void() + theme(legend.position = "bottom", legend.text = element_text(size = 10))
    legend_grob_fig7 <- cowplot::get_legend(p_legend_fig7)
    fig_7 <- wrap_plots(fig7, legend_grob_fig7, ncol = 1, heights = c(20, 1))
    
    ggsave(file.path(output_dir, "Figure_7_VP_Trials.png"), fig_7, width = 10, height = 14, dpi = 300, bg = "white")
    ggsave(file.path(output_dir, "Figure_7_VP_Trials.pdf"), fig_7, width = 10, height = 14)
    message("-> Abbildung 7 gespeichert")
  }
} else {
  message("[WARNUNG] data_fig7 nicht verfügbar - Figure 7 übersprungen")
}

# =============================================================================
# 9. SPEICHERN
# =============================================================================

message("\n", paste(rep("=", 70), collapse = ""))
message("SPEICHERN")
message(paste(rep("=", 70), collapse = ""))

save(
  data_trial,
  data_vp,
  file = file.path(output_dir, "01_trial_level.RData")
)

message("Gespeichert: 01_trial_level.RData")
message("  - data_trial: Trial-Level Metriken (valide Trials)")
message("  - data_vp: VP-Level Metriken pro Bedingung")

message("\n[OK] Deskriptive Statistik abgeschlossen.")
message("Weiter mit: 02_lmm.R")
