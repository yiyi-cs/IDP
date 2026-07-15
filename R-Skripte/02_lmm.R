# =============================================================================
# 02_LMM.R - Linear Mixed Models Analysis
# =============================================================================
# Validierungsstudie: CV Eye-Tracking vs. EyeLink 1000 Plus
#
# ZWECK:
#   Inferenzstatistische Analyse mittels linearer gemischter Modelle (LMM).
#   Erzeugt Tabellen und Abbildungen 5, 6 der Arbeit.
#
# WORKFLOW:
#   1. Daten aus 01_trial_level.RData laden
#   2. ICC-Analyse (Rechtfertigung Multilevel-Modellierung)
#   3. Modellselektion (Fixed Effects, Random Effects)
#   4. Finale Modelle fitten und Post-hoc-Analysen
#   5. APA-Tabellen erstellen
#   6. Visualisierungen (Haupteffekte, Interaktionen)
#   7. Diagnostik-Plots
#
# OUTPUT:
#   - Table_ModelSelection.docx
#   - Table_Results.docx
#   - Table_PostHoc.docx
#   - Figure_5_MainEffects.png/.pdf
#   - Figure_6_Interaction.png/.pdf
#   - Diagnostik-Plots
#   - 02_lmm_results.RData
#
# REFERENZEN:
#   Barr et al. (2013), Matuschek et al. (2017), Bates et al. (2018)
# =============================================================================

# =============================================================================
# 1. SETUP
# =============================================================================

library(lme4)
library(lmerTest)
library(emmeans)
library(MuMIn)
library(performance)
library(flextable)
library(officer)
library(ggplot2)
library(dplyr)
library(tidyr)
library(effectsize)
library(patchwork)
library(cowplot)

# -----------------------------------------------------------------------------
# Daten laden
# -----------------------------------------------------------------------------
if (!exists("data_trial")) {
  load(file.path(config$output_base, "01_descriptives_and_regression", "01_trial_level.RData"))
  message("Daten aus 01_trial_level.RData geladen")
}

data <- data_trial

# Output-Verzeichnis
output_dir <- file.path(config$output_base, "02_lmm")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# -----------------------------------------------------------------------------
# Kontraste prüfen (Sum-Coding aus 00_setup)
# -----------------------------------------------------------------------------
stopifnot(all(contrasts(data$method) == c(0.5, -0.5)))
stopifnot(all(contrasts(data$camera) == c(0.5, -0.5)))
stopifnot(all(contrasts(data$calibration) == c(0.5, -0.5)))

# AVs und Labels
avs <- c("r_z_x", "slope_x", "bias_x")
av_labels <- c(r_z_x = "Correlation", slope_x = "Slope", bias_x = "Bias")

results <- list()
models_final <- list()

message("\n", paste(rep("=", 70), collapse = ""))
message("LINEAR MIXED MODELS ANALYSIS")
message(paste(rep("=", 70), collapse = ""))

# =============================================================================
# 2. ICC-ANALYSE
# =============================================================================
# Prüft, ob Multilevel-Modellierung gerechtfertigt ist (ICC > .05)

message("\n=== ICC-ANALYSE ===")

icc_results <- data.frame(DV = av_labels[avs], ICC = NA)

for (i in seq_along(avs)) {
  av <- avs[i]
  null_model <- lmer(as.formula(paste(av, "~ 1 + (1 | vp_id)")), data = data, REML = TRUE)
  icc_val <- performance::icc(null_model)
  icc_results$ICC[i] <- round(icc_val$ICC_adjusted, 3)
}

print(icc_results)
message("-> Alle ICCs > 0.05: Multilevel-Modellierung gerechtfertigt")

# =============================================================================
# 3. MODELLSELEKTION
# =============================================================================

# -----------------------------------------------------------------------------
# Hilfsfunktion: Sicheres LMM-Fitting
# -----------------------------------------------------------------------------
safe_lmer <- function(formula, data, REML = TRUE) {
  tryCatch({
    m <- lmer(formula, data = data, REML = REML,
              control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    list(model = m, status = ifelse(isSingular(m), "singular", "ok"))
  }, error = function(e) list(model = NULL, status = "failed"),
  warning = function(w) {
    m <- suppressWarnings(lmer(formula, data = data, REML = REML,
                               control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))))
    list(model = m, status = "warning")
  })
}

# -----------------------------------------------------------------------------
# 3A: Fixed Effects Selektion
# -----------------------------------------------------------------------------
select_fixed_effects <- function(av, data) {
  message("\n--- Fixed Effects: ", av_labels[av], " ---")
  
  formulas <- list(
    `Full (3-way)` = paste(av, "~ method * camera * calibration + (1|vp_id)"),
    `All 2-way` = paste(av, "~ method*camera + method*calibration + camera*calibration + (1|vp_id)"),
    `Method × Camera` = paste(av, "~ method * camera + calibration + (1|vp_id)"),
    `Method × Calibration` = paste(av, "~ method * calibration + camera + (1|vp_id)"),
    `Camera × Calibration` = paste(av, "~ camera * calibration + method + (1|vp_id)"),
    `Main effects` = paste(av, "~ method + camera + calibration + (1|vp_id)")
  )
  
  models <- lapply(formulas, function(f) lmer(as.formula(f), data = data, REML = FALSE))
  
  comp <- data.frame(
    Model = names(models),
    k = sapply(models, function(m) attr(logLik(m), "df")),
    AIC = sapply(models, AIC),
    BIC = sapply(models, BIC)
  )
  comp$ΔAIC <- round(comp$AIC - min(comp$AIC), 1)
  comp$ΔBIC <- round(comp$BIC - min(comp$BIC), 1)
  comp <- comp[order(comp$AIC), ]
  
  cat("\nModel Comparison (sorted by AIC):\n")
  print(comp, row.names = FALSE)
  
  # LRT-Kaskade
  lrts <- data.frame(
    Comparison = c("Full (3-way) vs All 2-way", "All 2-way vs Main effects",
                   "Method × Camera vs Main effects", "Method × Camera vs All 2-way",
                   "Method × Calibration vs Main effects", "Camera × Calibration vs Main effects"),
    χ2 = NA, Δdf = NA, p = NA
  )
  
  lrt_results <- list(
    anova(models$`All 2-way`, models$`Full (3-way)`),
    anova(models$`Main effects`, models$`All 2-way`),
    anova(models$`Main effects`, models$`Method × Camera`),
    anova(models$`Method × Camera`, models$`All 2-way`),
    anova(models$`Main effects`, models$`Method × Calibration`),
    anova(models$`Main effects`, models$`Camera × Calibration`)
  )
  
  for (i in 1:6) {
    lrts[i, 2:4] <- c(lrt_results[[i]]$Chisq[2], lrt_results[[i]]$Df[2], lrt_results[[i]]$`Pr(>Chisq)`[2])
  }
  lrts$χ2 <- round(lrts$χ2, 2)
  lrts$p <- round(lrts$p, 4)
  
  cat("\nLRT Results:\n")
  print(lrts, row.names = FALSE)
  
  # Entscheidungslogik
  p_3way_vs_2way <- lrts$p[1]
  p_2way_vs_main <- lrts$p[2]
  p_mc_vs_main <- lrts$p[3]
  p_mcal_vs_main <- lrts$p[5]
  p_ccal_vs_main <- lrts$p[6]
  
  if (p_2way_vs_main >= 0.05) {
    selected <- "Main effects"
    formula_selected <- formulas$`Main effects`
  } else if (p_3way_vs_2way >= 0.05) {
    sig_mc <- p_mc_vs_main < 0.05
    sig_mcal <- p_mcal_vs_main < 0.05
    sig_ccal <- p_ccal_vs_main < 0.05
    n_sig <- sum(c(sig_mc, sig_mcal, sig_ccal))
    
    if (n_sig == 0) {
      selected <- "Main effects"
      formula_selected <- formulas$`Main effects`
    } else if (n_sig == 1 && sig_mc) {
      selected <- "Method × Camera"
      formula_selected <- formulas$`Method × Camera`
    } else if (n_sig == 1 && sig_mcal) {
      selected <- "Method × Calibration"
      formula_selected <- formulas$`Method × Calibration`
    } else if (n_sig == 1 && sig_ccal) {
      selected <- "Camera × Calibration"
      formula_selected <- formulas$`Camera × Calibration`
    } else {
      selected <- "All 2-way"
      formula_selected <- formulas$`All 2-way`
    }
  } else {
    selected <- "Full (3-way)"
    formula_selected <- formulas$`Full (3-way)`
  }
  
  cat("\n-> Selected:", selected, "\n")
  return(list(comparison = comp, lrts = lrts, selected = selected, formula = formula_selected, models = models))
}

# -----------------------------------------------------------------------------
# 3B: Random Effects Selektion
# -----------------------------------------------------------------------------
select_random_effects <- function(av, fixed_formula, data) {
  message("\n--- Random Effects: ", av_labels[av], " ---")
  
  fixed_part <- gsub("\\+ \\(1\\|vp_id\\)", "", fixed_formula)
  fixed_part <- trimws(fixed_part)
  
  random_specs <- list(
    `All slopes (corr)` = "(1 + method + camera + calibration | vp_id)",
    `All slopes (no corr)` = "(1 + method + camera + calibration || vp_id)",
    `Method slope` = "(1 + method | vp_id)",
    `Camera slope` = "(1 + camera | vp_id)",
    `Calibration slope` = "(1 + calibration | vp_id)",
    `Intercept only` = "(1 | vp_id)"
  )
  
  results_re <- data.frame(Structure = names(random_specs), Status = NA, AIC = NA, BIC = NA)
  models_re <- list()
  
  for (i in seq_along(random_specs)) {
    name <- names(random_specs)[i]
    f <- as.formula(paste(fixed_part, "+", random_specs[[i]]))
    fit <- safe_lmer(f, data, REML = TRUE)
    models_re[[name]] <- fit
    results_re$Status[i] <- fit$status
    if (fit$status == "ok") {
      results_re$AIC[i] <- round(AIC(fit$model), 1)
      results_re$BIC[i] <- round(BIC(fit$model), 1)
    }
  }
  
  cat("\nRandom Structures:\n")
  print(results_re, row.names = FALSE)
  
  valid <- results_re[results_re$Status == "ok", ]
  if (nrow(valid) == 0) valid <- results_re[results_re$Status != "failed", ]
  best <- valid[which.min(valid$AIC), "Structure"]
  
  # LRT gegen Intercept-only (α = .20 für Random Effects)
  if ("Intercept only" %in% valid$Structure && best != "Intercept only") {
    base_ml <- lmer(as.formula(paste(fixed_part, "+ (1 | vp_id)")), data = data, REML = FALSE)
    test_ml <- lmer(as.formula(paste(fixed_part, "+", random_specs[[best]])), data = data, REML = FALSE)
    lrt <- anova(base_ml, test_ml)
    cat(sprintf("\nLRT %s vs Intercept only: χ²(%d) = %.2f, p = %.4f\n", 
                best, lrt$Df[2], lrt$Chisq[2], lrt$`Pr(>Chisq)`[2]))
    if (lrt$`Pr(>Chisq)`[2] >= 0.20) {
      best <- "Intercept only"
      cat("-> p >= .20: Intercept only retained\n")
    }
  }
  
  cat("-> Selected:", best, "\n")
  final_formula <- as.formula(paste(fixed_part, "+", random_specs[[best]]))
  return(list(comparison = results_re, selected = best, formula = final_formula, random_spec = random_specs[[best]]))
}

# -----------------------------------------------------------------------------
# Modellselektion durchführen
# -----------------------------------------------------------------------------
message("\n========== MODEL SELECTION ==========")

selection_results <- list()

for (av in avs) {
  fe_result <- select_fixed_effects(av, data)
  re_result <- select_random_effects(av, fe_result$formula, data)
  selection_results[[av]] <- list(fixed = fe_result, random = re_result, final_formula = re_result$formula)
}

# =============================================================================
# 4. FINALE MODELLE FITTEN
# =============================================================================

message("\n========== FINAL MODELS ==========")

for (av in avs) {
  message("\n=== ", toupper(av_labels[av]), " ===")
  
  final_formula <- selection_results[[av]]$final_formula
  cat("Formula:", deparse(final_formula), "\n\n")
  
  model <- lmer(final_formula, data = data, REML = TRUE)
  models_final[[av]] <- model
  
  cat("--- Summary ---\n")
  print(summary(model))
  
  cat("\n--- Type III ANOVA ---\n")
  aov <- anova(model, type = 3, ddf = "Satterthwaite")
  print(aov)
  
  r2 <- r.squaredGLMM(model)
  cat("\nMarginal R²:", round(r2[1, "R2m"], 4))
  cat("\nConditional R²:", round(r2[1, "R2c"], 4), "\n")
  
  results[[av]] <- list(model = model, summary = summary(model), anova = aov, r2 = r2)
}

# =============================================================================
# 5. POST-HOC ANALYSEN
# =============================================================================

message("\n========== POST-HOC ANALYSES ==========")

posthoc_results <- list()

for (av in avs) {
  model <- models_final[[av]]
  aov <- results[[av]]$anova
  sig_effects <- rownames(aov)[aov$`Pr(>F)` < 0.05]
  
  if (length(sig_effects) == 0) {
    message("\n", av_labels[av], ": No significant effects")
    next
  }
  
  message("\n--- Post-hoc: ", av_labels[av], " ---")
  message("Significant effects: ", paste(sig_effects, collapse = ", "))
  
  posthoc_results[[av]] <- list()
  
  for (eff in sig_effects) {
    if (eff %in% c("method", "camera", "calibration")) {
      emm <- emmeans(model, as.formula(paste("~", eff)))
      pw <- pairs(emm)
      eff_size_result <- eff_size(emm, sigma = sigma(model), edf = df.residual(model))
      eff_size_df <- as.data.frame(eff_size_result)
      
      cat("\n", eff, ":\n")
      print(summary(emm))
      cat("Pairwise:\n")
      print(pw)
      cat("Hedges' g:", round(eff_size_df$effect.size[1], 3), "\n")
      
      posthoc_results[[av]][[eff]] <- list(emm = emm, pairs = pw, effect_size = eff_size_df)
      
    } else if (grepl(":", eff)) {
      vars <- strsplit(eff, ":")[[1]]
      emm <- emmeans(model, as.formula(paste("~", vars[1], "*", vars[2])))
      
      cat("\n", eff, " - Cell means:\n")
      print(summary(emm))
      
      cat("\nSimple Effects (", vars[1], "|", vars[2], "):\n")
      se1 <- pairs(emmeans(model, as.formula(paste("~", vars[1], "|", vars[2]))))
      print(se1)
      
      cat("\nSimple Effects (", vars[2], "|", vars[1], "):\n")
      se2 <- pairs(emmeans(model, as.formula(paste("~", vars[2], "|", vars[1]))))
      print(se2)
      
      posthoc_results[[av]][[eff]] <- list(emm = emm, simple1 = se1, simple2 = se2)
    }
  }
}

# =============================================================================
# 6. APA-TABELLEN
# =============================================================================

message("\n========== CREATING TABLES ==========")

# -----------------------------------------------------------------------------
# APA Theme für Tabellen
# -----------------------------------------------------------------------------
apa_theme <- function(ft) {
  ft %>%
    font(fontname = "Times New Roman", part = "all") %>%
    fontsize(size = 12, part = "all") %>%
    align(align = "center", part = "header") %>%
    align(align = "left", j = 1, part = "body") %>%
    hline_top(border = fp_border(width = 2), part = "header") %>%
    hline_bottom(border = fp_border(width = 2), part = "header") %>%
    hline_bottom(border = fp_border(width = 2), part = "body") %>%
    padding(padding = 3, part = "all")
}

add_dv_separators <- function(ft, dv_col = "DV", data) {
  dv_changes <- which(data[[dv_col]][-1] != data[[dv_col]][-nrow(data)])
  for (row in dv_changes) {
    ft <- hline(ft, i = row, border = fp_border(color = "gray70", width = 1))
  }
  return(ft)
}

z_to_r <- function(z) tanh(z)

# -----------------------------------------------------------------------------
# Tabelle: Modellvergleich
# -----------------------------------------------------------------------------
model_comp_all <- data.frame()
for (av in avs) {
  comp <- selection_results[[av]]$fixed$comparison
  lrts <- selection_results[[av]]$fixed$lrts
  comp$DV <- av_labels[av]
  comp$χ2 <- NA; comp$Δdf <- NA; comp$p_lrt <- NA
  
  for (i in 1:nrow(comp)) {
    model_name <- comp$Model[i]
    if (model_name == "Full (3-way)") { comp$χ2[i] <- lrts$χ2[1]; comp$Δdf[i] <- lrts$Δdf[1]; comp$p_lrt[i] <- lrts$p[1] }
    else if (model_name == "All 2-way") { comp$χ2[i] <- lrts$χ2[2]; comp$Δdf[i] <- lrts$Δdf[2]; comp$p_lrt[i] <- lrts$p[2] }
    else if (model_name == "Method × Camera") { comp$χ2[i] <- lrts$χ2[3]; comp$Δdf[i] <- lrts$Δdf[3]; comp$p_lrt[i] <- lrts$p[3] }
    else if (model_name == "Method × Calibration") { comp$χ2[i] <- lrts$χ2[5]; comp$Δdf[i] <- lrts$Δdf[5]; comp$p_lrt[i] <- lrts$p[5] }
    else if (model_name == "Camera × Calibration") { comp$χ2[i] <- lrts$χ2[6]; comp$Δdf[i] <- lrts$Δdf[6]; comp$p_lrt[i] <- lrts$p[6] }
  }
  model_comp_all <- rbind(model_comp_all, comp)
}

model_comp_table <- model_comp_all %>%
  select(DV, Model, k, AIC, BIC, ΔAIC, ΔBIC, χ2, Δdf, p_lrt) %>%
  mutate(
    AIC = sprintf("%.1f", AIC), BIC = sprintf("%.1f", BIC),
    χ2 = ifelse(is.na(χ2), "—", sprintf("%.2f", χ2)),
    Δdf = ifelse(is.na(Δdf), "—", as.character(Δdf)),
    p_lrt = case_when(is.na(p_lrt) ~ "—", p_lrt < .001 ~ "< .001", TRUE ~ sprintf("%.3f", p_lrt))
  )

ft_model_comp <- flextable(model_comp_table) %>%
  set_header_labels(DV = "DV", Model = "Model", k = "k", AIC = "AIC", BIC = "BIC",
                    ΔAIC = "ΔAIC", ΔBIC = "ΔBIC", χ2 = "χ²", Δdf = "Δdf", p_lrt = "p") %>%
  apa_theme() %>%
  merge_v(j = "DV") %>%
  add_dv_separators("DV", model_comp_table) %>%
  italic(j = "Model", part = "body") %>%
  add_footer_lines("Note. k = number of parameters; ΔAIC and ΔBIC = difference from best-fitting model.") %>%
  autofit()

save_as_docx(ft_model_comp, path = file.path(output_dir, "Table_ModelSelection.docx"))

# -----------------------------------------------------------------------------
# Tabelle: Kombinierte Ergebnisse
# -----------------------------------------------------------------------------
create_combined_table <- function(avs, models_final, results, icc_results) {
  all_params <- unique(unlist(lapply(models_final, function(m) names(fixef(m)))))
  rows <- list()
  
  # --- Fixed Effects Section ---
  rows[[length(rows) + 1]] <- data.frame(
    Section = "Fixed effects", Effect = "", Parameter = "",
    Correlation = "", Slope = "", Bias = "", stringsAsFactors = FALSE
  )
  
  for (param in all_params) {
    param_label <- case_when(
      param == "(Intercept)" ~ "Intercept", param == "method1" ~ "Method",
      param == "camera1" ~ "Camera", param == "calibration1" ~ "Calibration",
      param == "method1:camera1" ~ "Method × Camera",
      param == "method1:calibration1" ~ "Method × Calibration",
      param == "camera1:calibration1" ~ "Camera × Calibration", TRUE ~ param
    )
    param_symbol <- case_when(
      param == "(Intercept)" ~ "γ₀₀", param == "method1" ~ "γ₁₀",
      param == "camera1" ~ "γ₂₀", param == "calibration1" ~ "γ₃₀",
      param == "method1:camera1" ~ "γ₁₂",
      param == "method1:calibration1" ~ "γ₁₃",
      param == "camera1:calibration1" ~ "γ₂₃", TRUE ~ ""
    )
    
    corr_val <- ""; slope_val <- ""; bias_val <- ""
    for (av in avs) {
      model <- models_final[[av]]
      summ <- summary(model)
      coefs <- summ$coefficients
      if (param %in% rownames(coefs)) {
        est <- coefs[param, "Estimate"]; se <- coefs[param, "Std. Error"]; p <- coefs[param, "Pr(>|t|)"]
        stars <- case_when(p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", TRUE ~ "")
        val <- sprintf("%.2f%s (%.2f)", est, stars, se)
        if (av == "r_z_x") corr_val <- val
        else if (av == "slope_x") slope_val <- val
        else if (av == "bias_x") bias_val <- val
      }
    }
    rows[[length(rows) + 1]] <- data.frame(Section = "", Effect = param_label, Parameter = param_symbol,
                                           Correlation = corr_val, Slope = slope_val, Bias = bias_val, stringsAsFactors = FALSE)
  }
  
  # --- Random Effects Section ---
  rows[[length(rows) + 1]] <- data.frame(
    Section = "Random effects", Effect = "", Parameter = "",
    Correlation = "", Slope = "", Bias = "", stringsAsFactors = FALSE
  )
  
  # Level 1 Residual Variance (σ²ₑ)
  rows[[length(rows) + 1]] <- data.frame(
    Section = "", Effect = "Level 1 (Residual)", Parameter = "σ²ₑ",
    Correlation = sprintf("%.3f", sigma(models_final$r_z_x)^2),
    Slope = sprintf("%.3f", sigma(models_final$slope_x)^2),
    Bias = sprintf("%.3f", sigma(models_final$bias_x)^2),
    stringsAsFactors = FALSE
  )
  
  # Variance Components extrahieren
  corr_vc <- as.data.frame(VarCorr(models_final$r_z_x))
  slope_vc <- as.data.frame(VarCorr(models_final$slope_x))
  bias_vc <- as.data.frame(VarCorr(models_final$bias_x))
  
  # Level 2 Intercept Variance (σ²ᵤ₀)
  corr_int <- corr_vc[corr_vc$var1 == "(Intercept)" & is.na(corr_vc$var2), "vcov"][1]
  slope_int <- slope_vc[slope_vc$var1 == "(Intercept)" & is.na(slope_vc$var2), "vcov"][1]
  bias_int <- bias_vc[bias_vc$var1 == "(Intercept)" & is.na(bias_vc$var2), "vcov"][1]
  
  rows[[length(rows) + 1]] <- data.frame(
    Section = "", Effect = "Level 2 (VP)", Parameter = "σ²ᵤ₀",
    Correlation = sprintf("%.3f", corr_int),
    Slope = sprintf("%.3f", slope_int),
    Bias = sprintf("%.3f", bias_int),
    stringsAsFactors = FALSE
  )
  
  # Random Slopes (σ²ᵤ₁) - NEU HINZUGEFÜGT
  corr_slope_vc <- corr_vc[corr_vc$var1 != "(Intercept)" & is.na(corr_vc$var2), ]
  slope_slope_vc <- slope_vc[slope_vc$var1 != "(Intercept)" & is.na(slope_vc$var2), ]
  bias_slope_vc <- bias_vc[bias_vc$var1 != "(Intercept)" & is.na(bias_vc$var2), ]
  
  corr_slope_val <- if (nrow(corr_slope_vc) > 0) {
    sprintf("%.3f (%s)", corr_slope_vc$vcov[1], tools::toTitleCase(gsub("1$", "", corr_slope_vc$var1[1])))
  } else "—"
  
  slope_slope_val <- if (nrow(slope_slope_vc) > 0) {
    sprintf("%.3f (%s)", slope_slope_vc$vcov[1], tools::toTitleCase(gsub("1$", "", slope_slope_vc$var1[1])))
  } else "—"
  
  bias_slope_val <- if (nrow(bias_slope_vc) > 0) {
    sprintf("%.3f (%s)", bias_slope_vc$vcov[1], tools::toTitleCase(gsub("1$", "", bias_slope_vc$var1[1])))
  } else "—"
  
  rows[[length(rows) + 1]] <- data.frame(
    Section = "", Effect = "Random Slope", Parameter = "σ²ᵤ₁",
    Correlation = corr_slope_val,
    Slope = slope_slope_val,
    Bias = bias_slope_val,
    stringsAsFactors = FALSE
  )
  
  # Correlation between Random Intercept and Slope (rᵤ₀₁) - NEU HINZUGEFÜGT
  corr_corr_re <- corr_vc[!is.na(corr_vc$var2), ]
  slope_corr_re <- slope_vc[!is.na(slope_vc$var2), ]
  bias_corr_re <- bias_vc[!is.na(bias_vc$var2), ]
  
  corr_corr_val <- if (nrow(corr_corr_re) > 0) sprintf("%.2f", corr_corr_re$sdcor[1]) else "—"
  slope_corr_val <- if (nrow(slope_corr_re) > 0) sprintf("%.2f", slope_corr_re$sdcor[1]) else "—"
  bias_corr_val <- if (nrow(bias_corr_re) > 0) sprintf("%.2f", bias_corr_re$sdcor[1]) else "—"
  
  rows[[length(rows) + 1]] <- data.frame(
    Section = "", Effect = "Korrelation (Intercept, Slope)", Parameter = "rᵤ₀₁",
    Correlation = corr_corr_val,
    Slope = slope_corr_val,
    Bias = bias_corr_val,
    stringsAsFactors = FALSE
  )
  
  # --- Goodness of Fit Section ---
  rows[[length(rows) + 1]] <- data.frame(
    Section = "Goodness of fit", Effect = "", Parameter = "",
    Correlation = "", Slope = "", Bias = "", stringsAsFactors = FALSE
  )
  
  rows[[length(rows) + 1]] <- data.frame(
    Section = "", Effect = "Marginal R²", Parameter = "R²m",
    Correlation = sprintf("%.3f", results$r_z_x$r2[1, "R2m"]),
    Slope = sprintf("%.3f", results$slope_x$r2[1, "R2m"]),
    Bias = sprintf("%.3f", results$bias_x$r2[1, "R2m"]),
    stringsAsFactors = FALSE
  )
  
  rows[[length(rows) + 1]] <- data.frame(
    Section = "", Effect = "Conditional R²", Parameter = "R²c",
    Correlation = sprintf("%.3f", results$r_z_x$r2[1, "R2c"]),
    Slope = sprintf("%.3f", results$slope_x$r2[1, "R2c"]),
    Bias = sprintf("%.3f", results$bias_x$r2[1, "R2c"]),
    stringsAsFactors = FALSE
  )
  
  do.call(rbind, rows)
}

combined_table <- create_combined_table(avs, models_final, results, icc_results)
section_rows <- which(combined_table$Section != "")

ft_combined <- flextable(combined_table) %>%
  set_header_labels(Section = "", Effect = "Effect", Parameter = "Parameter",
                    Correlation = "Correlation", Slope = "Slope", Bias = "Bias") %>%
  apa_theme() %>%
  bold(i = ~ Section != "", j = "Section") %>%
  merge_v(j = "Section") %>%
  hline(i = section_rows[-1] - 1, border = fp_border(color = "gray70", width = 1)) %>%
  add_footer_lines("Note. Standard errors in parentheses. * p < .05. ** p < .01. *** p < .001.") %>%
  autofit()

save_as_docx(ft_combined, path = file.path(output_dir, "Table_Results.docx"))

# -----------------------------------------------------------------------------
# Tabelle: Post-hoc Vergleiche
# -----------------------------------------------------------------------------
posthoc_table <- data.frame()

for (av in avs) {
  if (is.null(posthoc_results[[av]])) next
  
  model <- models_final[[av]]
  aov <- results[[av]]$anova
  sig_effects <- rownames(aov)[aov$`Pr(>F)` < 0.05]
  
  # Residualstandardabweichung für Hedges' g
  sigma_model <- sigma(model)
  
  # Prüfe welche Haupteffekte in signifikanten Interaktionen involviert sind
  sig_interactions <- sig_effects[grepl(":", sig_effects)]
  main_effects_in_interactions <- c()
  for (int in sig_interactions) {
    vars <- strsplit(int, ":")[[1]]
    main_effects_in_interactions <- c(main_effects_in_interactions, vars)
  }
  main_effects_in_interactions <- unique(main_effects_in_interactions)
  
  for (eff in names(posthoc_results[[av]])) {
    
    # --- HAUPTEFFEKTE ---
    # NUR berichten wenn NICHT in signifikanter Interaktion involviert
    if (eff %in% c("method", "camera", "calibration")) {
      
      if (eff %in% main_effects_in_interactions) {
        message(sprintf("[INFO] %s: Haupteffekt '%s' übersprungen (in signifikanter Interaktion)", 
                        av_labels[av], eff))
        next
      }
      
      pairs_df <- as.data.frame(posthoc_results[[av]][[eff]]$pairs)
      eff_df <- posthoc_results[[av]][[eff]]$effect_size
      
      row <- data.frame(
        AV = av_labels[av],
        Effect = tools::toTitleCase(eff),
        Δ = sprintf("%.3f", pairs_df$estimate[1]),
        SE = sprintf("%.3f", pairs_df$SE[1]),
        `95% CI` = sprintf("[%.2f, %.2f]", 
                           pairs_df$estimate[1] - 1.96 * pairs_df$SE[1],
                           pairs_df$estimate[1] + 1.96 * pairs_df$SE[1]),
        t = sprintf("%.2f", pairs_df$t.ratio[1]),
        df = sprintf("%.1f", pairs_df$df[1]),
        p = ifelse(pairs_df$p.value[1] < .001, "< .001", sprintf("%.3f", pairs_df$p.value[1])),
        g = sprintf("%.2f", eff_df$effect.size[1]),
        check.names = FALSE, stringsAsFactors = FALSE
      )
      posthoc_table <- rbind(posthoc_table, row)
    }
    
    # --- INTERAKTIONEN: Simple Effects ---
    if (grepl(":", eff)) {
      vars <- strsplit(eff, ":")[[1]]
      
      # Simple Effects 1: vars[1] | vars[2]
      se1_df <- as.data.frame(posthoc_results[[av]][[eff]]$simple1)
      
      for (i in 1:nrow(se1_df)) {
        # KORREKTES Hedges' g: Δ / σ(model)
        hedges_g <- se1_df$estimate[i] / sigma_model
        
        row <- data.frame(
          AV = av_labels[av],
          Effect = paste0(tools::toTitleCase(vars[1]), " | ", tools::toTitleCase(vars[2])),
          Δ = sprintf("%.3f", se1_df$estimate[i]),
          SE = sprintf("%.3f", se1_df$SE[i]),
          `95% CI` = sprintf("[%.2f, %.2f]", 
                             se1_df$estimate[i] - 1.96 * se1_df$SE[i],
                             se1_df$estimate[i] + 1.96 * se1_df$SE[i]),
          t = sprintf("%.2f", se1_df$t.ratio[i]),
          df = sprintf("%.1f", se1_df$df[i]),
          p = ifelse(se1_df$p.value[i] < .001, "< .001", sprintf("%.3f", se1_df$p.value[i])),
          g = sprintf("%.2f", hedges_g),
          check.names = FALSE, stringsAsFactors = FALSE
        )
        posthoc_table <- rbind(posthoc_table, row)
      }
      
      # Simple Effects 2: vars[2] | vars[1]
      se2_df <- as.data.frame(posthoc_results[[av]][[eff]]$simple2)
      
      for (i in 1:nrow(se2_df)) {
        # KORREKTES Hedges' g: Δ / σ(model)
        hedges_g <- se2_df$estimate[i] / sigma_model
        
        row <- data.frame(
          AV = av_labels[av],
          Effect = paste0(tools::toTitleCase(vars[2]), " | ", tools::toTitleCase(vars[1])),
          Δ = sprintf("%.3f", se2_df$estimate[i]),
          SE = sprintf("%.3f", se2_df$SE[i]),
          `95% CI` = sprintf("[%.2f, %.2f]", 
                             se2_df$estimate[i] - 1.96 * se2_df$SE[i],
                             se2_df$estimate[i] + 1.96 * se2_df$SE[i]),
          t = sprintf("%.2f", se2_df$t.ratio[i]),
          df = sprintf("%.1f", se2_df$df[i]),
          p = ifelse(se2_df$p.value[i] < .001, "< .001", sprintf("%.3f", se2_df$p.value[i])),
          g = sprintf("%.2f", hedges_g),
          check.names = FALSE, stringsAsFactors = FALSE
        )
        posthoc_table <- rbind(posthoc_table, row)
      }
    }
  }
}

# Tabelle erstellen
if (nrow(posthoc_table) > 0) {
  ft_posthoc <- flextable(posthoc_table) %>%
    set_header_labels(
      AV = "AV", Effect = "Effekt",
      Δ = "Δ", SE = "SE", `95% CI` = "95% KI", t = "t", df = "df", p = "p", g = "g"
    ) %>%
    apa_theme() %>%
    merge_v(j = "AV") %>%
    add_dv_separators("AV", posthoc_table) %>%
    autofit()
  
  save_as_docx(ft_posthoc, path = file.path(output_dir, "Table_PostHoc.docx"))
  message("-> Table_PostHoc.docx gespeichert")
}

message("-> Tables saved")

# =============================================================================
# 7. VISUALISIERUNGEN
# =============================================================================

message("\n========== CREATING FIGURES ==========")

# -----------------------------------------------------------------------------
# Setup: Farben, Labels, Theme
# -----------------------------------------------------------------------------
vp_mapping <- c(
  "beo7" = "VP 1",  "bjs4" = "VP 2",  "egf5" = "VP 3",  "fbn6" = "VP 4",
  "fgt6" = "VP 5",  "jkl7" = "VP 6",  "kdn8" = "VP 7",  "kro3" = "VP 8",
  "ldj9" = "VP 9",  "mhe9" = "VP 10", "oem4" = "VP 11", "ogt7" = "VP 12"
)

vp_ids_original <- sort(unique(as.character(data$vp_id)))
vp_labels <- vp_mapping[vp_ids_original]

vp_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
               "#A65628", "#F781BF", "#1B9E77", "#D95F02", "#7570B3", "#E7298A")
names(vp_colors) <- vp_ids_original

vp_shapes <- c(16, 16, 16, 16, 17, 17, 17, 17, 15, 15, 15, 15)
names(vp_shapes) <- vp_ids_original

labels_de <- list(
  method = c("mediapipe" = "MediaPipe", "ptgaze" = "PTGaze"),
  camera = c("25hz" = "System-\nkamera", "60hz" = "Webcam"),
  camera_interaction = c("25hz" = "Systemkamera", "60hz" = "Webcam"),
  calibration = c("FullCalib" = "Voll", "BegFirst10EndFix" = "Reduziert")
)

theme_apa_fig <- function(base_size = 9) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(family = "sans"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
      axis.title = element_text(face = "bold", size = base_size),
      axis.title.y = element_text(margin = margin(r = 3)),
      axis.title.x = element_text(margin = margin(t = 3)),
      axis.text = element_text(color = "black", size = base_size - 1),
      axis.line = element_line(color = "black", linewidth = 0.4),
      legend.position = "none",
      strip.text = element_text(face = "bold", size = base_size),
      strip.background = element_blank(),
      plot.title = element_text(face = "bold", size = base_size, hjust = 0.5),
      plot.margin = margin(2, 4, 2, 2)
    )
}

# -----------------------------------------------------------------------------
# VP-Mittelwerte berechnen
# -----------------------------------------------------------------------------
vp_means_method <- data %>%
  group_by(vp_id, method) %>%
  summarise(bias = mean(bias_x, na.rm = TRUE), slope = mean(slope_x, na.rm = TRUE),
            r = z_to_r(mean(r_z_x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(vp_id = factor(vp_id, levels = vp_ids_original))

vp_means_camera <- data %>%
  group_by(vp_id, camera) %>%
  summarise(bias = mean(bias_x, na.rm = TRUE), slope = mean(slope_x, na.rm = TRUE),
            r = z_to_r(mean(r_z_x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(vp_id = factor(vp_id, levels = vp_ids_original))

vp_means_calib <- data %>%
  group_by(vp_id, calibration) %>%
  summarise(bias = mean(bias_x, na.rm = TRUE), slope = mean(slope_x, na.rm = TRUE),
            r = z_to_r(mean(r_z_x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(vp_id = factor(vp_id, levels = vp_ids_original))

vp_means_interaction <- data %>%
  group_by(vp_id, method, camera) %>%
  summarise(r = z_to_r(mean(r_z_x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(vp_id = factor(vp_id, levels = vp_ids_original))

obs_means_method <- vp_means_method %>% group_by(method) %>% summarise(across(c(bias, slope, r), mean), .groups = "drop")
obs_means_camera <- vp_means_camera %>% group_by(camera) %>% summarise(across(c(bias, slope, r), mean), .groups = "drop")
obs_means_calib <- vp_means_calib %>% group_by(calibration) %>% summarise(across(c(bias, slope, r), mean), .groups = "drop")
obs_means_interaction <- vp_means_interaction %>% group_by(method, camera) %>% summarise(r = mean(r), .groups = "drop")

# -----------------------------------------------------------------------------
# EMMs extrahieren
# -----------------------------------------------------------------------------
emm_bias_method <- as.data.frame(emmeans(models_final$bias_x, ~ method))
emm_bias_camera <- as.data.frame(emmeans(models_final$bias_x, ~ camera))
emm_bias_calib <- as.data.frame(emmeans(models_final$bias_x, ~ calibration))

emm_slope_method <- as.data.frame(emmeans(models_final$slope_x, ~ method))
emm_slope_camera <- as.data.frame(emmeans(models_final$slope_x, ~ camera))
emm_slope_calib <- as.data.frame(emmeans(models_final$slope_x, ~ calibration))

emm_r_method <- as.data.frame(emmeans(models_final$r_z_x, ~ method)) %>%
  mutate(emmean = z_to_r(emmean), lower.CL = z_to_r(lower.CL), upper.CL = z_to_r(upper.CL))
emm_r_camera <- as.data.frame(emmeans(models_final$r_z_x, ~ camera)) %>%
  mutate(emmean = z_to_r(emmean), lower.CL = z_to_r(lower.CL), upper.CL = z_to_r(upper.CL))
emm_r_calib <- as.data.frame(emmeans(models_final$r_z_x, ~ calibration)) %>%
  mutate(emmean = z_to_r(emmean), lower.CL = z_to_r(lower.CL), upper.CL = z_to_r(upper.CL))
emm_r_interaction <- as.data.frame(emmeans(models_final$r_z_x, ~ method * camera)) %>%
  mutate(emmean = z_to_r(emmean), lower.CL = z_to_r(lower.CL), upper.CL = z_to_r(upper.CL))

# -----------------------------------------------------------------------------
# Plot-Funktion: Haupteffekt
# -----------------------------------------------------------------------------
plot_main_effect <- function(vp_data, emm_data, obs_means, x_var, y_var, x_labels, y_label,
                             ref_line = NULL, sig_label = "ns", y_limits = NULL, show_y_axis = TRUE, sig_y_offset = 0.08) {
  
  vp_data <- vp_data %>% mutate(vp_id = factor(vp_id, levels = vp_ids_original), x_num = as.numeric(factor(.data[[x_var]])))
  emm_data <- emm_data %>% mutate(x_num = as.numeric(factor(.data[[x_var]])))
  obs_means <- obs_means %>% mutate(x_num = as.numeric(factor(.data[[x_var]])))
  
  if (!is.null(y_limits)) { y_max <- y_limits[2]; y_min <- y_limits[1] }
  else { y_max <- max(vp_data[[y_var]], na.rm = TRUE); y_min <- min(vp_data[[y_var]], na.rm = TRUE) }
  y_range <- y_max - y_min
  sig_y <- y_max - y_range * sig_y_offset
  n_levels <- length(unique(vp_data[[x_var]]))
  sig_x <- (1 + n_levels) / 2
  
  p <- ggplot() +
    geom_violin(data = vp_data, aes(x = .data[[x_var]], y = .data[[y_var]]),
                fill = "gray85", color = "gray60", alpha = 0.8, width = 0.28, trim = TRUE, scale = "width") +
    geom_point(data = vp_data, aes(x = .data[[x_var]], y = .data[[y_var]], color = vp_id, shape = vp_id),
               position = position_jitter(width = 0.05, seed = 42), size = 1.2, alpha = 0.9, stroke = 0.3) +
    geom_crossbar(data = obs_means, aes(x = .data[[x_var]], y = .data[[y_var]], ymin = .data[[y_var]], ymax = .data[[y_var]]),
                  width = 0.22, color = "steelblue", linewidth = 0.5, fatten = 0) +
    geom_pointrange(data = emm_data, aes(x = x_num + 0.25, y = emmean, ymin = lower.CL, ymax = upper.CL),
                    color = "#D62728", size = 0.25, linewidth = 0.45) +
    annotate("text", x = sig_x, y = sig_y, label = sig_label, size = 2.5, fontface = "bold") +
    scale_x_discrete(labels = x_labels) +
    scale_color_manual(values = vp_colors, labels = vp_labels, drop = FALSE) +
    scale_shape_manual(values = vp_shapes, labels = vp_labels, drop = FALSE) +
    labs(x = NULL, y = if(show_y_axis) y_label else NULL) +
    theme_apa_fig()
  
  if (!show_y_axis) p <- p + theme(axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())
  if (!is.null(ref_line)) p <- p + geom_hline(yintercept = ref_line, linetype = "dashed", color = "gray40", linewidth = 0.4)
  if (!is.null(y_limits)) p <- p + coord_cartesian(ylim = y_limits)
  
  return(p)
}

# -----------------------------------------------------------------------------
# Plot-Funktion: Interaktion
# -----------------------------------------------------------------------------
plot_interaction_figure <- function(vp_data, emm_data, obs_means, y_var = "r", y_label = "Korrelation (r)",
                                    ref_line = 1, y_limits = NULL) {
  
  vp_data <- vp_data %>% mutate(vp_id = factor(vp_id, levels = vp_ids_original), x_num = as.numeric(factor(method)))
  emm_data <- emm_data %>% mutate(x_num = as.numeric(factor(method)))
  obs_means <- obs_means %>% mutate(x_num = as.numeric(factor(method)))
  
  p <- ggplot() +
    facet_wrap(~ camera, labeller = labeller(camera = labels_de$camera_interaction)) +
    geom_violin(data = vp_data, aes(x = method, y = .data[[y_var]]),
                fill = "gray85", color = "gray60", alpha = 0.8, width = 0.32, scale = "width") +
    geom_point(data = vp_data, aes(x = method, y = .data[[y_var]], color = vp_id, shape = vp_id),
               position = position_jitter(width = 0.05, seed = 42), size = 1.4, alpha = 0.9, stroke = 0.3) +
    geom_crossbar(data = obs_means, aes(x = method, y = .data[[y_var]], ymin = .data[[y_var]], ymax = .data[[y_var]]),
                  width = 0.28, color = "steelblue", linewidth = 0.55, fatten = 0) +
    geom_pointrange(data = emm_data, aes(x = x_num + 0.28, y = emmean, ymin = lower.CL, ymax = upper.CL),
                    color = "#D62728", size = 0.35, linewidth = 0.5) +
    geom_hline(yintercept = ref_line, linetype = "dashed", color = "gray40", linewidth = 0.4) +
    scale_x_discrete(labels = labels_de$method) +
    scale_color_manual(values = vp_colors, labels = vp_labels, drop = FALSE) +
    scale_shape_manual(values = vp_shapes, labels = vp_labels, drop = FALSE) +
    labs(x = NULL, y = y_label) +
    theme_apa_fig(base_size = 10) +
    theme(strip.text = element_text(face = "bold", size = 10))
  
  if (!is.null(y_limits)) p <- p + coord_cartesian(ylim = y_limits)
  return(p)
}

# -----------------------------------------------------------------------------
# Haupteffekt-Plots erstellen
# -----------------------------------------------------------------------------
ylim_bias <- c(-4, 4)
ylim_slope <- c(0.1, 1.15)
ylim_r <- c(0.15, 1.05)

# Row A: Correlation
p_r_method <- plot_main_effect(vp_means_method, emm_r_method, obs_means_method, "method", "r",
                               labels_de$method, "Korrelation (r)", 1, "***", ylim_r, TRUE, 0.02)
p_r_camera <- plot_main_effect(vp_means_camera, emm_r_camera, obs_means_camera, "camera", "r",
                               labels_de$camera, NULL, 1, "***", ylim_r, FALSE, 0.02)
p_r_calib <- plot_main_effect(vp_means_calib, emm_r_calib, obs_means_calib, "calibration", "r",
                              labels_de$calibration, NULL, 1, "ns", ylim_r, FALSE, 0.02)

# Row B: Slope
p_slope_method <- plot_main_effect(vp_means_method, emm_slope_method, obs_means_method, "method", "slope",
                                   labels_de$method, "Steigung", 1, "***", ylim_slope, TRUE)
p_slope_camera <- plot_main_effect(vp_means_camera, emm_slope_camera, obs_means_camera, "camera", "slope",
                                   labels_de$camera, NULL, 1, "***", ylim_slope, FALSE)
p_slope_calib <- plot_main_effect(vp_means_calib, emm_slope_calib, obs_means_calib, "calibration", "slope",
                                  labels_de$calibration, NULL, 1, "*", ylim_slope, FALSE)

# Row C: Bias
p_bias_method <- plot_main_effect(vp_means_method, emm_bias_method, obs_means_method, "method", "bias",
                                  labels_de$method, "Bias (°)", 0, "ns", ylim_bias, TRUE)
p_bias_camera <- plot_main_effect(vp_means_camera, emm_bias_camera, obs_means_camera, "camera", "bias",
                                  labels_de$camera, NULL, 0, "ns", ylim_bias, FALSE)
p_bias_calib <- plot_main_effect(vp_means_calib, emm_bias_calib, obs_means_calib, "calibration", "bias",
                                 labels_de$calibration, NULL, 0, "ns", ylim_bias, FALSE)

# Spalten-Header
create_col_header <- function(title) {
  ggplot() + annotate("text", x = 0.5, y = 0.5, label = title, fontface = "bold", size = 3.2) +
    theme_void() + theme(plot.margin = margin(0, 0, 0, 0))
}

header_method <- create_col_header("Methode")
header_camera <- create_col_header("Kamera")
header_calib <- create_col_header("Kalibrierung")

# Legende
legend_data <- data.frame(vp_id = factor(vp_ids_original, levels = vp_ids_original), x = rep(1:6, 2), y = rep(2:1, each = 6))
p_legend_base <- ggplot(legend_data, aes(x = x, y = y, color = vp_id, shape = vp_id)) +
  geom_point(size = 2.5) +
  scale_color_manual(values = vp_colors, name = NULL, labels = vp_labels) +
  scale_shape_manual(values = vp_shapes, name = NULL, labels = vp_labels) +
  theme_void() +
  theme(legend.position = "bottom", legend.text = element_text(size = 7), legend.key.size = unit(0.35, "cm")) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE), shape = guide_legend(nrow = 2, byrow = TRUE))
legend_grob <- cowplot::get_legend(p_legend_base)

# Row Labels
p_r_method_lab <- p_r_method + labs(tag = "A") + theme(plot.tag = element_text(face = "bold", size = 10), plot.tag.position = c(0.02, 0.98))
p_slope_method_lab <- p_slope_method + labs(tag = "B") + theme(plot.tag = element_text(face = "bold", size = 10), plot.tag.position = c(0.02, 0.98))
p_bias_method_lab <- p_bias_method + labs(tag = "C") + theme(plot.tag = element_text(face = "bold", size = 10), plot.tag.position = c(0.02, 0.98))

# Zusammenfügen
row_header <- (header_method | header_camera | header_calib) + plot_layout(widths = c(1.1, 1, 1))
row_A <- (p_r_method_lab | p_r_camera | p_r_calib) + plot_layout(widths = c(1.1, 1, 1))
row_B <- (p_slope_method_lab | p_slope_camera | p_slope_calib) + plot_layout(widths = c(1.1, 1, 1))
row_C <- (p_bias_method_lab | p_bias_camera | p_bias_calib) + plot_layout(widths = c(1.1, 1, 1))

fig_5_main_effects <- (row_header / row_A / row_B / row_C / wrap_elements(full = legend_grob)) +
  plot_layout(heights = c(0.06, 1, 1, 1, 0.12))

ggsave(file.path(output_dir, "Figure_5_MainEffects.png"), fig_5_main_effects,
       width = 130, height = 165, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(output_dir, "Figure_5_MainEffects.pdf"), fig_5_main_effects,
       width = 130, height = 165, units = "mm", bg = "white")

message("-> Figure_5_MainEffects.png/.pdf gespeichert")

# Interaktionsplot
p_interaction_base <- plot_interaction_figure(vp_means_interaction, emm_r_interaction, obs_means_interaction,
                                              "r", "Korrelation (r)", 1, ylim_r)
fig_6_interaction <- p_interaction_base / wrap_elements(full = legend_grob) + plot_layout(heights = c(1, 0.18))

ggsave(file.path(output_dir, "Figure_6_Interaction.png"), fig_6_interaction,
       width = 130, height = 105, units = "mm", dpi = 300, bg = "white")
ggsave(file.path(output_dir, "Figure_6_Interaction.pdf"), fig_6_interaction,
       width = 130, height = 105, units = "mm", bg = "white")

message("-> Figure_6_Interaction.png/.pdf gespeichert")

# =============================================================================
# 8. DIAGNOSTIK-PLOTS
# =============================================================================
# QQ-Plots und Residuenplots zur Prüfung der Modellannahmen

message("\n=== DIAGNOSTIK-PLOTS ===")

for (av in avs) {
  model <- models_final[[av]]
  
  png(file.path(output_dir, paste0("Figure_Diagnostics_", av_labels[av], ".png")),
      width = 10, height = 8, units = "in", res = 300, bg = "white")
  par(mfrow = c(2, 2), family = "serif")
  
  # Residuen vs. Fitted
  plot(fitted(model), residuals(model),
       xlab = "Fitted", ylab = "Residuals", main = "Residuals vs Fitted")
  abline(h = 0, lty = 2, col = "red")
  
  # QQ-Plot Residuen
  qqnorm(residuals(model), main = "QQ-Plot Residuals")
  qqline(residuals(model), col = "red")
  
  # Scale-Location
  plot(fitted(model), sqrt(abs(residuals(model))),
       xlab = "Fitted", ylab = "√|Residuals|", main = "Scale-Location")
  
  # QQ-Plot Random Effects
  re <- ranef(model)$vp_id[, 1]
  qqnorm(re, main = "QQ-Plot Random Effects")
  qqline(re, col = "red")
  
  dev.off()
  message("-> Figure_Diagnostics_", av_labels[av], ".png gespeichert")
}

# =============================================================================
# 9. SPEICHERN
# =============================================================================

message("\n========== SUMMARY ==========\n")

summary_df <- data.frame(
  DV = sapply(avs, function(av) av_labels[av]),
  `Fixed Effects` = sapply(avs, function(av) selection_results[[av]]$fixed$selected),
  `Random Slope` = sapply(avs, function(av) selection_results[[av]]$random$selected),
  `Sig. Effects` = sapply(avs, function(av) {
    sig <- rownames(results[[av]]$anova)[results[[av]]$anova$`Pr(>F)` < 0.05]
    paste(sig, collapse = ", ")
  }),
  R2m = sapply(avs, function(av) sprintf("%.3f", results[[av]]$r2[1, "R2m"])),
  R2c = sapply(avs, function(av) sprintf("%.3f", results[[av]]$r2[1, "R2c"])),
  check.names = FALSE
)

print(summary_df, row.names = FALSE)

save(results, models_final, selection_results, posthoc_results, icc_results,
     file = file.path(output_dir, "02_lmm_results.RData"))

message("\n=== GENERIERTE DATEIEN ===")
message("Tabellen: Table_ModelSelection.docx, Table_Results.docx, Table_PostHoc.docx")
message("Abbildungen: Figure_5_MainEffects.png/.pdf, Figure_6_Interaction.png/.pdf")
message("Diagnostik: Figure_Diagnostics_*.png")
message("Daten: 02_lmm_results.RData")

message("Finale Objekte: results, models_final, selection_results, posthoc_results, icc_results")

message("\n[OK] LMM-Analyse abgeschlossen.")
message("Weiter mit: 03_RM_ANOVA.R (Robustheitscheck)")
