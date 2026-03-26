use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SmoothingConfigParam {
    pub method: String,
    pub window_size: usize,
    pub poly_order: Option<usize>,
    pub remove_outliers: bool,
    pub outlier_sigma: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaGPoint {
    #[serde(rename = "G")]
    pub g: f64,
    #[serde(rename = "dG")]
    pub dg: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExtractedParams {
    #[serde(rename = "Gmin")]
    pub g_min: f64,
    #[serde(rename = "Gmax")]
    pub g_max: f64,
    #[serde(rename = "GminStd")]
    pub g_min_std: f64,
    #[serde(rename = "GmaxStd")]
    pub g_max_std: f64,
    #[serde(rename = "onOffRatio")]
    pub on_off_ratio: f64,
    #[serde(rename = "dynamicRange")]
    pub dynamic_range: f64,
    #[serde(rename = "alphaP")]
    pub alpha_p: f64,
    #[serde(rename = "alphaD")]
    pub alpha_d: f64,
    #[serde(rename = "rSquaredP")]
    pub r_squared_p: f64,
    #[serde(rename = "rSquaredD")]
    pub r_squared_d: f64,
    #[serde(rename = "alphaPPercycle", skip_serializing_if = "Option::is_none")]
    pub alpha_p_percycle: Option<f64>,
    #[serde(rename = "alphaDPercycle", skip_serializing_if = "Option::is_none")]
    pub alpha_d_percycle: Option<f64>,
    #[serde(rename = "alphaPPercycleStd", skip_serializing_if = "Option::is_none")]
    pub alpha_p_percycle_std: Option<f64>,
    #[serde(rename = "alphaDPercycleStd", skip_serializing_if = "Option::is_none")]
    pub alpha_d_percycle_std: Option<f64>,
    #[serde(rename = "ccvPercent")]
    pub ccv_percent: f64,
    #[serde(rename = "ccvPotentiation")]
    pub ccv_potentiation: f64,
    #[serde(rename = "ccvDepression")]
    pub ccv_depression: f64,
    #[serde(rename = "writeNoise")]
    pub write_noise: f64,
    #[serde(rename = "numLevelsP")]
    pub num_levels_p: usize,
    #[serde(rename = "numLevelsD")]
    pub num_levels_d: usize,
    #[serde(rename = "weightBits")]
    pub weight_bits: usize,
    #[serde(rename = "potentiationRaw")]
    pub potentiation_raw: Vec<f64>,
    #[serde(rename = "potentiationSmoothed")]
    pub potentiation_smoothed: Vec<f64>,
    #[serde(rename = "potentiationFitted")]
    pub potentiation_fitted: Vec<f64>,
    #[serde(rename = "depressionRaw")]
    pub depression_raw: Vec<f64>,
    #[serde(rename = "depressionSmoothed")]
    pub depression_smoothed: Vec<f64>,
    #[serde(rename = "depressionFitted")]
    pub depression_fitted: Vec<f64>,
    #[serde(rename = "deltaG")]
    pub delta_g: Vec<DeltaGPoint>,
    #[serde(rename = "memoryWindow")]
    pub memory_window: f64,
    #[serde(rename = "programmingMargin")]
    pub programming_margin: f64,
    #[serde(rename = "asymmetryIndex")]
    pub asymmetry_index: f64,
    #[serde(rename = "switchingUniformity")]
    pub switching_uniformity: f64,
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Standard deviation with Bessel's correction (ddof=1).
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    var.sqrt()
}

/// CCV% from a set of step sizes: std(|steps|, ddof=1) / mean(|steps|) × 100
fn ccv_from_steps(steps: &[f64]) -> f64 {
    if steps.is_empty() {
        return 0.0;
    }
    let abs_steps: Vec<f64> = steps.iter().map(|x| x.abs()).collect();
    let mean_abs = abs_steps.iter().sum::<f64>() / abs_steps.len() as f64;
    if mean_abs.abs() < 1e-15 {
        return 0.0;
    }
    let s = std_dev(&abs_steps);
    s / mean_abs * 100.0
}

// ─── Main extraction ─────────────────────────────────────────────────

#[tauri::command]
pub fn extract_parameters(
    potentiation_raw: Vec<f64>,
    depression_raw: Vec<f64>,
    smoothing_config: SmoothingConfigParam,
    v_read: f64,
    is_current: bool,
    _multi_cycle_data: Option<Vec<Vec<f64>>>,
    potentiation_cycles: Option<Vec<Vec<f64>>>,
    depression_cycles: Option<Vec<Vec<f64>>>,
) -> Result<ExtractedParams, String> {
    if potentiation_raw.is_empty() || depression_raw.is_empty() {
        return Err("Potentiation and depression data cannot be empty".to_string());
    }

    // Convert current to conductance if needed
    let p_raw: Vec<f64> = if is_current {
        if v_read.abs() < 1e-15 {
            return Err("V_read cannot be zero for current-to-conductance conversion".to_string());
        }
        potentiation_raw.iter().map(|i| i / v_read).collect()
    } else {
        potentiation_raw.clone()
    };

    let d_raw: Vec<f64> = if is_current {
        depression_raw.iter().map(|i| i / v_read).collect()
    } else {
        depression_raw.clone()
    };

    // Apply smoothing to averaged curves
    let p_smoothed = apply_smoothing(&p_raw, &smoothing_config)?;
    let d_smoothed = apply_smoothing(&d_raw, &smoothing_config)?;

    // ─── Step 3: G_min and G_max ───────────────────────────────────────
    // G_max = mean of end-of-potentiation values across cycles
    // G_min = mean of end-of-depression values across cycles
    let (g_min, g_max, g_min_std, g_max_std) =
        if let (Some(ref p_cyc), Some(ref d_cyc)) = (&potentiation_cycles, &depression_cycles) {
            let pot_ends: Vec<f64> = p_cyc.iter().filter_map(|c| c.last().copied()).collect();
            let dep_ends: Vec<f64> = d_cyc.iter().filter_map(|c| c.last().copied()).collect();

            if pot_ends.is_empty() || dep_ends.is_empty() {
                let gmax = *p_smoothed.last().unwrap_or(&0.0);
                let gmin = *d_smoothed.last().unwrap_or(&0.0);
                (gmin, gmax, 0.0, 0.0)
            } else {
                let gmax = pot_ends.iter().sum::<f64>() / pot_ends.len() as f64;
                let gmin = dep_ends.iter().sum::<f64>() / dep_ends.len() as f64;
                (gmin, gmax, std_dev(&dep_ends), std_dev(&pot_ends))
            }
        } else {
            // Single cycle: use end-of-phase from smoothed curves
            let gmax = *p_smoothed.last().unwrap_or(&0.0);
            let gmin = *d_smoothed.last().unwrap_or(&0.0);
            (gmin, gmax, 0.0, 0.0)
        };

    let on_off_ratio = if g_min.abs() > 1e-15 {
        g_max / g_min
    } else {
        f64::INFINITY
    };
    let dynamic_range = g_max - g_min;

    let memory_window = if g_min > 1e-15 {
        20.0 * (g_max / g_min).log10()
    } else {
        0.0
    };

    let programming_margin = if (g_max + g_min).abs() > 1e-15 {
        (g_max - g_min) / (g_max + g_min) * 100.0
    } else {
        0.0
    };

    // ─── Step 4: Alpha fitting (averaged curve) ────────────────────────
    let (alpha_p, r_squared_p, p_fitted) = fit_alpha(&p_smoothed, g_min, g_max, true);
    let (alpha_d, r_squared_d, d_fitted) = fit_alpha(&d_smoothed, g_min, g_max, false);

    // Per-cycle alpha fitting
    let (alpha_p_percycle, alpha_d_percycle, alpha_p_percycle_std, alpha_d_percycle_std) =
        if let (Some(ref p_cyc), Some(ref d_cyc)) = (&potentiation_cycles, &depression_cycles) {
            let mut ap_vals = Vec::new();
            for cycle in p_cyc {
                if cycle.len() < 2 {
                    continue;
                }
                let c_min = cycle.iter().cloned().fold(f64::INFINITY, f64::min);
                let c_max = cycle.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let (a, _, _) = fit_alpha(cycle, c_min, c_max, true);
                ap_vals.push(a);
            }
            let mut ad_vals = Vec::new();
            for cycle in d_cyc {
                if cycle.len() < 2 {
                    continue;
                }
                let c_min = cycle.iter().cloned().fold(f64::INFINITY, f64::min);
                let c_max = cycle.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let (a, _, _) = fit_alpha(cycle, c_min, c_max, false);
                ad_vals.push(a);
            }

            if ap_vals.is_empty() || ad_vals.is_empty() {
                (None, None, None, None)
            } else {
                let ap_mean = ap_vals.iter().sum::<f64>() / ap_vals.len() as f64;
                let ad_mean = ad_vals.iter().sum::<f64>() / ad_vals.len() as f64;
                (
                    Some(ap_mean),
                    Some(ad_mean),
                    Some(std_dev(&ap_vals)),
                    Some(std_dev(&ad_vals)),
                )
            }
        } else {
            (None, None, None, None)
        };

    // ─── Step 5: CCV and Write Noise ───────────────────────────────────
    let (ccv_percent, write_noise, ccv_potentiation, ccv_depression) =
        compute_ccv_noise(
            &p_smoothed,
            &d_smoothed,
            dynamic_range,
            &potentiation_cycles,
            &depression_cycles,
        );

    // ─── Step 6: Distinguishable Levels ────────────────────────────────
    let num_levels_p = compute_num_levels(&p_smoothed);
    let num_levels_d = compute_num_levels(&d_smoothed);
    let weight_bits = {
        let min_levels = num_levels_p.min(num_levels_d).max(1);
        (min_levels as f64).log2().floor() as usize
    };

    // ─── Step 7: Asymmetry Index ───────────────────────────────────────
    // Use per-cycle alpha when averaged-curve R² < 0
    let effective_alpha_p = if r_squared_p < 0.0 {
        alpha_p_percycle.unwrap_or(alpha_p)
    } else {
        alpha_p
    };
    let effective_alpha_d = if r_squared_d < 0.0 {
        alpha_d_percycle.unwrap_or(alpha_d)
    } else {
        alpha_d
    };
    let asymmetry_index = if effective_alpha_p.max(effective_alpha_d) > 1e-10 {
        (effective_alpha_p - effective_alpha_d).abs()
            / effective_alpha_p.max(effective_alpha_d)
    } else {
        0.0
    };

    // ─── Step 9: Delta-G scatter data ──────────────────────────────────
    let mut delta_g = Vec::new();
    for i in 0..p_smoothed.len().saturating_sub(1) {
        delta_g.push(DeltaGPoint {
            g: p_smoothed[i],
            dg: p_smoothed[i + 1] - p_smoothed[i],
        });
    }
    for i in 0..d_smoothed.len().saturating_sub(1) {
        delta_g.push(DeltaGPoint {
            g: d_smoothed[i],
            dg: d_smoothed[i + 1] - d_smoothed[i],
        });
    }

    // ─── Step 8: Switching Uniformity ──────────────────────────────────
    let switching_uniformity = compute_switching_uniformity(&delta_g);

    Ok(ExtractedParams {
        g_min,
        g_max,
        g_min_std,
        g_max_std,
        on_off_ratio,
        dynamic_range,
        alpha_p,
        alpha_d,
        r_squared_p,
        r_squared_d,
        alpha_p_percycle,
        alpha_d_percycle,
        alpha_p_percycle_std,
        alpha_d_percycle_std,
        ccv_percent,
        ccv_potentiation,
        ccv_depression,
        write_noise,
        num_levels_p,
        num_levels_d,
        weight_bits,
        potentiation_raw: p_raw,
        potentiation_smoothed: p_smoothed,
        potentiation_fitted: p_fitted,
        depression_raw: d_raw,
        depression_smoothed: d_smoothed,
        depression_fitted: d_fitted,
        delta_g,
        memory_window,
        programming_margin,
        asymmetry_index,
        switching_uniformity,
    })
}

fn apply_smoothing(data: &[f64], config: &SmoothingConfigParam) -> Result<Vec<f64>, String> {
    crate::smoothing::smooth_data(
        data.to_vec(),
        config.method.clone(),
        config.window_size,
        config.poly_order,
        config.remove_outliers,
        config.outlier_sigma,
        None, // strength
        None, // enforce_monotonic
        None, // monotonic_direction
        None, // bandwidth
    )
}

// ─── Nonlinear model ─────────────────────────────────────────────────

fn nonlinear_model(n: f64, n_total: f64, g_start: f64, g_end: f64, alpha: f64) -> f64 {
    if alpha.abs() < 1e-10 {
        // Linear case
        g_start + (g_end - g_start) * n / n_total
    } else {
        let exp_term = (-alpha * n / n_total).exp();
        let norm = 1.0 - (-alpha).exp();
        if norm.abs() < 1e-15 {
            g_start + (g_end - g_start) * n / n_total
        } else {
            g_start + (g_end - g_start) * (1.0 - exp_term) / norm
        }
    }
}

// ─── Alpha fitting ───────────────────────────────────────────────────

fn fit_alpha(
    data: &[f64],
    g_min: f64,
    g_max: f64,
    is_potentiation: bool,
) -> (f64, f64, Vec<f64>) {
    let n = data.len();
    if n < 2 {
        return (0.0, 0.0, data.to_vec());
    }

    let n_total = (n - 1) as f64;
    let (g_start, g_end) = if is_potentiation {
        (g_min, g_max)
    } else {
        (g_max, g_min)
    };

    // Coarse grid search: α from 0.01 to 15.0, step 0.05
    let mut best_alpha = 0.01f64;
    let mut best_sse = f64::INFINITY;

    let mut alpha = 0.01;
    while alpha <= 15.0 {
        let sse: f64 = (0..n)
            .map(|i| {
                let pred = nonlinear_model(i as f64, n_total, g_start, g_end, alpha);
                (data[i] - pred).powi(2)
            })
            .sum();
        if sse < best_sse {
            best_sse = sse;
            best_alpha = alpha;
        }
        alpha += 0.05;
    }

    // Fine grid search: best_alpha ± 0.3, step 0.001
    let fine_start = (best_alpha - 0.3).max(0.001);
    let fine_end = best_alpha + 0.3;
    alpha = fine_start;
    while alpha <= fine_end {
        let sse: f64 = (0..n)
            .map(|i| {
                let pred = nonlinear_model(i as f64, n_total, g_start, g_end, alpha);
                (data[i] - pred).powi(2)
            })
            .sum();
        if sse < best_sse {
            best_sse = sse;
            best_alpha = alpha;
        }
        alpha += 0.001;
    }

    // Generate fitted curve
    let fitted: Vec<f64> = (0..n)
        .map(|i| nonlinear_model(i as f64, n_total, g_start, g_end, best_alpha))
        .collect();

    // Compute R²
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    let ss_res: f64 = data
        .iter()
        .zip(fitted.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum();

    let r_squared = if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (best_alpha, r_squared, fitted)
}

// ─── CCV and write noise ─────────────────────────────────────────────

fn compute_ccv_noise(
    p_smoothed: &[f64],
    d_smoothed: &[f64],
    dynamic_range: f64,
    p_cycles: &Option<Vec<Vec<f64>>>,
    d_cycles: &Option<Vec<Vec<f64>>>,
) -> (f64, f64, f64, f64) {
    // Collect ALL step sizes from per-cycle data if available
    let (pot_steps, dep_steps) =
        if let (Some(pc), Some(dc)) = (p_cycles, d_cycles) {
            let mut ps = Vec::new();
            for cycle in pc {
                for i in 0..cycle.len().saturating_sub(1) {
                    ps.push(cycle[i + 1] - cycle[i]);
                }
            }
            let mut ds = Vec::new();
            for cycle in dc {
                for i in 0..cycle.len().saturating_sub(1) {
                    ds.push(cycle[i + 1] - cycle[i]);
                }
            }
            (ps, ds)
        } else {
            // Fallback: use smoothed single-cycle data
            let ps: Vec<f64> = p_smoothed.windows(2).map(|w| w[1] - w[0]).collect();
            let ds: Vec<f64> = d_smoothed.windows(2).map(|w| w[1] - w[0]).collect();
            (ps, ds)
        };

    let all_delta_g: Vec<f64> = pot_steps
        .iter()
        .chain(dep_steps.iter())
        .copied()
        .collect();

    if all_delta_g.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Combined CCV
    let ccv = ccv_from_steps(&all_delta_g);
    // Separate CCV for P and D
    let ccv_p = ccv_from_steps(&pot_steps);
    let ccv_d = ccv_from_steps(&dep_steps);

    // Write noise: σ_w = std(all_delta_g, ddof=1) / dynamic_range
    let write_noise = if dynamic_range.abs() > 1e-15 {
        std_dev(&all_delta_g) / dynamic_range
    } else {
        0.0
    };

    (ccv, write_noise, ccv_p, ccv_d)
}

// ─── Distinguishable levels ──────────────────────────────────────────

fn compute_num_levels(data: &[f64]) -> usize {
    if data.len() < 3 {
        return data.len().max(2);
    }

    let g_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = g_max - g_min;

    if range < 1e-15 {
        return 2;
    }

    let steps: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    if steps.is_empty() {
        return 2;
    }

    let mean_step = steps.iter().sum::<f64>() / steps.len() as f64;
    let std_step = std_dev(&steps);

    // effective_step = mean_step + 2 * sigma (2-sigma margin)
    let effective_step = mean_step + 2.0 * std_step;
    if effective_step < 1e-15 {
        return data.len().min(256).max(2);
    }

    let levels = (range / effective_step).round() as usize;
    levels.max(2).min(256)
}

// ─── Switching uniformity ────────────────────────────────────────────

fn compute_switching_uniformity(delta_g: &[DeltaGPoint]) -> f64 {
    if delta_g.len() < 2 {
        return 1.0;
    }

    let n = delta_g.len() as f64;
    let mean_g = delta_g.iter().map(|p| p.g).sum::<f64>() / n;
    let mean_dg = delta_g.iter().map(|p| p.dg).sum::<f64>() / n;

    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    for p in delta_g {
        ss_xx += (p.g - mean_g).powi(2);
        ss_xy += (p.g - mean_g) * (p.dg - mean_dg);
    }

    let slope = if ss_xx.abs() > 1e-15 {
        ss_xy / ss_xx
    } else {
        0.0
    };

    (1.0 - slope.abs()).clamp(0.0, 1.0)
}
