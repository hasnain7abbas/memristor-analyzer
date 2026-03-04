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
    #[serde(rename = "ccvPercent")]
    pub ccv_percent: f64,
    #[serde(rename = "writeNoise")]
    pub write_noise: f64,
    #[serde(rename = "numLevelsP")]
    pub num_levels_p: usize,
    #[serde(rename = "numLevelsD")]
    pub num_levels_d: usize,
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
}

#[tauri::command]
pub fn extract_parameters(
    potentiation_raw: Vec<f64>,
    depression_raw: Vec<f64>,
    smoothing_config: SmoothingConfigParam,
    v_read: f64,
    is_current: bool,
    multi_cycle_data: Option<Vec<Vec<f64>>>,
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

    // Apply smoothing
    let p_smoothed = apply_smoothing(&p_raw, &smoothing_config)?;
    let d_smoothed = apply_smoothing(&d_raw, &smoothing_config)?;

    // Find Gmin, Gmax
    let g_min = p_smoothed
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let g_max = p_smoothed
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let on_off_ratio = if g_min.abs() > 1e-15 {
        g_max / g_min
    } else {
        f64::INFINITY
    };
    let dynamic_range = g_max - g_min;

    // Fit alpha for potentiation
    let (alpha_p, r_squared_p, p_fitted) = fit_alpha(&p_smoothed, g_min, g_max, true);

    // Fit alpha for depression
    let (alpha_d, r_squared_d, d_fitted) = fit_alpha(&d_smoothed, g_min, g_max, false);

    // Compute CCV and write noise
    let (ccv_percent, write_noise) =
        compute_ccv_noise(&p_smoothed, &d_smoothed, g_max - g_min, &multi_cycle_data);

    // Build delta G scatter data
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

    Ok(ExtractedParams {
        g_min,
        g_max,
        on_off_ratio,
        dynamic_range,
        alpha_p,
        alpha_d,
        r_squared_p,
        r_squared_d,
        ccv_percent,
        write_noise,
        num_levels_p: p_raw.len(),
        num_levels_d: d_raw.len(),
        potentiation_raw: p_raw,
        potentiation_smoothed: p_smoothed,
        potentiation_fitted: p_fitted,
        depression_raw: d_raw,
        depression_smoothed: d_smoothed,
        depression_fitted: d_fitted,
        delta_g,
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
    )
}

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

    // Coarse grid search
    let mut best_alpha = 0.01f64;
    let mut best_sse = f64::INFINITY;

    let mut alpha = 0.01;
    while alpha <= 12.0 {
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

    // Fine grid search
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

fn compute_ccv_noise(
    p_data: &[f64],
    d_data: &[f64],
    dynamic_range: f64,
    multi_cycle: &Option<Vec<Vec<f64>>>,
) -> (f64, f64) {
    let delta_g: Vec<f64> = match multi_cycle {
        Some(cycles) => {
            let mut all_dg = Vec::new();
            for cycle in cycles {
                for i in 0..cycle.len().saturating_sub(1) {
                    all_dg.push(cycle[i + 1] - cycle[i]);
                }
            }
            all_dg
        }
        None => {
            let mut dg = Vec::new();
            for i in 0..p_data.len().saturating_sub(1) {
                dg.push(p_data[i + 1] - p_data[i]);
            }
            for i in 0..d_data.len().saturating_sub(1) {
                dg.push(d_data[i + 1] - d_data[i]);
            }
            dg
        }
    };

    if delta_g.is_empty() {
        return (0.0, 0.0);
    }

    let abs_dg: Vec<f64> = delta_g.iter().map(|x| x.abs()).collect();
    let n = delta_g.len() as f64;

    let mean_abs = abs_dg.iter().sum::<f64>() / n;
    let std_abs = if delta_g.len() > 1 {
        let variance =
            abs_dg.iter().map(|x| (x - mean_abs).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    let ccv = if mean_abs.abs() > 1e-15 {
        std_abs / mean_abs * 100.0
    } else {
        0.0
    };

    let mean_dg = delta_g.iter().sum::<f64>() / n;
    let std_dg = if delta_g.len() > 1 {
        let variance = delta_g
            .iter()
            .map(|x| (x - mean_dg).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    let write_noise = if dynamic_range.abs() > 1e-15 {
        std_dg / dynamic_range
    } else {
        0.0
    };

    (ccv, write_noise)
}
