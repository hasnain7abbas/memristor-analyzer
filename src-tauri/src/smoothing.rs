#[tauri::command]
pub fn smooth_data(
    data: Vec<f64>,
    method: String,
    window_size: usize,
    poly_order: Option<usize>,
    remove_outliers: bool,
    outlier_sigma: f64,
) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Ok(vec![]);
    }

    // Ensure window_size is odd and >= 3
    let ws = if window_size < 3 {
        3
    } else if window_size % 2 == 0 {
        window_size + 1
    } else {
        window_size
    };

    // Step 1: outlier removal
    let cleaned = if remove_outliers {
        remove_outliers_fn(&data, outlier_sigma)
    } else {
        data.clone()
    };

    // Step 2: apply smoothing
    match method.as_str() {
        "none" => Ok(cleaned),
        "moving_avg" => Ok(moving_average(&cleaned, ws)),
        "median" => Ok(median_filter(&cleaned, ws)),
        "savitzky_golay" => {
            let po = poly_order.unwrap_or(2);
            if po >= ws {
                return Err("Polynomial order must be less than window size".to_string());
            }
            Ok(savitzky_golay(&cleaned, ws, po))
        }
        _ => Err(format!("Unknown smoothing method: {}", method)),
    }
}

fn remove_outliers_fn(data: &[f64], sigma_threshold: f64) -> Vec<f64> {
    let n = data.len();
    if n < 3 {
        return data.to_vec();
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = variance.sqrt();

    if std < 1e-15 {
        return data.to_vec();
    }

    let mut result = data.to_vec();
    for i in 0..n {
        if (result[i] - mean).abs() > sigma_threshold * std {
            if i == 0 {
                result[i] = result[1];
            } else if i == n - 1 {
                result[i] = result[n - 2];
            } else {
                result[i] = (data[i - 1] + data[i + 1]) / 2.0;
            }
        }
    }
    result
}

fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    let n = data.len();
    let half = window_size / 2;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let sum: f64 = data[start..end].iter().sum();
        result.push(sum / (end - start) as f64);
    }
    result
}

fn median_filter(data: &[f64], window_size: usize) -> Vec<f64> {
    let n = data.len();
    let half = window_size / 2;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let mut window: Vec<f64> = data[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = window.len() / 2;
        let median = if window.len() % 2 == 0 {
            (window[mid - 1] + window[mid]) / 2.0
        } else {
            window[mid]
        };
        result.push(median);
    }
    result
}

fn savitzky_golay(data: &[f64], window_size: usize, poly_order: usize) -> Vec<f64> {
    let n = data.len();
    if n < window_size {
        return data.to_vec();
    }
    let half = window_size / 2;
    let mut result = Vec::with_capacity(n);

    // Precompute SG coefficients for the center window
    let coeffs = sg_coefficients(window_size, poly_order);

    for i in 0..n {
        if i >= half && i + half < n {
            // Full window available - use precomputed coefficients
            let mut val = 0.0;
            for j in 0..window_size {
                val += coeffs[j] * data[i - half + j];
            }
            result.push(val);
        } else {
            // Edge: use asymmetric local polynomial fit
            let start = if i >= half { i - half } else { 0 };
            let end = (i + half + 1).min(n);
            let local: Vec<f64> = data[start..end].to_vec();
            let local_idx = i - start;
            let fitted = local_poly_fit(&local, poly_order, local_idx);
            result.push(fitted);
        }
    }
    result
}

fn sg_coefficients(window_size: usize, poly_order: usize) -> Vec<f64> {
    let half = window_size as isize / 2;
    let m = poly_order + 1;

    // Build Vandermonde matrix
    let mut x_vals: Vec<f64> = Vec::with_capacity(window_size);
    for i in -half..=half {
        x_vals.push(i as f64);
    }

    // J^T J and J^T e_0  (we want the 0th row of (J^T J)^-1 J^T)
    // Equivalently, solve for coefficients and evaluate at x=0
    let mut jtj = vec![0.0f64; m * m];
    let mut jt_cols = vec![vec![0.0f64; window_size]; m];

    for (k, &x) in x_vals.iter().enumerate() {
        let mut xp = 1.0;
        for i in 0..m {
            jt_cols[i][k] = xp;
            let mut xp2 = 1.0;
            for j in 0..m {
                jtj[i * m + j] += xp * xp2;
                xp2 *= x;
            }
            xp *= x;
        }
    }

    // Solve jtj * c = e_0 for the smoothing coefficients
    // e_0 = [1, 0, 0, ...] since we want the value at x=0 (constant term)
    let c = solve_linear_system(&jtj, &{
        let mut e = vec![0.0; m];
        e[0] = 1.0;
        e
    }, m);

    // Smoothing coefficients = sum_i c[i] * J^T[i][k]
    let mut coeffs = vec![0.0; window_size];
    for k in 0..window_size {
        for i in 0..m {
            coeffs[k] += c[i] * jt_cols[i][k];
        }
    }

    coeffs
}

fn local_poly_fit(data: &[f64], poly_order: usize, eval_idx: usize) -> f64 {
    let n = data.len();
    let po = poly_order.min(n - 1);
    let m = po + 1;

    let mut jtj = vec![0.0f64; m * m];
    let mut jty = vec![0.0f64; m];

    for k in 0..n {
        let x = k as f64;
        let y = data[k];
        let mut xp = 1.0;
        for i in 0..m {
            jty[i] += xp * y;
            let mut xp2 = 1.0;
            for j in 0..m {
                jtj[i * m + j] += xp * xp2;
                xp2 *= x;
            }
            xp *= x;
        }
    }

    let c = solve_linear_system(&jtj, &jty, m);

    let x_eval = eval_idx as f64;
    let mut result = 0.0;
    let mut xp = 1.0;
    for ci in c.iter().take(m) {
        result += ci * xp;
        xp *= x_eval;
    }
    result
}

fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Gaussian elimination with partial pivoting
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        if diag.abs() > 1e-15 {
            x[i] = sum / diag;
        }
    }
    x
}
