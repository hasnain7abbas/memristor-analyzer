#[tauri::command]
pub fn smooth_data(
    data: Vec<f64>,
    method: String,
    window_size: usize,
    poly_order: Option<usize>,
    remove_outliers: bool,
    outlier_sigma: f64,
    strength: Option<f64>,
    enforce_monotonic: Option<bool>,
    monotonic_direction: Option<String>,
    bandwidth: Option<f64>,
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

    let str_val = strength.unwrap_or(1.0).clamp(0.0, 1.0);
    let mono = enforce_monotonic.unwrap_or(false);
    let mono_dir = monotonic_direction.unwrap_or_else(|| "auto".to_string());
    let bw = bandwidth.unwrap_or(0.3).clamp(0.05, 0.9);

    // Step 1: outlier removal
    let cleaned = if remove_outliers {
        remove_outliers_fn(&data, outlier_sigma)
    } else {
        data.clone()
    };

    // Step 2: apply smoothing
    let smoothed = match method.as_str() {
        "none" => cleaned.clone(),
        "moving_avg" => moving_average(&cleaned, ws),
        "median" => median_filter(&cleaned, ws),
        "savitzky_golay" => {
            let po = poly_order.unwrap_or(2);
            if po >= ws {
                return Err("Polynomial order must be less than window size".to_string());
            }
            savitzky_golay(&cleaned, ws, po)
        }
        "loess" => loess(&cleaned, bw),
        "gaussian" => gaussian_kernel(&cleaned, ws),
        _ => return Err(format!("Unknown smoothing method: {}", method)),
    };

    // Step 3: apply strength blending (mix raw and smoothed)
    let blended = if str_val < 1.0 && method != "none" {
        cleaned
            .iter()
            .zip(smoothed.iter())
            .map(|(raw, sm)| raw * (1.0 - str_val) + sm * str_val)
            .collect()
    } else {
        smoothed
    };

    // Step 4: enforce monotonicity if requested
    let result = if mono && method != "none" {
        let direction = match mono_dir.as_str() {
            "increasing" => MonoDirection::Increasing,
            "decreasing" => MonoDirection::Decreasing,
            _ => {
                // Auto-detect: check if data generally increases or decreases
                if blended.last().unwrap_or(&0.0) >= blended.first().unwrap_or(&0.0) {
                    MonoDirection::Increasing
                } else {
                    MonoDirection::Decreasing
                }
            }
        };
        isotonic_regression(&blended, direction)
    } else {
        blended
    };

    Ok(result)
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

    // Create reflected boundary-padded data to fix edge discontinuity
    let mut padded = Vec::with_capacity(n + 2 * half);
    for i in (1..=half).rev() {
        let idx = i.min(n - 1);
        padded.push(2.0 * data[0] - data[idx]);
    }
    padded.extend_from_slice(data);
    for i in 1..=half {
        let idx = (n - 1).saturating_sub(i);
        padded.push(2.0 * data[n - 1] - data[idx]);
    }

    // Apply SG filter using padded data — full window always available
    for i in 0..n {
        let pi = i + half; // index in padded array
        let mut val = 0.0;
        for j in 0..window_size {
            val += coeffs[j] * padded[pi - half + j];
        }
        result.push(val);
    }

    result
}

/// LOESS/LOWESS smoothing — locally weighted scatterplot smoothing
fn loess(data: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = data.len();
    if n < 3 {
        return data.to_vec();
    }

    let span = (bandwidth * n as f64).max(3.0).min(n as f64) as usize;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let x_i = i as f64;

        // Find the `span` nearest neighbors
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|j| (j, (j as f64 - x_i).abs()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors = &dists[..span.min(n)];

        let max_dist = neighbors.last().map(|(_, d)| *d).unwrap_or(1.0).max(1e-10);

        // Tricube weight function: w(u) = (1 - |u|^3)^3 for |u| < 1
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wxx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wxy = 0.0;

        for &(j, dist) in neighbors {
            let u = dist / max_dist;
            let w = if u < 1.0 {
                let t = 1.0 - u * u * u;
                t * t * t
            } else {
                0.0
            };

            let xj = j as f64 - x_i;
            let yj = data[j];

            sum_w += w;
            sum_wx += w * xj;
            sum_wxx += w * xj * xj;
            sum_wy += w * yj;
            sum_wxy += w * xj * yj;
        }

        // Weighted linear regression: y = a + b*x at x=0
        let det = sum_w * sum_wxx - sum_wx * sum_wx;
        let fitted = if det.abs() > 1e-15 {
            (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
        } else {
            sum_wy / sum_w.max(1e-15)
        };

        result.push(fitted);
    }

    result
}

/// Gaussian kernel smoothing
fn gaussian_kernel(data: &[f64], window_size: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return data.to_vec();
    }

    let sigma = window_size as f64 / 4.0; // window covers ~2 sigma on each side
    let half = window_size / 2;
    let mut result = Vec::with_capacity(n);

    // Precompute kernel weights
    let mut kernel = Vec::with_capacity(window_size);
    for j in 0..window_size {
        let x = (j as f64 - half as f64) / sigma;
        kernel.push((-0.5 * x * x).exp());
    }

    for i in 0..n {
        let mut sum = 0.0;
        let mut wsum = 0.0;
        for j in 0..window_size {
            let idx = i as isize + j as isize - half as isize;
            // Reflected boundary
            let idx = if idx < 0 {
                (-idx) as usize
            } else if idx >= n as isize {
                2 * n - 2 - idx as usize
            } else {
                idx as usize
            };
            let idx = idx.min(n - 1);
            sum += kernel[j] * data[idx];
            wsum += kernel[j];
        }
        result.push(sum / wsum.max(1e-15));
    }

    result
}

#[derive(Clone, Copy)]
enum MonoDirection {
    Increasing,
    Decreasing,
}

/// Isotonic regression — pool adjacent violators algorithm
fn isotonic_regression(data: &[f64], direction: MonoDirection) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return data.to_vec();
    }

    let is_increasing = matches!(direction, MonoDirection::Increasing);

    // Pool adjacent violators
    let mut blocks: Vec<(f64, usize)> = Vec::new(); // (mean, count)
    for &val in data {
        blocks.push((val, 1));
        while blocks.len() >= 2 {
            let len = blocks.len();
            let should_merge = if is_increasing {
                blocks[len - 2].0 > blocks[len - 1].0
            } else {
                blocks[len - 2].0 < blocks[len - 1].0
            };
            if should_merge {
                let b = blocks.pop().unwrap();
                let a = blocks.last_mut().unwrap();
                let total = a.1 + b.1;
                a.0 = (a.0 * a.1 as f64 + b.0 * b.1 as f64) / total as f64;
                a.1 = total;
            } else {
                break;
            }
        }
    }

    // Expand blocks back to full array
    let mut result = Vec::with_capacity(n);
    for (mean, count) in blocks {
        for _ in 0..count {
            result.push(mean);
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
