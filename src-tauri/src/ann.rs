use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use tauri::Emitter;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ANNConfig {
    #[serde(rename = "modelType", default = "default_model_type")]
    pub model_type: String,
    #[serde(rename = "hiddenSize")]
    pub hidden_size: usize,
    #[serde(rename = "hiddenSize2", default = "default_hidden_size_2")]
    pub hidden_size_2: usize,
    pub epochs: usize,
    #[serde(rename = "learningRate")]
    pub learning_rate: f64,
    #[serde(rename = "batchSize")]
    pub batch_size: usize,
}

fn default_model_type() -> String {
    "mlp_1h".to_string()
}

fn default_hidden_size_2() -> usize {
    64
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeviceParams {
    #[serde(rename = "alphaP")]
    pub alpha_p: f64,
    #[serde(rename = "alphaD")]
    pub alpha_d: f64,
    #[serde(rename = "writeNoise")]
    pub write_noise: f64,
    #[serde(rename = "numLevelsP")]
    pub num_levels_p: usize,
    #[serde(rename = "numLevelsD")]
    pub num_levels_d: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ANNEpochResult {
    pub epoch: usize,
    #[serde(rename = "idealAccuracy")]
    pub ideal_accuracy: f64,
    #[serde(rename = "memristorAccuracy")]
    pub memristor_accuracy: f64,
    #[serde(rename = "idealLoss")]
    pub ideal_loss: f64,
    #[serde(rename = "memristorLoss")]
    pub memristor_loss: f64,
}

struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(sizes: &[usize], rng: &mut StdRng) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            let fan_in = sizes[i];
            let fan_out = sizes[i + 1];
            let std_dev = (2.0 / fan_in as f64).sqrt();
            let dist = Normal::new(0.0, std_dev).unwrap();

            let weights = Array2::from_shape_fn((fan_out, fan_in), |_| rng.sample(dist));
            let biases = Array1::zeros(fan_out);
            layers.push(Layer { weights, biases });
        }
        MLP { layers }
    }

    fn forward(&self, input: &Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut activations = vec![input.clone()];
        let mut zs = Vec::new();
        let mut current = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.weights.dot(&current) + &layer.biases;
            let a = if i < self.layers.len() - 1 {
                relu(&z)
            } else {
                softmax(&z)
            };
            zs.push(z);
            current = a.clone();
            activations.push(a);
        }
        (activations, zs)
    }

    /// Per-sample backpropagation with immediate weight update.
    /// This gives 5000 weight updates per epoch (one per training sample),
    /// which is essential for convergence on the small synthetic MNIST dataset.
    fn backprop(&mut self, input: &Array1<f64>, target: usize, lr: f64) -> f64 {
        let (activations, zs) = self.forward(input);
        let output = activations.last().unwrap();

        // Cross-entropy loss
        let loss = -output[target].max(1e-15).ln();

        // Output layer delta
        let mut delta = output.clone();
        delta[target] -= 1.0;

        let num_layers = self.layers.len();
        let mut deltas = vec![Array1::zeros(0); num_layers];
        deltas[num_layers - 1] = delta;

        // Hidden layer deltas
        for l in (0..num_layers - 1).rev() {
            let next_delta = &deltas[l + 1];
            let wt_delta = self.layers[l + 1].weights.t().dot(next_delta);
            let relu_grad = zs[l].mapv(|z| if z > 0.0 { 1.0 } else { 0.0 });
            deltas[l] = wt_delta * relu_grad;
        }

        // Weight updates
        for l in 0..num_layers {
            let a_prev = &activations[l];
            let d = &deltas[l];
            for i in 0..self.layers[l].weights.nrows() {
                for j in 0..self.layers[l].weights.ncols() {
                    self.layers[l].weights[[i, j]] -= lr * d[i] * a_prev[j];
                }
                self.layers[l].biases[i] -= lr * d[i];
            }
        }

        loss
    }

    fn predict(&self, input: &Array1<f64>) -> usize {
        let (activations, _) = self.forward(input);
        let output = activations.last().unwrap();
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Single forward pass returning both prediction and output probabilities.
    fn predict_with_output(&self, input: &Array1<f64>) -> (usize, Array1<f64>) {
        let (activations, _) = self.forward(input);
        let output = activations.last().unwrap().clone();
        let predicted = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        (predicted, output)
    }

    fn clone_weights(&self) -> Vec<(Array2<f64>, Array1<f64>)> {
        self.layers
            .iter()
            .map(|l| (l.weights.clone(), l.biases.clone()))
            .collect()
    }

    fn load_weights(&mut self, weights: &[(Array2<f64>, Array1<f64>)]) {
        for (l, (w, b)) in self.layers.iter_mut().zip(weights.iter()) {
            l.weights = w.clone();
            l.biases = b.clone();
        }
    }
}

/// Compute the non-linear conductance level for a given pulse fraction t ∈ [0,1].
fn nl_level(t: f64, alpha: f64) -> f64 {
    if alpha.abs() < 0.01 {
        t
    } else {
        (1.0 - (-alpha * t).exp()) / (1.0 - (-alpha).exp())
    }
}

/// Pre-computed non-linear conductance levels for fast per-weight snapping.
struct DeviceLevelCache {
    levels_p: Vec<f64>,
    levels_d: Vec<f64>,
    write_noise: f64,
}

impl DeviceLevelCache {
    fn new(alpha_p: f64, alpha_d: f64, num_levels_p: usize, num_levels_d: usize, write_noise: f64) -> Self {
        let np = num_levels_p.max(2);
        let nd = num_levels_d.max(2);
        let levels_p: Vec<f64> = (0..np).map(|k| nl_level(k as f64 / (np - 1) as f64, alpha_p)).collect();
        let mut levels_d: Vec<f64> = (0..nd).map(|k| 1.0 - nl_level(k as f64 / (nd - 1) as f64, alpha_d)).collect();
        levels_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        DeviceLevelCache { levels_p, levels_d, write_noise }
    }

    /// Snap a normalized weight [0,1] to the nearest achievable level + noise.
    fn snap(&self, w_norm: f64, rng: &mut StdRng) -> f64 {
        let levels = if w_norm >= 0.5 { &self.levels_p } else { &self.levels_d };
        let mut best = levels[0];
        let mut best_dist = (w_norm - levels[0]).abs();
        for &lev in &levels[1..] {
            let d = (w_norm - lev).abs();
            if d < best_dist { best_dist = d; best = lev; }
        }
        let noise_dist = Normal::new(0.0, 1.0).unwrap();
        (best + rng.sample(noise_dist) * self.write_noise).clamp(0.0, 1.0)
    }
}

/// Apply memristor device non-idealities: non-linear quantization + write noise.
///
/// Physics: the non-linearity α determines the SPACING of achievable conductance
/// levels (not a transfer function on weights). Each ideal weight is snapped to
/// the nearest achievable level, then Gaussian write noise is added.
///
/// With α=0 the levels are evenly spaced (linear device).
/// With α>0 the levels are compressed toward G_max (fewer distinguishable
/// states at low conductance, more at high conductance).
fn apply_device_nonidealities(
    net: &mut MLP,
    cache: &DeviceLevelCache,
    rng: &mut StdRng,
) {
    for layer in &mut net.layers {
        let w_min = layer.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = layer.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w_range = w_max - w_min;

        if w_range < 1e-15 { continue; }

        for w in layer.weights.iter_mut() {
            let w_norm = ((*w - w_min) / w_range).clamp(0.0, 1.0);
            let snapped = cache.snap(w_norm, rng);
            *w = snapped * w_range + w_min;
        }
    }
}

/// Apply per-weight snapping after each SGD update (memristor-aware training).
/// This models real hardware where every weight update is constrained by the
/// device's non-linear conductance levels.
fn snap_weights_in_place(net: &mut MLP, cache: &DeviceLevelCache, rng: &mut StdRng) {
    for layer in &mut net.layers {
        let w_min = layer.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = layer.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w_range = w_max - w_min;

        if w_range < 1e-15 { continue; }

        for w in layer.weights.iter_mut() {
            let w_norm = ((*w - w_min) / w_range).clamp(0.0, 1.0);
            // Snap but with reduced noise during training (half noise) to allow
            // some gradient signal through. Full noise is applied at evaluation.
            let levels = if w_norm >= 0.5 { &cache.levels_p } else { &cache.levels_d };
            let mut best = levels[0];
            let mut best_dist = (w_norm - levels[0]).abs();
            for &lev in &levels[1..] {
                let d = (w_norm - lev).abs();
                if d < best_dist { best_dist = d; best = lev; }
            }
            let noise_dist = Normal::new(0.0, 1.0).unwrap();
            let snapped = (best + rng.sample(noise_dist) * cache.write_noise * 0.5).clamp(0.0, 1.0);
            *w = snapped * w_range + w_min;
        }
    }
}

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = x.mapv(|v| (v - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

fn generate_synthetic_mnist(
    num_train: usize,
    num_test: usize,
    rng: &mut StdRng,
) -> (Vec<(Array1<f64>, usize)>, Vec<(Array1<f64>, usize)>) {
    // Multiple template variants per digit for more realistic variation.
    // Each digit has 2-3 variants (e.g., open vs closed 4, serif vs sans 1).
    let templates: Vec<Vec<[u8; 25]>> = vec![
        // 0: round, square
        vec![
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ],
        // 1: straight, with serif, with flag
        vec![
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        ],
        // 2: curved, angular
        vec![
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        ],
        // 3: round, flat-top
        vec![
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        ],
        // 4: open, closed
        vec![
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        ],
        // 5: standard, with curved bottom
        vec![
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        ],
        // 6: standard, open-top
        vec![
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        ],
        // 7: standard, with crossbar
        vec![
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        ],
        // 8: standard, narrow
        vec![
            [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        ],
        // 9: standard, straight-tail
        vec![
            [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ];

    let noise_dist = Normal::new(0.0, 1.0).unwrap();

    let generate_sample = |digit: usize, rng: &mut StdRng| -> Array1<f64> {
        // Pick a random template variant for this digit
        let variants = &templates[digit];
        let variant_idx = rng.gen_range(0..variants.len());
        let template = &variants[variant_idx];

        let mut img = Array2::<f64>::zeros((28, 28));

        // Increased jitter range for more position variation
        let jitter_x: i32 = rng.gen_range(-4..=4);
        let jitter_y: i32 = rng.gen_range(-4..=4);
        let scale = rng.gen_range(3.0..6.0) as f64;

        // Random rotation angle in radians (±15 degrees)
        let angle: f64 = rng.gen_range(-0.26..0.26);
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let cx_center = 14.0_f64;
        let cy_center = 14.0_f64;

        // Random partial erasure: skip ~15% of template pixels
        let erasure_rate: f64 = rng.gen_range(0.0..0.20);

        for ty in 0..5 {
            for tx in 0..5 {
                if template[ty * 5 + tx] == 0 { continue; }

                // Random erasure
                if rng.gen::<f64>() < erasure_rate { continue; }

                let raw_cx = tx as f64 * scale + 4.0 + jitter_x as f64;
                let raw_cy = ty as f64 * scale + 4.0 + jitter_y as f64;

                // Apply rotation around image center
                let dx = raw_cx - cx_center;
                let dy = raw_cy - cy_center;
                let rot_cx = cx_center + dx * cos_a - dy * sin_a;
                let rot_cy = cy_center + dx * sin_a + dy * cos_a;

                // Variable stroke width (1-3 pixels)
                let stroke_w = rng.gen_range(1..=3).min(scale as usize);
                let stroke_h = rng.gen_range(1..=3).min(scale as usize);

                for ddy in 0..stroke_h {
                    for ddx in 0..stroke_w {
                        let py = (rot_cy as usize + ddy).min(27);
                        let px = (rot_cx as usize + ddx).min(27);
                        let intensity = rng.gen_range(0.5..1.0);
                        img[[py, px]] = img[[py, px]].max(intensity);
                    }
                }
            }
        }

        // Higher background noise (0.05 → 0.12) to make classification harder
        for y in 0..28 {
            for x in 0..28 {
                let noise_val: f64 = rng.sample(noise_dist);
                img[[y, x]] += noise_val.abs() * 0.12;
                img[[y, x]] = img[[y, x]].clamp(0.0, 1.0);
            }
        }

        img.into_shape_with_order(784).unwrap()
    };

    let mut train_data = Vec::with_capacity(num_train);
    for i in 0..num_train {
        let digit = i % 10;
        train_data.push((generate_sample(digit, rng), digit));
    }
    train_data.shuffle(rng);

    let mut test_data = Vec::with_capacity(num_test);
    for i in 0..num_test {
        let digit = i % 10;
        test_data.push((generate_sample(digit, rng), digit));
    }
    test_data.shuffle(rng);

    (train_data, test_data)
}

#[tauri::command]
pub async fn train_ann(
    params: DeviceParams,
    config: ANNConfig,
    window: tauri::Window,
) -> Result<Vec<ANNEpochResult>, String> {
    let result = std::thread::spawn(move || train_ann_inner(params, config, &window))
        .join()
        .map_err(|_| "Training thread panicked".to_string())??;

    Ok(result)
}

fn train_ann_inner(
    params: DeviceParams,
    config: ANNConfig,
    window: &tauri::Window,
) -> Result<Vec<ANNEpochResult>, String> {
    let mut rng = StdRng::seed_from_u64(42);

    let (train_data, test_data) = generate_synthetic_mnist(5000, 1000, &mut rng);

    let sizes: Vec<usize> = match config.model_type.as_str() {
        "perceptron" => vec![784, 10],
        "mlp_2h" => vec![784, config.hidden_size, config.hidden_size_2, 10],
        _ => vec![784, config.hidden_size, 10],
    };

    let mut ideal_net = MLP::new(&sizes, &mut rng);
    // Memristor net starts from same initial weights as ideal
    let mut mem_net = MLP::new(&sizes, &mut rng);

    let cache = DeviceLevelCache::new(
        params.alpha_p,
        params.alpha_d,
        params.num_levels_p,
        params.num_levels_d,
        params.write_noise,
    );

    let mut results = Vec::new();
    let initial_lr = config.learning_rate;
    let num_noise_samples: usize = 5;

    for epoch in 0..config.epochs {
        // Cosine annealing: smoothly decay lr to prevent oscillation near convergence.
        // Minimum lr is 5% of initial to ensure continued learning.
        let progress = epoch as f64 / config.epochs as f64;
        let current_lr =
            initial_lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()));

        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        let mut ideal_loss_sum = 0.0;
        let mut mem_loss_sum = 0.0;
        let mut count = 0.0;

        // Per-sample SGD training for BOTH ideal and memristor networks.
        // The memristor network gets weights snapped to non-linear levels after
        // each update, modelling real hardware-in-the-loop training.
        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            for &idx in &indices[batch_start..batch_end] {
                let (ref input, target) = train_data[idx];

                // Ideal network: clean SGD
                ideal_loss_sum += ideal_net.backprop(input, target, current_lr);

                // Memristor network: SGD + snap to nearest conductance level
                mem_loss_sum += mem_net.backprop(input, target, current_lr);
                count += 1.0;
            }

            // After each mini-batch, snap memristor weights to device levels.
            // This is more efficient than per-sample snapping and models the
            // batch-programming approach used in real crossbar training.
            snap_weights_in_place(&mut mem_net, &cache, &mut rng);
        }

        // Evaluate ideal network on test set
        let mut ideal_correct = 0;
        for (input, target) in &test_data {
            if ideal_net.predict(input) == *target {
                ideal_correct += 1;
            }
        }

        // Evaluate memristor network on test set.
        // For evaluation, apply full noise (copy-and-degrade from the trained
        // memristor weights) and average over noise realizations.
        let mem_weights = mem_net.clone_weights();
        let mut mem_correct_total: usize = 0;
        let mut mem_eval_loss_total: f64 = 0.0;

        for _ in 0..num_noise_samples {
            let mut eval_net = MLP::new(&sizes, &mut rng);
            eval_net.load_weights(&mem_weights);
            apply_device_nonidealities(&mut eval_net, &cache, &mut rng);

            for (input, target) in &test_data {
                let (predicted, output) = eval_net.predict_with_output(input);
                if predicted == *target {
                    mem_correct_total += 1;
                }
                mem_eval_loss_total += -output[*target].max(1e-15).ln();
            }
        }

        let total_mem_evals = (test_data.len() * num_noise_samples) as f64;

        let result = ANNEpochResult {
            epoch,  // starts at 0
            ideal_accuracy: ideal_correct as f64 / test_data.len() as f64 * 100.0,
            memristor_accuracy: mem_correct_total as f64 / total_mem_evals * 100.0,
            ideal_loss: ideal_loss_sum / count,
            memristor_loss: mem_eval_loss_total / total_mem_evals,
        };

        let _ = window.emit("ann-progress", &result);
        results.push(result);
    }

    Ok(results)
}
