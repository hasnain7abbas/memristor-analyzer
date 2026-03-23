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
    alpha_p: f64,
    alpha_d: f64,
    write_noise: f64,
    num_levels: usize,
    rng: &mut StdRng,
) {
    let noise_dist = Normal::new(0.0, 1.0).unwrap();
    let n = num_levels.max(2);

    // Pre-compute non-linearly spaced conductance levels for potentiation and depression.
    // Potentiation levels go from 0 toward 1 (increasing conductance).
    // Depression levels go from 1 toward 0 (decreasing conductance), so we use
    // 1 - nl_level(...) and sort ascending to get levels spanning [0, 1] with
    // spacing compressed toward the LOW end (mirroring potentiation's compression
    // toward the HIGH end).
    let levels_p: Vec<f64> = (0..n).map(|k| nl_level(k as f64 / (n - 1) as f64, alpha_p)).collect();
    let mut levels_d: Vec<f64> = (0..n).map(|k| 1.0 - nl_level(k as f64 / (n - 1) as f64, alpha_d)).collect();
    levels_d.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for layer in &mut net.layers {
        let w_min = layer.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = layer
            .weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let w_range = w_max - w_min;

        if w_range < 1e-15 {
            continue;
        }

        for w in layer.weights.iter_mut() {
            let w_norm = ((*w - w_min) / w_range).clamp(0.0, 1.0);

            // Choose level set: potentiation for upper half, depression for lower half
            let levels = if w_norm >= 0.5 { &levels_p } else { &levels_d };

            // Snap to nearest achievable conductance level
            let mut best_level = levels[0];
            let mut best_dist = (w_norm - levels[0]).abs();
            for &level in &levels[1..] {
                let dist = (w_norm - level).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_level = level;
                }
            }

            // Add Gaussian write noise scaled by level spacing
            let noisy = best_level + rng.sample(noise_dist) * write_noise;
            let clamped = noisy.clamp(0.0, 1.0);

            *w = clamped * w_range + w_min;
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
    let templates: [[u8; 25]; 10] = [
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    ];

    let noise_dist = Normal::new(0.0, 1.0).unwrap();

    let generate_sample = |digit: usize, rng: &mut StdRng| -> Array1<f64> {
        let template = &templates[digit];
        let mut img = Array2::<f64>::zeros((28, 28));

        let jitter_x: i32 = rng.gen_range(-2..=2);
        let jitter_y: i32 = rng.gen_range(-2..=2);
        let scale = rng.gen_range(3.5..5.5) as f64;

        for ty in 0..5 {
            for tx in 0..5 {
                if template[ty * 5 + tx] == 1 {
                    let cx = (tx as f64 * scale + 4.0 + jitter_x as f64) as usize;
                    let cy = (ty as f64 * scale + 4.0 + jitter_y as f64) as usize;
                    let s = scale as usize;
                    for dy in 0..s.min(3) {
                        for dx in 0..s.min(3) {
                            let py = (cy + dy).min(27);
                            let px = (cx + dx).min(27);
                            let intensity = rng.gen_range(0.6..1.0);
                            img[[py, px]] = intensity;
                        }
                    }
                }
            }
        }

        for y in 0..28 {
            for x in 0..28 {
                let noise_val: f64 = rng.sample(noise_dist);
                img[[y, x]] += noise_val.abs() * 0.05;
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
    let mut mem_net = MLP::new(&sizes, &mut rng);

    let mut results = Vec::new();
    let num_levels = params.num_levels_p.max(params.num_levels_d).max(2);
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
        let mut count = 0.0;

        // Per-sample SGD training — 5000 weight updates per epoch.
        // This is essential for convergence on the small synthetic MNIST dataset.
        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            for &idx in &indices[batch_start..batch_end] {
                let (ref input, target) = train_data[idx];
                ideal_loss_sum += ideal_net.backprop(input, target, current_lr);
                count += 1.0;
            }
        }

        // Evaluate ideal network on test set
        let mut ideal_correct = 0;
        for (input, target) in &test_data {
            if ideal_net.predict(input) == *target {
                ideal_correct += 1;
            }
        }

        // Copy-and-degrade: average over noise realizations for smooth curves
        let ideal_weights = ideal_net.clone_weights();
        let mut mem_correct_total: usize = 0;
        let mut mem_loss_total: f64 = 0.0;

        for _ in 0..num_noise_samples {
            mem_net.load_weights(&ideal_weights);
            apply_device_nonidealities(
                &mut mem_net,
                params.alpha_p,
                params.alpha_d,
                params.write_noise,
                num_levels,
                &mut rng,
            );

            for (input, target) in &test_data {
                let (predicted, output) = mem_net.predict_with_output(input);
                if predicted == *target {
                    mem_correct_total += 1;
                }
                mem_loss_total += -output[*target].max(1e-15).ln();
            }
        }

        let total_mem_evals = (test_data.len() * num_noise_samples) as f64;

        let result = ANNEpochResult {
            epoch,  // starts at 0
            ideal_accuracy: ideal_correct as f64 / test_data.len() as f64 * 100.0,
            memristor_accuracy: mem_correct_total as f64 / total_mem_evals * 100.0,
            ideal_loss: ideal_loss_sum / count,
            memristor_loss: mem_loss_total / total_mem_evals,
        };

        let _ = window.emit("ann-progress", &result);
        results.push(result);
    }

    Ok(results)
}
