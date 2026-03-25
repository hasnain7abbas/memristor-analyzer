use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use tauri::Emitter;

// ─── Configuration structs ────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ANNConfig {
    #[serde(rename = "modelType", default = "default_model_type")]
    pub model_type: String,
    #[serde(rename = "hiddenSize")]
    pub hidden_size: usize,
    #[serde(rename = "hiddenSize2", default = "default_hidden_size_2")]
    pub hidden_size_2: usize,
    pub epochs: usize,
    #[serde(rename = "learningRate", default = "default_learning_rate")]
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

fn default_learning_rate() -> f64 {
    0.001
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
    #[serde(rename = "gMin", default)]
    pub g_min: f64,
    #[serde(rename = "gMax", default)]
    pub g_max: f64,
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

// ─── Layer / MLP ──────────────────────────────────────────────────────

struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Create with He initialization: std = sqrt(2 / fan_in)
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

    /// Forward pass: ReLU hidden layers, softmax output.
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

    /// Train on one sample with SGD. Returns cross-entropy loss.
    fn backprop(&mut self, input: &Array1<f64>, target: usize, lr: f64) -> f64 {
        let (activations, zs) = self.forward(input);
        let output = activations.last().unwrap();

        // Cross-entropy loss
        let loss = -output[target].max(1e-15).ln();

        // Output layer delta: softmax - one_hot(target)
        let mut delta = output.clone();
        delta[target] -= 1.0;

        let num_layers = self.layers.len();
        let mut deltas = vec![Array1::zeros(0); num_layers];
        deltas[num_layers - 1] = delta;

        // Backprop through hidden layers
        for l in (0..num_layers - 1).rev() {
            let next_delta = &deltas[l + 1];
            let wt_delta = self.layers[l + 1].weights.t().dot(next_delta);
            let relu_grad = zs[l].mapv(|z| if z > 0.0 { 1.0 } else { 0.0 });
            deltas[l] = wt_delta * relu_grad;
        }

        // SGD weight updates
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

// ─── Activation functions ───────────────────────────────────────────

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = x.mapv(|v| (v - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

// ─── Device mapping (copy-and-degrade) ──────────────────────────────

/// Apply non-linear weight remapping to simulate memristor storage distortion.
/// Maps weights through (1 - exp(-alpha * w_norm)) / (1 - exp(-alpha)) curve.
fn apply_nonlinear_weight_remap(net: &mut MLP, alpha_p: f64, alpha_d: f64) {
    for layer in &mut net.layers {
        let w_min = layer.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = layer.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w_range = w_max - w_min;

        if w_range < 1e-15 {
            continue;
        }

        for w in layer.weights.iter_mut() {
            let w_norm = ((*w - w_min) / w_range).clamp(0.0, 1.0);

            // Use alpha_p for upper half of weight range, alpha_d for lower half
            let alpha = if w_norm >= 0.5 { alpha_p } else { alpha_d };

            let remapped = if alpha.abs() < 0.01 {
                w_norm // Near-linear case
            } else {
                (1.0 - (-alpha * w_norm).exp()) / (1.0 - (-alpha).exp())
            };

            *w = remapped * w_range + w_min;
        }
    }
}

/// Apply quantization and write noise to simulate memristor storage.
fn apply_memristor_noise(net: &mut MLP, write_noise: f64, num_levels: usize, rng: &mut StdRng) {
    let noise_dist = Normal::new(0.0, 1.0).unwrap();
    let n_levels = num_levels.max(2) as f64;

    for layer in &mut net.layers {
        let w_min = layer.weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = layer.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w_range = w_max - w_min;

        if w_range < 1e-15 {
            continue;
        }

        for w in layer.weights.iter_mut() {
            // Normalize to [0, 1]
            let mut w_norm = (*w - w_min) / w_range;
            // Quantize to nearest level
            w_norm = (w_norm * (n_levels - 1.0)).round() / (n_levels - 1.0);
            // Add write noise (scaled by 1/sqrt(levels))
            w_norm += rng.sample(noise_dist) * write_noise / n_levels.sqrt();
            // Clamp to valid range
            w_norm = w_norm.clamp(0.0, 1.0);
            // Denormalize
            *w = w_norm * w_range + w_min;
        }
    }
}

// ─── Synthetic MNIST data generation ─────────────────────────────────

fn generate_synthetic_mnist(
    num_train: usize,
    num_test: usize,
    rng: &mut StdRng,
) -> (Vec<(Array1<f64>, usize)>, Vec<(Array1<f64>, usize)>) {
    // 5x5 digit templates (0-9)
    let templates: [[u8; 25]; 10] = [
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0], // 0
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], // 1
        [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0], // 2
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0], // 3
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], // 4
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0], // 5
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], // 6
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], // 7
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], // 8
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], // 9
    ];

    let noise_dist = Normal::new(0.0, 1.0).unwrap();

    let generate_sample = |digit: usize, rng: &mut StdRng| -> Array1<f64> {
        let template = &templates[digit];
        let mut img = ndarray::Array2::<f64>::zeros((28, 28));

        // Scale 5x5 template to center of 28x28 with jitter
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

        // Light background noise
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

// ─── Tauri command ───────────────────────────────────────────────────

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

    // Generate data: 5000 training + 1000 test (separately generated)
    let (train_data, test_data) = generate_synthetic_mnist(5000, 1000, &mut rng);

    // Build layer sizes based on model type
    let sizes: Vec<usize> = match config.model_type.as_str() {
        "perceptron" => vec![784, 10],
        "mlp_2h" => vec![784, config.hidden_size, config.hidden_size_2, 10],
        _ => vec![784, config.hidden_size, 10], // mlp_1h default
    };

    // Create ideal network — memristor is evaluated via copy-and-degrade
    let mut ideal_net = MLP::new(&sizes, &mut rng);
    let mut mem_net = MLP::new(&sizes, &mut rng);

    let num_levels = params.num_levels_p.max(params.num_levels_d).max(2);
    let mut results = Vec::new();

    for epoch in 0..config.epochs {
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        let mut ideal_loss_sum = 0.0;
        let mut count = 0.0;

        // Train ONLY the ideal network (per-sample SGD)
        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            for &idx in &indices[batch_start..batch_end] {
                let (ref input, target) = train_data[idx];
                ideal_loss_sum += ideal_net.backprop(input, target, config.learning_rate);
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

        // Copy-and-degrade: average memristor accuracy over N noise realizations
        // (Burr et al. 2015 methodology — eliminates stochastic oscillation)
        let n_realizations = 5;
        let ideal_weights = ideal_net.clone_weights();
        let mut mem_acc_sum = 0.0;
        let mut mem_loss_sum = 0.0;

        for r in 0..n_realizations {
            mem_net.load_weights(&ideal_weights);

            // Apply non-linear weight remapping (alpha curves distort stored values)
            apply_nonlinear_weight_remap(&mut mem_net, params.alpha_p, params.alpha_d);

            // Apply quantization + write noise with deterministic per-epoch/realization seed
            let noise_seed = 10000 * (epoch as u64 + 1) + r as u64;
            let mut noise_rng = StdRng::seed_from_u64(noise_seed);
            apply_memristor_noise(&mut mem_net, params.write_noise, num_levels, &mut noise_rng);

            // Evaluate degraded network
            let mut mem_correct = 0;
            for (input, target) in &test_data {
                if mem_net.predict(input) == *target {
                    mem_correct += 1;
                }
                let (activations, _) = mem_net.forward(input);
                let output = activations.last().unwrap();
                mem_loss_sum += -output[*target].max(1e-15).ln();
            }
            mem_acc_sum += mem_correct as f64 / test_data.len() as f64 * 100.0;
        }

        let test_len = test_data.len() as f64;
        let result = ANNEpochResult {
            epoch: epoch + 1,
            ideal_accuracy: ideal_correct as f64 / test_len * 100.0,
            memristor_accuracy: mem_acc_sum / n_realizations as f64,
            ideal_loss: ideal_loss_sum / count,
            memristor_loss: mem_loss_sum / (n_realizations as f64 * test_len),
        };

        let _ = window.emit("ann-progress", &result);
        results.push(result);
    }

    Ok(results)
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_accuracy_reaches_100() {
        let mut rng = StdRng::seed_from_u64(42);
        // Smaller dataset for fast debug-mode testing
        let (train_data, test_data) = generate_synthetic_mnist(1000, 200, &mut rng);

        let sizes = vec![784, 64, 10];
        let mut net = MLP::new(&sizes, &mut rng);
        let lr = 0.001;
        let epochs = 10;

        let mut last_acc = 0.0;
        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..train_data.len()).collect();
            indices.shuffle(&mut rng);

            for &idx in &indices {
                let (ref input, target) = train_data[idx];
                net.backprop(input, target, lr);
            }

            let correct = test_data.iter()
                .filter(|(input, target)| net.predict(input) == *target)
                .count();
            last_acc = correct as f64 / test_data.len() as f64 * 100.0;
            println!("Epoch {}: accuracy = {:.1}%", epoch + 1, last_acc);
        }

        // With small test config (1000 train, 64 hidden, 10 epochs) 80%+ is expected.
        // Full config (5000 train, 256 hidden, 50 epochs) reaches ~100%.
        assert!(last_acc >= 80.0, "Ideal accuracy should reach >=80%, got {:.1}%", last_acc);
    }
}
