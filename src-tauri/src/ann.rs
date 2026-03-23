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
    /// Momentum velocity buffers for weights
    vel_w: Vec<Array2<f64>>,
    /// Momentum velocity buffers for biases
    vel_b: Vec<Array1<f64>>,
}

impl MLP {
    fn new(sizes: &[usize], rng: &mut StdRng) -> Self {
        let mut layers = Vec::new();
        let mut vel_w = Vec::new();
        let mut vel_b = Vec::new();
        for i in 0..sizes.len() - 1 {
            let fan_in = sizes[i];
            let fan_out = sizes[i + 1];
            let std_dev = (2.0 / fan_in as f64).sqrt();
            let dist = Normal::new(0.0, std_dev).unwrap();

            let weights = Array2::from_shape_fn((fan_out, fan_in), |_| rng.sample(dist));
            let biases = Array1::zeros(fan_out);
            layers.push(Layer { weights, biases });
            vel_w.push(Array2::zeros((fan_out, fan_in)));
            vel_b.push(Array1::zeros(fan_out));
        }
        MLP {
            layers,
            vel_w,
            vel_b,
        }
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

    /// Accumulate gradients for one sample directly into pre-allocated buffers.
    /// No per-sample allocation — only the forward/backward vectors are created.
    fn accumulate_gradients(
        &self,
        input: &Array1<f64>,
        target: usize,
        acc_w: &mut [Array2<f64>],
        acc_b: &mut [Array1<f64>],
    ) -> f64 {
        let (activations, zs) = self.forward(input);
        let output = activations.last().unwrap();
        let loss = -output[target].max(1e-15).ln();

        // Output layer delta
        let mut delta = output.clone();
        delta[target] -= 1.0;

        let num_layers = self.layers.len();
        let mut deltas = vec![Array1::zeros(0); num_layers];
        deltas[num_layers - 1] = delta;

        for l in (0..num_layers - 1).rev() {
            let next_delta = &deltas[l + 1];
            let wt_delta = self.layers[l + 1].weights.t().dot(next_delta);
            let relu_grad = zs[l].mapv(|z| if z > 0.0 { 1.0 } else { 0.0 });
            deltas[l] = wt_delta * relu_grad;
        }

        // Accumulate outer-product gradients directly into the buffers
        for l in 0..num_layers {
            let a_prev = &activations[l];
            let d = &deltas[l];
            for i in 0..self.layers[l].weights.nrows() {
                acc_b[l][i] += d[i];
                for j in 0..self.layers[l].weights.ncols() {
                    acc_w[l][[i, j]] += d[i] * a_prev[j];
                }
            }
        }

        loss
    }

    /// Apply batch-averaged gradients with SGD + momentum.
    fn apply_gradients(
        &mut self,
        w_grads: &[Array2<f64>],
        b_grads: &[Array1<f64>],
        lr: f64,
        momentum: f64,
    ) {
        for l in 0..self.layers.len() {
            self.vel_w[l].mapv_inplace(|v| v * momentum);
            self.vel_w[l].scaled_add(lr, &w_grads[l]);
            self.vel_b[l].mapv_inplace(|v| v * momentum);
            self.vel_b[l].scaled_add(lr, &b_grads[l]);

            self.layers[l].weights.scaled_add(-1.0, &self.vel_w[l]);
            self.layers[l].biases.scaled_add(-1.0, &self.vel_b[l]);
        }
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

    /// Forward pass that returns both prediction and output probabilities (avoids double forward).
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

fn apply_nonlinear_weight_remap(net: &mut MLP, alpha_p: f64, alpha_d: f64) {
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
            let alpha = if w_norm >= 0.5 { alpha_p } else { alpha_d };

            let remapped = if alpha.abs() < 0.01 {
                w_norm
            } else {
                (1.0 - (-alpha * w_norm).exp()) / (1.0 - (-alpha).exp())
            };

            *w = remapped * w_range + w_min;
        }
    }
}

fn apply_memristor_noise(net: &mut MLP, write_noise: f64, num_levels: usize, rng: &mut StdRng) {
    let noise_dist = Normal::new(0.0, 1.0).unwrap();
    let n_levels = num_levels.max(2) as f64;

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
            let mut w_norm = (*w - w_min) / w_range;
            w_norm = (w_norm * (n_levels - 1.0)).round() / (n_levels - 1.0);
            w_norm += rng.sample(noise_dist) * write_noise / n_levels.sqrt();
            w_norm = w_norm.clamp(0.0, 1.0);
            *w = w_norm * w_range + w_min;
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
    let momentum = 0.9;
    let initial_lr = config.learning_rate;

    // Pre-allocate gradient accumulators ONCE (reused every batch)
    let mut acc_w: Vec<Array2<f64>> = ideal_net
        .layers
        .iter()
        .map(|l| Array2::zeros((l.weights.nrows(), l.weights.ncols())))
        .collect();
    let mut acc_b: Vec<Array1<f64>> = ideal_net
        .layers
        .iter()
        .map(|l| Array1::zeros(l.biases.len()))
        .collect();

    for epoch in 0..config.epochs {
        // Cosine annealing learning rate
        let progress = epoch as f64 / config.epochs as f64;
        let current_lr = initial_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());

        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        let mut ideal_loss_sum = 0.0;
        let mut count = 0.0;

        // Mini-batch training with efficient gradient accumulation
        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            let batch_len = (batch_end - batch_start) as f64;

            // Zero the pre-allocated accumulators
            for i in 0..acc_w.len() {
                acc_w[i].fill(0.0);
                acc_b[i].fill(0.0);
            }

            // Accumulate gradients directly into buffers (no per-sample allocation)
            for &idx in &indices[batch_start..batch_end] {
                let (ref input, target) = train_data[idx];
                ideal_loss_sum +=
                    ideal_net.accumulate_gradients(input, target, &mut acc_w, &mut acc_b);
                count += 1.0;
            }

            // Average over batch
            for i in 0..acc_w.len() {
                acc_w[i].mapv_inplace(|v| v / batch_len);
                acc_b[i].mapv_inplace(|v| v / batch_len);
            }

            // Apply with momentum
            ideal_net.apply_gradients(&acc_w, &acc_b, current_lr, momentum);
        }

        // Evaluate ideal network
        let mut ideal_correct = 0;
        for (input, target) in &test_data {
            if ideal_net.predict(input) == *target {
                ideal_correct += 1;
            }
        }

        // Copy-and-degrade with 2 noise samples averaged for stability
        let ideal_weights = ideal_net.clone_weights();
        let mut mem_correct_total: usize = 0;
        let mut mem_loss_total: f64 = 0.0;
        let num_noise_samples: usize = 2;

        for _ in 0..num_noise_samples {
            mem_net.load_weights(&ideal_weights);
            apply_nonlinear_weight_remap(&mut mem_net, params.alpha_p, params.alpha_d);
            apply_memristor_noise(&mut mem_net, params.write_noise, num_levels, &mut rng);

            for (input, target) in &test_data {
                // Single forward pass for both prediction and loss
                let (predicted, output) = mem_net.predict_with_output(input);
                if predicted == *target {
                    mem_correct_total += 1;
                }
                mem_loss_total += -output[*target].max(1e-15).ln();
            }
        }

        let total_mem_evals = (test_data.len() * num_noise_samples) as f64;

        let result = ANNEpochResult {
            epoch: epoch + 1,
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
