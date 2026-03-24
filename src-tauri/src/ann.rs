use ndarray::{Array1, Array2, Axis};
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

// ─── Layer / MLP structures ──────────────────────────────────────────

struct Layer {
    weights: Array2<f64>,  // shape: (fan_out, fan_in)
    biases: Array1<f64>,   // shape: (fan_out,)
}

struct AdamState {
    /// First moment (mean of gradients) for each layer's weights and biases
    m_w: Vec<Array2<f64>>,
    m_b: Vec<Array1<f64>>,
    /// Second moment (mean of squared gradients)
    v_w: Vec<Array2<f64>>,
    v_b: Vec<Array1<f64>>,
    /// Timestep counter
    t: usize,
    /// Hyperparameters
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl AdamState {
    fn new(sizes: &[usize], lr: f64) -> Self {
        let num_layers = sizes.len() - 1;
        let mut m_w = Vec::with_capacity(num_layers);
        let mut m_b = Vec::with_capacity(num_layers);
        let mut v_w = Vec::with_capacity(num_layers);
        let mut v_b = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let fan_in = sizes[i];
            let fan_out = sizes[i + 1];
            m_w.push(Array2::zeros((fan_out, fan_in)));
            m_b.push(Array1::zeros(fan_out));
            v_w.push(Array2::zeros((fan_out, fan_in)));
            v_b.push(Array1::zeros(fan_out));
        }

        AdamState {
            m_w,
            m_b,
            v_w,
            v_b,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn step(&mut self, layers: &mut [Layer], grad_w: &[Array2<f64>], grad_b: &[Array1<f64>]) {
        self.t += 1;
        let t = self.t as f64;

        for i in 0..layers.len() {
            // Update biased first moment
            self.m_w[i] = &self.m_w[i] * self.beta1 + &grad_w[i] * (1.0 - self.beta1);
            self.m_b[i] = &self.m_b[i] * self.beta1 + &grad_b[i] * (1.0 - self.beta1);

            // Update biased second moment
            self.v_w[i] = &self.v_w[i] * self.beta2 + &(&grad_w[i] * &grad_w[i]) * (1.0 - self.beta2);
            self.v_b[i] = &self.v_b[i] * self.beta2 + &(&grad_b[i] * &grad_b[i]) * (1.0 - self.beta2);

            // Bias correction
            let bc1 = 1.0 - self.beta1.powf(t);
            let bc2 = 1.0 - self.beta2.powf(t);

            let m_hat_w = &self.m_w[i] / bc1;
            let m_hat_b = &self.m_b[i] / bc1;
            let v_hat_w = &self.v_w[i] / bc2;
            let v_hat_b = &self.v_b[i] / bc2;

            // Update weights: w = w - lr * m_hat / (sqrt(v_hat) + eps)
            layers[i].weights = &layers[i].weights - &(m_hat_w / &(v_hat_w.mapv(f64::sqrt) + self.eps)) * self.lr;
            layers[i].biases = &layers[i].biases - &(m_hat_b / &(v_hat_b.mapv(f64::sqrt) + self.eps)) * self.lr;
        }
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Create a new MLP with Xavier/Glorot initialization.
    /// std = sqrt(2 / (fan_in + fan_out))
    fn new(sizes: &[usize], rng: &mut StdRng) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            let fan_in = sizes[i];
            let fan_out = sizes[i + 1];
            let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();
            let dist = Normal::new(0.0, std_dev).unwrap();

            let weights = Array2::from_shape_fn((fan_out, fan_in), |_| rng.sample(dist));
            let biases = Array1::zeros(fan_out);
            layers.push(Layer { weights, biases });
        }
        MLP { layers }
    }

    /// Forward pass returning raw logits (NO softmax on output layer).
    /// Returns (activations_per_layer, pre_activation_z_per_layer).
    /// activations[0] = input, activations[last] = output logits
    fn forward(&self, input: &Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut activations = vec![input.clone()];
        let mut zs = Vec::new();
        let mut current = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.weights.dot(&current) + &layer.biases;
            let a = if i < self.layers.len() - 1 {
                // Hidden layer: ReLU
                relu(&z)
            } else {
                // Output layer: raw logits (no activation)
                z.clone()
            };
            zs.push(z);
            current = a.clone();
            activations.push(a);
        }
        (activations, zs)
    }

    /// Forward pass for a batch. Returns (batch_logits, batch_activations, batch_zs).
    fn forward_batch(
        &self,
        inputs: &[&Array1<f64>],
    ) -> (Vec<Array1<f64>>, Vec<Vec<Array1<f64>>>, Vec<Vec<Array1<f64>>>) {
        let mut all_logits = Vec::with_capacity(inputs.len());
        let mut all_activations = Vec::with_capacity(inputs.len());
        let mut all_zs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let (acts, zs) = self.forward(input);
            all_logits.push(acts.last().unwrap().clone());
            all_activations.push(acts);
            all_zs.push(zs);
        }
        (all_logits, all_activations, all_zs)
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

// ─── Activation & loss functions ─────────────────────────────────────

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Softmax for computing probabilities from logits.
fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|v| (v - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

/// Cross-entropy loss from raw logits and a target class index.
/// loss = -log(softmax(logits)[target])
/// Also returns softmax probabilities for gradient computation.
fn cross_entropy_loss(logits: &Array1<f64>, target: usize) -> (f64, Array1<f64>) {
    let probs = softmax(logits);
    let loss = -probs[target].max(1e-15).ln();
    (loss, probs)
}

// ─── Backpropagation (computes gradients, does NOT update weights) ───

/// Compute gradients for all layers given one sample.
/// Returns (grad_weights, grad_biases) for each layer.
fn backward(
    layers: &[Layer],
    activations: &[Array1<f64>],
    zs: &[Array1<f64>],
    probs: &Array1<f64>,
    target: usize,
) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
    let num_layers = layers.len();
    let mut grad_w: Vec<Array2<f64>> = Vec::with_capacity(num_layers);
    let mut grad_b: Vec<Array1<f64>> = Vec::with_capacity(num_layers);

    // Output layer delta: softmax(logits) - one_hot(target)
    let mut delta = probs.clone();
    delta[target] -= 1.0;

    // Initialize gradient storage
    for _ in 0..num_layers {
        grad_w.push(Array2::zeros((0, 0)));
        grad_b.push(Array1::zeros(0));
    }

    // Output layer gradients
    let a_prev = &activations[num_layers - 1];
    // grad_w[L] = outer(delta, a_prev)
    let gw = delta
        .view()
        .insert_axis(Axis(1))
        .dot(&a_prev.view().insert_axis(Axis(0)));
    grad_w[num_layers - 1] = gw;
    grad_b[num_layers - 1] = delta.clone();

    // Backpropagate through hidden layers
    for l in (0..num_layers - 1).rev() {
        let wt_delta = layers[l + 1].weights.t().dot(&delta);
        let relu_grad = zs[l].mapv(|z| if z > 0.0 { 1.0 } else { 0.0 });
        delta = wt_delta * relu_grad;

        let a_prev = &activations[l];
        let gw = delta
            .view()
            .insert_axis(Axis(1))
            .dot(&a_prev.view().insert_axis(Axis(0)));
        grad_w[l] = gw;
        grad_b[l] = delta.clone();
    }

    (grad_w, grad_b)
}

/// Accumulate gradients over a mini-batch (average).
fn compute_batch_gradients(
    layers: &[Layer],
    all_activations: &[Vec<Array1<f64>>],
    all_zs: &[Vec<Array1<f64>>],
    all_probs: &[Array1<f64>],
    targets: &[usize],
) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
    let batch_size = targets.len();
    let num_layers = layers.len();

    // Initialize accumulators with correct shapes
    let mut acc_w: Vec<Array2<f64>> = layers
        .iter()
        .map(|l| Array2::zeros(l.weights.raw_dim()))
        .collect();
    let mut acc_b: Vec<Array1<f64>> = layers
        .iter()
        .map(|l| Array1::zeros(l.biases.raw_dim()))
        .collect();

    for i in 0..batch_size {
        let (gw, gb) = backward(layers, &all_activations[i], &all_zs[i], &all_probs[i], targets[i]);
        for l in 0..num_layers {
            acc_w[l] = &acc_w[l] + &gw[l];
            acc_b[l] = &acc_b[l] + &gb[l];
        }
    }

    // Average
    let bs = batch_size as f64;
    for l in 0..num_layers {
        acc_w[l] = &acc_w[l] / bs;
        acc_b[l] = &acc_b[l] / bs;
    }

    (acc_w, acc_b)
}

// ─── Device mapping ──────────────────────────────────────────────────

/// Apply memristor device mapping to weight matrices (not biases).
///
/// Bug 2 fix: Uses FIXED bounds [-1, 1] — never recalculates from weight tensor.
/// Bug 3 fix: Uses STOCHASTIC rounding — deterministic snap freezes learning.
/// Noise fix: Effective sigma = raw_sigma / sqrt(num_levels) to account for
///            statistical averaging during multi-pulse programming.
///
/// Procedure for each weight w:
///   1. Clamp to [-1, 1] and normalize to [0, 1]
///   2. Stochastic round to nearest hardware level
///   3. Map to conductance and add multiplicative noise
///   4. Map back to weight space using fixed bounds
fn apply_device_mapping(
    net: &mut MLP,
    g_min: f64,
    g_max: f64,
    raw_sigma_w: f64,
    num_levels_p: usize,
    num_levels_d: usize,
    rng: &mut StdRng,
) {
    let num_levels = (num_levels_p * num_levels_d).max(2);
    let noise_dist = Normal::new(0.0, 1.0).unwrap();
    let g_range = g_max - g_min;
    let nl_f64 = (num_levels - 1) as f64;

    // Effective noise: raw sigma / sqrt(total levels) — validated in Phase 1
    let effective_sigma = raw_sigma_w / (num_levels as f64).sqrt();

    // FIXED bounds (Bug 2 fix) — never recalculate from weight tensor
    let w_min: f64 = -1.0;
    let w_max: f64 = 1.0;
    let w_range: f64 = 2.0;

    for layer in &mut net.layers {
        for w in layer.weights.iter_mut() {
            // 1. Clamp to fixed bounds and normalize to [0, 1]
            let w_clamped = (*w).clamp(w_min, w_max);
            let w_norm = (w_clamped - w_min) / w_range;

            // 2. STOCHASTIC ROUNDING (Bug 3 fix)
            let level_float = w_norm * nl_f64;
            let lower = level_float.floor();
            let p_upper = level_float - lower;
            let chosen = if rng.gen::<f64>() < p_upper {
                (lower + 1.0).min(nl_f64)
            } else {
                lower.max(0.0)
            };

            // 3. Map to conductance and add multiplicative noise
            let g_quantized = g_min + (chosen / nl_f64) * g_range;
            let noise = rng.sample(noise_dist) * effective_sigma;
            let g_noisy = g_quantized * (1.0 + noise);
            let g_clamped = g_noisy.clamp(g_min, g_max);

            // 4. Map back to weight space using FIXED bounds
            *w = w_min + w_range * (g_clamped - g_min) / g_range;
        }
    }
}

// ─── Synthetic MNIST data generation ─────────────────────────────────

fn generate_synthetic_mnist(
    num_samples: usize,
    rng: &mut StdRng,
) -> Vec<(Array1<f64>, usize)> {
    // Multiple template variants per digit for more realistic variation.
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
        let variants = &templates[digit];
        let variant_idx = rng.gen_range(0..variants.len());
        let template = &variants[variant_idx];

        let mut img = ndarray::Array2::<f64>::zeros((28, 28));

        let jitter_x: i32 = rng.gen_range(-4..=4);
        let jitter_y: i32 = rng.gen_range(-4..=4);
        let scale = rng.gen_range(3.0..6.0_f64);

        // Random rotation angle in radians (+/-15 degrees)
        let angle: f64 = rng.gen_range(-0.26..0.26);
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let cx_center = 14.0_f64;
        let cy_center = 14.0_f64;

        // Random partial erasure
        let erasure_rate: f64 = rng.gen_range(0.0..0.20);

        for ty in 0..5 {
            for tx in 0..5 {
                if template[ty * 5 + tx] == 0 {
                    continue;
                }
                if rng.gen::<f64>() < erasure_rate {
                    continue;
                }

                let raw_cx = tx as f64 * scale + 4.0 + jitter_x as f64;
                let raw_cy = ty as f64 * scale + 4.0 + jitter_y as f64;

                let dx = raw_cx - cx_center;
                let dy = raw_cy - cy_center;
                let rot_cx = cx_center + dx * cos_a - dy * sin_a;
                let rot_cy = cy_center + dx * sin_a + dy * cos_a;

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

        // Background noise
        for y in 0..28 {
            for x in 0..28 {
                let noise_val: f64 = rng.sample(noise_dist);
                img[[y, x]] += noise_val.abs() * 0.12;
                img[[y, x]] = img[[y, x]].clamp(0.0, 1.0);
            }
        }

        img.into_shape_with_order(784).unwrap()
    };

    let mut data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let digit = i % 10;
        data.push((generate_sample(digit, rng), digit));
    }
    data.shuffle(rng);
    data
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
    // Fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate 5000 synthetic MNIST samples, split 4000 train + 1000 test
    let all_data = generate_synthetic_mnist(5000, &mut rng);
    let (train_data, test_data) = all_data.split_at(4000);
    let train_data: Vec<(Array1<f64>, usize)> = train_data.to_vec();
    let test_data: Vec<(Array1<f64>, usize)> = test_data.to_vec();

    // Build layer sizes
    let sizes: Vec<usize> = match config.model_type.as_str() {
        "perceptron" => vec![784, 10],
        "mlp_2h" => vec![784, config.hidden_size, config.hidden_size_2, 10],
        _ => vec![784, config.hidden_size, 10],
    };

    // Initialize ideal network
    let mut ideal_net = MLP::new(&sizes, &mut rng);

    // Memristor network starts from SAME initial weights
    let initial_weights = ideal_net.clone_weights();
    let mut mem_net = MLP::new(&sizes, &mut rng); // placeholder
    mem_net.load_weights(&initial_weights);

    // Adam optimizers for both networks
    // Bug 1 fix: divide lr by batch_size to get per-sample effective lr
    let lr = config.learning_rate / config.batch_size as f64;
    let mut ideal_adam = AdamState::new(&sizes, lr);
    let mut mem_adam = AdamState::new(&sizes, lr);

    // Device mapping parameters -- use provided g_min/g_max, fallback to defaults
    let g_min = if params.g_min > 0.0 { params.g_min } else { 0.1306e-6 };
    let g_max = if params.g_max > 0.0 { params.g_max } else { 0.2557e-6 };
    let sigma_w = params.write_noise;
    let num_levels_p = params.num_levels_p;
    let num_levels_d = params.num_levels_d;

    let mut results = Vec::new();

    for epoch in 0..config.epochs {
        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        let mut ideal_loss_sum = 0.0;
        let mut mem_loss_sum = 0.0;
        let mut total_samples = 0usize;

        // Mini-batch training
        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];
            let batch_size = batch_indices.len();

            // Collect batch inputs and targets
            let batch_inputs: Vec<&Array1<f64>> =
                batch_indices.iter().map(|&i| &train_data[i].0).collect();
            let batch_targets: Vec<usize> =
                batch_indices.iter().map(|&i| train_data[i].1).collect();

            // === IDEAL NETWORK ===
            {
                let (_logits, all_acts, all_zs) = ideal_net.forward_batch(&batch_inputs);

                // Compute loss and get softmax probs for each sample
                let mut all_probs = Vec::with_capacity(batch_size);
                for (i, acts) in all_acts.iter().enumerate() {
                    let logits = acts.last().unwrap();
                    let (loss, probs) = cross_entropy_loss(logits, batch_targets[i]);
                    ideal_loss_sum += loss;
                    all_probs.push(probs);
                }

                // Compute averaged gradients
                let (grad_w, grad_b) = compute_batch_gradients(
                    &ideal_net.layers,
                    &all_acts,
                    &all_zs,
                    &all_probs,
                    &batch_targets,
                );

                // Adam update
                ideal_adam.step(&mut ideal_net.layers, &grad_w, &grad_b);
            }

            // === MEMRISTOR NETWORK ===
            {
                let (_logits, all_acts, all_zs) = mem_net.forward_batch(&batch_inputs);

                let mut all_probs = Vec::with_capacity(batch_size);
                for (i, acts) in all_acts.iter().enumerate() {
                    let logits = acts.last().unwrap();
                    let (loss, probs) = cross_entropy_loss(logits, batch_targets[i]);
                    mem_loss_sum += loss;
                    all_probs.push(probs);
                }

                let (grad_w, grad_b) = compute_batch_gradients(
                    &mem_net.layers,
                    &all_acts,
                    &all_zs,
                    &all_probs,
                    &batch_targets,
                );

                // Adam update
                mem_adam.step(&mut mem_net.layers, &grad_w, &grad_b);

                // Apply device mapping AFTER weight update
                apply_device_mapping(
                    &mut mem_net,
                    g_min,
                    g_max,
                    sigma_w,
                    num_levels_p,
                    num_levels_d,
                    &mut rng,
                );
            }

            total_samples += batch_size;
        }

        // === EVALUATE ON TEST SET ===
        let mut ideal_correct = 0;
        for (input, target) in &test_data {
            if ideal_net.predict(input) == *target {
                ideal_correct += 1;
            }
        }

        let mut mem_correct = 0;
        for (input, target) in &test_data {
            if mem_net.predict(input) == *target {
                mem_correct += 1;
            }
        }

        let test_len = test_data.len() as f64;

        let result = ANNEpochResult {
            epoch,
            ideal_accuracy: ideal_correct as f64 / test_len * 100.0,
            memristor_accuracy: mem_correct as f64 / test_len * 100.0,
            ideal_loss: ideal_loss_sum / total_samples as f64,
            memristor_loss: mem_loss_sum / total_samples as f64,
        };

        let _ = window.emit("ann-progress", &result);
        results.push(result);
    }

    Ok(results)
}
