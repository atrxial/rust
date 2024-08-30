use ndarray::{Array, Array1, Array2, Array4, Axis};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};

// Activation functions
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|a| (a - max).exp());
    let sum = exp.sum();
    exp / sum
}

// Layers
struct Conv2D {
    filters: Array4<f32>,
    bias: Array1<f32>,
}

impl Conv2D {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.1, 0.1);
        
        Conv2D {
            filters: Array::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), 
                |_| dist.sample(&mut rng)),
            bias: Array::zeros(out_channels),
        }
    }

    fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, in_channels, height, width) = input.dim();
        let (out_channels, _, kernel_size, _) = self.filters.dim();
        let out_height = height - kernel_size + 1;
        let out_width = width - kernel_size + 1;

        let mut output = Array4::<f32>::zeros((batch_size, out_channels, out_height, out_width));

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let mut sum = 0.0;
                        for ic in 0..in_channels {
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    sum += input[[b, ic, h + kh, w + kw]] * self.filters[[oc, ic, kh, kw]];
                                }
                            }
                        }
                        output[[b, oc, h, w]] = sum + self.bias[oc];
                    }
                }
            }
        }

        output
    }
}

struct Dense {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl Dense {
    fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-0.1, 0.1);
        
        Dense {
            weights: Array::from_shape_fn((in_features, out_features), |_| dist.sample(&mut rng)),
            bias: Array::zeros(out_features),
        }
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.bias
    }
}

// CNN Model
struct CNN {
    conv1: Conv2D,
    conv2: Conv2D,
    dense1: Dense,
    dense2: Dense,
}

impl CNN {
    fn new() -> Self {
        CNN {
            conv1: Conv2D::new(1, 32, 3),
            conv2: Conv2D::new(32, 64, 3),
            dense1: Dense::new(64 * 5 * 5, 128),
            dense2: Dense::new(128, 10),
        }
    }

    fn forward(&self, input: &Array4<f32>) -> Array2<f32> {
        let x = self.conv1.forward(input).mapv(relu);
        let x = x.slice_axis(Axis(2), ndarray::Slice::new(0, Some(-1), 2))
                 .slice_axis(Axis(3), ndarray::Slice::new(0, Some(-1), 2));
        
        let x = self.conv2.forward(&x).mapv(relu);
        let x = x.slice_axis(Axis(2), ndarray::Slice::new(0, Some(-1), 2))
                 .slice_axis(Axis(3), ndarray::Slice::new(0, Some(-1), 2));
        
        let batch_size = x.shape()[0];
        let x = x.into_shape((batch_size, 64 * 5 * 5)).unwrap();
        
        let x = self.dense1.forward(&x).mapv(relu);
        self.dense2.forward(&x)
    }

    fn predict(&self, input: &Array4<f32>) -> Array2<f32> {
        let output = self.forward(input);
        output.map_axis(Axis(1), |row| softmax(&row))
    }
}

// Training function (simplified, without optimization)
fn train(model: &mut CNN, inputs: &Array4<f32>, targets: &Array2<f32>, learning_rate: f32) {
    // Forward pass
    let output = model.forward(inputs);
    
    // Compute loss (cross-entropy)
    let loss = -(&output * targets).sum() / inputs.shape()[0] as f32;
    
    // Backpropagation and parameter update would go here
    // For brevity, we're omitting the actual training logic
    
    println!("Loss: {}", loss);
}

fn main() {
    // Initialize model
    let mut model = CNN::new();

    // Generate dummy data (replace with actual MNIST data)
    let batch_size = 64;
    let inputs = Array4::<f32>::zeros((batch_size, 1, 28, 28));
    let targets = Array2::<f32>::zeros((batch_size, 10));

    // Training loop
    for epoch in 0..10 {
        println!("Epoch {}", epoch);
        train(&mut model, &inputs, &targets, 0.01);
    }

    // Make predictions
    let predictions = model.predict(&inputs);
    println!("Predictions shape: {:?}", predictions.shape());
}