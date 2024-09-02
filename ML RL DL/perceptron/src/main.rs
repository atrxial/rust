use std::error::Error; // Import the Error trait
use std::path::Path; // Import the Path trait
use std::fs::File; // Import the File struct
use std::io::{BufRead, BufReader}; // Import the BufRead and BufReader traits




fn main() -> Result<(), Box<dyn Error>> {
    // Read the dataset from a CSV file
    let dataset = read_dataset("files/dataset.csv")?;

    // Set the learning rate and number of epochs
    let learning_rate = 0.01;
    let epochs = 200;

    // Initialize the weights and biases for the first layer
    let mut weights1 = [[0.0, 0.0], [0.0, 0.0]];
    let mut biases1 = [0.0, 0.0];

    // Initialize the weights and bias for the second layer
    let mut weights2 = [0.0, 0.0];
    let mut bias2 = 0.0;

    // Train the perceptron
    for _ in 0..epochs {
        for (x1, x2, label) in &dataset { // Iterate over the dataset, The & symbol means that the loop borrows the elements, avoiding ownership transfer and allowing the dataset to be used elsewhere in the code.
            // Calculate the output of the first layer
            let output1 = [
                sigmoid(weights1[0][0] * x1 + weights1[0][1] * x2 + biases1[0]),
                sigmoid(weights1[1][0] * x1 + weights1[1][1] * x2 + biases1[1]),
            ];

            // Calculate the output of the second layer
            let output2 = sigmoid(weights2[0] * output1[0] + weights2[1] * output1[1] + bias2);

            // Calculate the predicted label
            let predicted_label = if output2 >= 0.0 { 1 } else { -1 };

            // Update the weights and biases based on the prediction error
            let error = label - predicted_label;

            // Update the weights and biases of the second layer
            weights2[0] += learning_rate * error as f64 * output1[0];
            weights2[1] += learning_rate * error as f64 * output1[1];
            bias2 += learning_rate * error as f64;

            // Update the weights and biases of the first layer
            weights1[0][0] += learning_rate * error as f64 * x1 * weights2[0] * sigmoid_derivative(output1[0]);
            weights1[0][1] += learning_rate * error as f64 * x2 * weights2[0] * sigmoid_derivative(output1[0]);
            weights1[1][0] += learning_rate * error as f64 * x1 * weights2[1] * sigmoid_derivative(output1[1]);
            weights1[1][1] += learning_rate * error as f64 * x2 * weights2[1] * sigmoid_derivative(output1[1]);
            biases1[0] += learning_rate * error as f64 * weights2[0] * sigmoid_derivative(output1[0]);
            biases1[1] += learning_rate * error as f64 * weights2[1] * sigmoid_derivative(output1[1]);
        }
    }

    // Test the perceptron
    let mut correct_predictions = 0;
    for (x1, x2, label) in &dataset {
        // Calculate the output of the first layer
        let output1 = [
            sigmoid(weights1[0][0] * x1 + weights1[0][1] * x2 + biases1[0]),
            sigmoid(weights1[1][0] * x1 + weights1[1][1] * x2 + biases1[1]),
        ];

        // Calculate the output of the second layer
        let output2 = sigmoid(weights2[0] * output1[0] + weights2[1] * output1[1] + bias2);

        // Calculate the predicted label
        let predicted_label = if output2 <= 0.5 { 1 } else { -1 };

        if label == &predicted_label {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f64 / dataset.len() as f64 * 100.0;
    println!("Accuracy: {}%", accuracy);

    Ok(())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}




fn read_dataset<P: AsRef<Path>>(file_path: P) -> Result<Vec<(f64, f64, i32)>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut dataset = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let values: Vec<&str> = line.split(',').collect();

        let x1: Result<f64, _> = values[0].parse();
        let x2: Result<f64, _> = values[1].parse();
        let label: Result<i32, _> = values[2].parse();

        if let (Ok(x1), Ok(x2), Ok(label)) = (x1, x2, label) {
            dataset.push((x1, x2, label));
        } else {
            continue; // Skip the current iteration and proceed to the next line in the dataset
        }
    }

    Ok(dataset)
}
