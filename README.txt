
This will train the model for 10 epochs and generate an accuracy plot.

## Project Structure

- `src/main.rs`: Contains the main code including model definition, training loop, and accuracy plotting
- `accuracy_plot.png`: Generated plot showing test accuracy over epochs

## Model Architecture

- Conv2D (1 -> 32 channels, 3x3 kernel)
- ReLU
- MaxPool2D
- Conv2D (32 -> 64 channels, 3x3 kernel)
- ReLU
- MaxPool2D
- Fully Connected (1600 -> 128)
- ReLU
- Fully Connected (128 -> 10)

## License

[MIT License](LICENSE)