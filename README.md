# Soft Computing Course - 2024  
## Assignment 4 – Regression Using a Neural Network

### Overview
This project involves implementing a Feedforward Neural Network (FFNN) from scratch to predict the concrete compressive strength based on construction data. The neural network has three layers: input, one hidden, and output. The network is trained using the provided dataset (`concrete_data.xlsx`) and utilizes the sigmoid activation function in the hidden layer.

### Problem Description
Regression is a supervised learning problem that aims to estimate the relationships between dependent and independent variables. In this project, we use a neural network to predict the cement strength (the target) based on input features such as cement, water, superplasticizer, and age.

### Data
The dataset (`concrete_data.xlsx`) contains 700 records, each with 5 columns:
- **Cement**: Amount of cement used (kg/m³)
- **Water**: Amount of water used (kg/m³)
- **Superplasticizer**: Amount of superplasticizer used (kg/m³)
- **Age**: Age of the concrete (days)
- **Concrete Strength**: The target value to be predicted (MPa)

### Objective
You are required to:
1. Implement a Feedforward Neural Network (FFNN) from scratch with:
   - 3 Layers: Input, Hidden (1 layer), and Output.
   - Sigmoid activation function for the hidden layer.
   
2. Preprocess the provided dataset:
   - Normalize the input features using Min-Max or Variance normalization.
   - Split the dataset into **training (75%)** and **testing (25%)** sets.

3. Implement the `NeuralNetwork` class with the following functionalities:
   - **Architecture Setup**: Set the number of neurons in each layer and the learning rate.
   - **Training**: Train the network for a specified number of epochs using forward and backward propagation.
   - **Prediction**: Predict concrete strength for new input data based on the trained model.
   - **Error Calculation**: Calculate the Mean Squared Error (MSE) for the model on both training and testing sets.

### Instructions
1. **Set Up the Environment**:
   - Install Python 3.x and required packages (`numpy`, `pandas`, `scikit-learn`).
   - Ensure the file `concrete_data.xlsx` is located in the working directory.

2. **Project Structure**:
   - `neural_network.py`: Contains the implementation of the `NeuralNetwork` class and associated methods.
   - `concrete_data.xlsx`: The dataset file for training and testing the model.
   - `README.md`: This file.

3. **How to Run**:
   1. Clone or download this repository.
   2. Install necessary Python packages:
      ```bash
      pip install numpy pandas scikit-learn
      ```
   3. Run the script to train the neural network and make predictions.
      ```bash
      python neural_network.py
      ```

### Classes and Methods
- **NeuralNetwork**:
  - `__init__(self, input_size, hidden_size, output_size, learning_rate=0.5, seed=None)`: Initializes the neural network with given architecture and hyperparameters.
  - `sigmoid(self, x)`: Sigmoid activation function.
  - `sigmoid_derivative(self, x)`: Derivative of the sigmoid function.
  - `forward(self, inputs, target=None)`: Forward pass through the network.
  - `backward(self, target)`: Backward pass using backpropagation.
  - `train(self, X_train, y_train, epochs=1000, acceptable_error=0.01)`: Train the network with training data and update weights.
  - `predict(self, X)`: Predict the target values for new input data.

- **Utility Functions**:
  - `reverse_scale_targets(predictions, min_target, max_target)`: Reverse scaling of the target predictions.
  - `calculate_accuracy(y_true, y_pred, tolerance=0.1)`: Calculate the accuracy of predictions within a given tolerance.
  - `load_and_preprocess_data(file_path, normalization_method='minmax')`: Load and preprocess the data, normalize features, and scale targets.

### Example Usage
```python
# Initialize Neural Network
input_size = X_train.shape[1]  # Number of input features
hidden_size = 16  # Number of neurons in hidden layer
output_size = 1  # Number of output neurons
learning_rate = 0.1
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, seed=42)

# Train the network
nn.train(X_train, y_train, epochs=2000, acceptable_error=0.00057)

# Evaluate predictions on training data
train_predictions_scaled = nn.predict(X_train)
train_predictions = reverse_scale_targets(train_predictions_scaled, min_target, max_target)
y_train_original = reverse_scale_targets(y_train, min_target, max_target)
train_mse = np.mean((y_train_original - train_predictions) ** 2)
train_accuracy = calculate_accuracy(y_train_original, train_predictions)
print(f"Training MSE: {train_mse:.4f}, Training Accuracy: {train_accuracy:.2f}%")

# Evaluate predictions on testing data
test_predictions_scaled = nn.predict(X_test)
test_predictions = reverse_scale_targets(test_predictions_scaled, min_target, max_target)
y_test_original = reverse_scale_targets(y_test, min_target, max_target)
test_mse = np.mean((y_test_original - test_predictions) ** 2)
test_accuracy = calculate_accuracy(y_test_original, test_predictions)
print(f"Testing MSE: {test_mse:.4f}, Testing Accuracy: {test_accuracy:.2f}%")
```


### Notes
- Ensure the input data is normalized correctly based on the selected normalization method (`minmax`, `variance`, etc.).
- The network uses backpropagation to minimize the error, adjusting weights after each training iteration.
- The training process will stop early if the error reaches an acceptable threshold.

### License
This project is for educational purposes only.

---

