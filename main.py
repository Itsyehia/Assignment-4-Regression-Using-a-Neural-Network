import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, seed=None):
        """
        Initialize the neural network with given architecture and hyperparameters.

        :param input_size: Number of input neurons
        :param hidden_size: Number of hidden neurons
        :param output_size: Number of output neurons
        :param learning_rate: Learning rate for weight updates
        :param seed: Random seed for reproducibility
        """
        if seed:
            np.random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Calculate total number of weights
        # 20 in this case
        # 16 input, 4 output
        total_weights = input_size * hidden_size + hidden_size * output_size
        limit = 1 / total_weights  # Range is -1/total_weights to 1/total_weights

        # Initialize weights
        self.weights_input_hidden = np.random.uniform(
            -limit, limit, (self.hidden_size, self.input_size)
        )

        self.weights_hidden_output = np.random.uniform(
            -limit, limit, (self.output_size, self.hidden_size)
        )

    """works"""
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    """works"""
    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    """works"""
    def forward(self, inputs, target=None):
        """
        Perform the forward pass and optionally calculate the error.

        :param inputs: Input data array
        :param target: Target value (optional)
        :return: Tuple of hidden activations, output activations, and error (if target is provided)
        """
        # Store inputs
        self.input = inputs

        # Calculate hidden layer input as the sum of weighted inputs for each hidden node
        self.hidden_input = np.dot(self.weights_input_hidden, inputs)

        # Apply sigmoid function to calculate hidden layer output
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Calculate output layer input as the sum of weighted hidden outputs
        self.output_input = np.dot(self.weights_hidden_output, self.hidden_output)

        # Apply sigmoid function to calculate final output
        self.output = self.sigmoid(self.output_input)

        # Calculate error if target is provided
        error = None
        if target is not None:
            error = target - self.output

        return self.hidden_output, self.output, error

    """updated to follow our logic """
    def backward(self, target):
        """
        Perform the backward pass and update weights.

        :param target: The target value
        """
        # Step 1: Calculate error for output neuron
        # error = Output(1-Output)(Target – Output)
        output_error = self.output * (1 - self.output) * (target - self.output)

        # Step 2: Update weights for the output layer
        # WeightUpdated = weight + n * (error * hiddenNOutput)
        for i in range(self.output_size):  # For each output neuron
            for j in range(self.hidden_size):  # For each hidden layer neuron
                self.weights_hidden_output[i][j] += self.learning_rate * output_error[i] * self.hidden_output[j]

        # Step 3: Backpropagate error to hidden layer
        # error at hidden node = outputOfHiddenNode * (1 - outputOfHiddenNode) * (errorOfOutputNeuron)
        hidden_errors = self.hidden_output * (1 - self.hidden_output) * np.dot(output_error, self.weights_hidden_output[:, :])

        # Step 4: Update weights for the hidden layer
        # updatedHiddenLayer = hiddenlayerWeight + n * (error at this node) * (input to this edge)
        for j in range(self.hidden_size):  # For each hidden layer neuron
            for k in range(self.input_size):  # For each input to the hidden layer
                self.weights_input_hidden[j][k] += self.learning_rate * hidden_errors[j] * self.input[k]

    """updated to follow our logic """
    def train(self, X_train, y_train, epochs=1000, acceptable_error=0.01):
        """
        Train the neural network row by row.

        :param X_train: Training features
        :param y_train: Training targets
        :param epochs: Number of training epochs
        :param acceptable_error: Training stops if total error <= acceptable_error
        """
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X_train, y_train):
                # Step 1: Forward pass
                _, output, error = self.forward(inputs, target)

                # Step 2: Calculate squared error for this example
                squared_error = np.sum((target - output) ** 2)
                total_error += squared_error

                # Step 3: Check if total error is within acceptable limits
                if (0.5 * total_error) <= acceptable_error:  # Use 0.5 as per the formula
                    print(f"Training stopped at epoch {epoch} with total error: {0.5 * total_error:.4f}")
                    return

                # Step 4: Backward pass
                self.backward(target)

            # Step 5: Calculate and log the Mean Squared Error (MSE) for this epoch
            mse = (0.5 * total_error) / len(X_train)  # Divided by total samples as per formula

            # print every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Total Error: {0.5 * total_error:.4f}, MSE: {mse:.4f}")

    # to be removed maybe ?
    def predict(self, X):
        """
        Predict the target values for given inputs.

        :param X: Input features
        :return: Predicted outputs
        """
        predictions = []
        for inputs in X:
            _, output, error = self.forward(inputs)
            predictions.append(output[0])
        return np.array(predictions)


def reverse_scale_targets(predictions, min_target, max_target):
    """
    Reverse the scaling of the targets to original range.

    :param predictions: Scaled predictions
    :param min_target: Minimum target value in original data
    :param max_target: Maximum target value in original data
    :return: Rescaled predictions
    """
    return (predictions - 0.01) / 0.98 * (max_target - min_target) + min_target


def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    """
    Calculate the percentage of predictions within a specified tolerance.

    :param y_true: Actual target values
    :param y_pred: Predicted target values
    :param tolerance: Tolerance level (e.g., 0.1 for 10%)
    :return: Accuracy percentage
    """
    lower_bound = y_true * (1 - tolerance)
    upper_bound = y_true * (1 + tolerance)
    accurate = np.logical_and(y_pred >= lower_bound, y_pred <= upper_bound)
    accuracy = np.mean(accurate) * 100
    return accuracy


def load_and_preprocess_data(file_path, normalization_method='minmax'):
    data = pd.read_excel(file_path)

    features = data.iloc[:, 0:4].values
    targets = data.iloc[:, 4].values.reshape(-1, 1)

    if normalization_method == 'minmax':
        features_min = features.min(axis=0)
        features_max = features.max(axis=0)
        features_normalized = (features - features_min) / (features_max - features_min)

    elif normalization_method == 'variance':
        mean = np.mean(features, axis=0)
        std_dev = np.std(features, axis=0)
        features_normalized = (features - mean) / std_dev

    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    min_target = targets.min()
    max_target = targets.max()
    targets_scaled = (targets - min_target) / (max_target - min_target)

    return features_normalized, targets_scaled, min_target, max_target

def main():
    # Load and preprocess data
    file_path = 'cleaned_concrete_data.xlsx'  # Ensure this file is in the working directory

    # Select the normalization method
    normalization_method = 'minmax'  # Options: 'euclidean', 'percentage', 'variance'

    # Load data with the selected normalization method
    features, targets, normalization_params, min_target, max_target = load_and_preprocess_data(
        file_path, normalization_method=normalization_method
    )

    # Split data into training and testing sets (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42
    )

    # Get min and max of original targets for rescaling later
    data = pd.read_excel(file_path)
    original_targets = data.iloc[:, 4].values
    min_target = original_targets.min()
    max_target = original_targets.max()

    # Initialize Neural Network
    input_size = X_train.shape[1]  # 4 columns
    hidden_size = 16  # number of nodes in the hidden layer
    output_size = 1
    learning_rate = 0.1
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, seed=42)

    # Train the network
    nn.train(X_train, y_train, epochs=2000, acceptable_error=0.00057)

    # **Evaluate on training data**
    train_predictions_scaled = nn.predict(X_train)
    train_predictions = reverse_scale_targets(train_predictions_scaled, min_target, max_target)
    y_train_original = reverse_scale_targets(y_train, min_target, max_target)

    # Calculate MSE on training data
    train_mse_original = np.mean((y_train_original - train_predictions) ** 2)
    train_mse_scaled = np.mean(0.5 * (y_train - train_predictions_scaled) ** 2)
    print(f"Training Data: Original MSE: {train_mse_original:.4f}, Scaled MSE: {train_mse_scaled:.4f}")

    # Calculate accuracy for training data
    train_accuracy = calculate_accuracy(y_train_original, train_predictions, tolerance=0.1)
    print(f"Training Data: Accuracy within 10% tolerance: {train_accuracy:.2f}%")

    # **Evaluate on testing data**
    test_predictions_scaled = nn.predict(X_test)
    test_predictions = reverse_scale_targets(test_predictions_scaled, min_target, max_target)
    y_test_original = reverse_scale_targets(y_test, min_target, max_target)

    # Calculate MSE on testing data
    test_mse_original = np.mean((y_test_original - test_predictions) ** 2)
    test_mse_scaled = np.mean(0.5 * (y_test - test_predictions_scaled) ** 2)
    print(f"Testing Data: Original MSE: {test_mse_original:.4f}, Scaled MSE: {test_mse_scaled:.4f}")

    # Calculate accuracy for testing data
    test_accuracy = calculate_accuracy(y_test_original, test_predictions, tolerance=0.1)
    print(f"Testing Data: Accuracy within 10% tolerance: {test_accuracy:.2f}%")

    # **Interactive Prediction**
    print("\n--- Concrete Compressive Strength Prediction ---")
    print("Enter 'exit' at any time to quit.\n")

    while True:
        try:
            # Prompt user for input features
            print("Please enter the following features:")

            cement = input("Cement (kg/m^3): ")
            if cement.lower() == 'exit':
                break
            cement = float(cement)

            water = input("Water (kg/m^3): ")
            if water.lower() == 'exit':
                break
            water = float(water)

            superplasticizer = input("Superplasticizer (kg/m^3): ")
            if superplasticizer.lower() == 'exit':
                break
            superplasticizer = float(superplasticizer)

            age = input("Age (days): ")
            if age.lower() == 'exit':
                break
            age = float(age)

            # Create input array
            user_input = np.array([cement, water, superplasticizer, age])

            # Normalize the input based on the selected normalization method
            if normalization_method == 'euclidean':
                norm = normalization_params['norms']
                user_input_normalized = user_input / norm.flatten()

            elif normalization_method == 'percentage':
                row_sum = normalization_params['row_sums']
                user_input_normalized = user_input / row_sum.flatten()

            elif normalization_method == 'variance':
                mean = normalization_params['mean'].flatten()
                std_dev = normalization_params['std_dev'].flatten()
                user_input_normalized = (user_input - mean) / std_dev

            else:
                raise ValueError(f"Unknown normalization method: {normalization_method}")

            # Predict using the neural network
            _, output_scaled, _ = nn.forward(user_input_normalized)
            output = reverse_scale_targets(output_scaled, min_target, max_target)[0]

            print(f"Predicted Concrete Compressive Strength: {output:.2f} MPa\n")

        except ValueError:
            print("Invalid input. Please enter numerical values for all features.\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()
