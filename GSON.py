import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# Self-Repairing Neural Network Class with Accuracy and Forced Transition
class SelfRepairingNeuralNetwork:
    def __init__(self, input_size=10, hidden_size=20, output_size=5, num_weights=1000, max_spherical_steps=5):
        # Neural Network Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

        # Geometric Space Metrics
        self.num_weights = num_weights
        self.weights = np.random.uniform(-1, 1, num_weights)
        self.energy = np.sum(np.abs(self.weights))
        self.entropy = self.calculate_entropy()
        self.time_dimension = 2.41
        self.geometric_space = 'Spherical'
        self.spherical_counter = 0
        self.max_spherical_steps = max_spherical_steps

        # History for Metrics
        self.history = {'Energy': [], 'Entropy': [], 'Space': [], 'Time': []}

    def forward(self, x):
        """ Forward pass of the neural network. """
        hidden = np.tanh(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        return output

    def backpropagate(self, x, y, lr=0.01):
        """ Backpropagation for training the network. """
        # Forward pass
        hidden = np.tanh(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output

        # Loss and gradients
        loss = np.mean((output - y) ** 2)
        output_grad = 2 * (output - y)
        grad_weights_hidden_output = np.dot(hidden.T, output_grad)
        grad_bias_output = np.sum(output_grad, axis=0)

        hidden_grad = np.dot(output_grad, self.weights_hidden_output.T) * (1 - hidden**2)
        grad_weights_input_hidden = np.dot(x.T, hidden_grad)
        grad_bias_hidden = np.sum(hidden_grad, axis=0)

        # Update weights and biases
        self.weights_hidden_output -= lr * grad_weights_hidden_output
        self.bias_output -= lr * grad_bias_output
        self.weights_input_hidden -= lr * grad_weights_input_hidden
        self.bias_hidden -= lr * grad_bias_hidden

        return loss

    def calculate_entropy(self):
        """ Calculate entropy as a measure of weight dispersion. """
        prob, _ = np.histogram(self.weights, bins=20, density=True)
        prob = prob[prob > 0]
        if len(prob) == 0:
            return 0.0
        return -np.sum(prob * np.log(prob))

    def adjust_weights_fractal(self):
        """ Energy regeneration in fractal space. """
        fractal_dimension = np.random.uniform(1.0, 2.41)
        self.weights += np.random.normal(0, 0.1, size=self.num_weights) / fractal_dimension
        self.energy = np.sum(np.abs(self.weights))

    def adjust_weights_pyramid(self):
        """ Entropy regulation in pyramid space. """
        self.weights *= np.random.uniform(0.5, 0.9)
        self.entropy = self.calculate_entropy()

    def adjust_weights_spherical(self):
        """ Free thinking in spherical space. """
        self.weights += np.random.uniform(-0.5, 0.5, size=self.num_weights)
        self.entropy = self.calculate_entropy()

    def detect_environment_and_navigate(self):
        """ Detect energy, entropy, and navigate spaces. Force Pyramid after N spherical steps. """
        if self.geometric_space == 'Spherical':
            self.spherical_counter += 1
        else:
            self.spherical_counter = 0

        # Force Pyramid Space after max_spherical_steps
        if self.spherical_counter >= self.max_spherical_steps:
            self.geometric_space = 'Pyramid'
            self.adjust_weights_pyramid()
            self.spherical_counter = 0  # Reset counter
        elif self.energy < 50:
            self.geometric_space = 'Fractal'
            self.adjust_weights_fractal()
        elif self.entropy < 2.0:
            self.geometric_space = 'Pyramid'
            self.adjust_weights_pyramid()
        else:
            self.geometric_space = 'Spherical'
            self.adjust_weights_spherical()

        # Time stabilization
        self.time_dimension = 2.41 - (self.entropy / 10)
        self.energy = np.sum(np.abs(self.weights))
        self.entropy = self.calculate_entropy()
        self.history['Energy'].append(self.energy)
        self.history['Entropy'].append(self.entropy)
        self.history['Space'].append(self.geometric_space)
        self.history['Time'].append(self.time_dimension)

    def calculate_accuracy(self, X, Y):
        """ Calculate accuracy based on predictions. """
        predictions = self.forward(X)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(Y, axis=1)
        return accuracy_score(true_labels, predicted_labels) * 100

    def train_with_accuracy(self, X, Y, steps=100, lr=0.01, target_accuracy=98):
        """ Train the network, stop if target accuracy is reached. """
        for step in range(steps):
            loss = self.backpropagate(X, Y, lr)
            self.detect_environment_and_navigate()
            accuracy = self.calculate_accuracy(X, Y)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, "
                      f"Energy: {self.energy:.2f}, Entropy: {self.entropy:.2f}, Space: {self.geometric_space}")
            if accuracy >= target_accuracy:
                print(f"Stopping training at Step {step}, Accuracy: {accuracy:.2f}% achieved!")
                break
        return accuracy

# Generate Dummy Data
np.random.seed(42)
X_train = np.random.randn(100, 10)
Y_train = np.eye(5)[np.random.choice(5, 100)]  # One-hot encoded targets

# Hyperparameter tuning
param_grid = {
    'hidden_size': [10, 20, 30],
    'lr': [0.01, 0.005, 0.001],
    'max_spherical_steps': [3, 5, 7]
}

best_accuracy = 0.0
best_params = None
best_model = None

for hs in param_grid['hidden_size']:
    for lr in param_grid['lr']:
        for mss in param_grid['max_spherical_steps']:
            print(f"Training with hidden_size={hs}, lr={lr}, max_spherical_steps={mss}")
            model = SelfRepairingNeuralNetwork(hidden_size=hs, max_spherical_steps=mss)
            acc = model.train_with_accuracy(X_train, Y_train, steps=200, lr=lr, target_accuracy=98)
            print(f"Final Accuracy: {acc:.2f}%\n")

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = (hs, lr, mss)
                best_model = model

            # If we already hit or exceeded 98%, we can stop hyperparameter tuning early
            if best_accuracy >= 98:
                break
        if best_accuracy >= 98:
            break
    if best_accuracy >= 98:
        break

print("Best Accuracy Achieved:", best_accuracy)
print("Best Params: hidden_size=%d, lr=%.4f, max_spherical_steps=%d" % best_params)

# Visualization of the best model metrics if available
if best_model is not None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Energy', color=color)
    ax1.plot(best_model.history['Energy'], color=color, label='Energy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Entropy', color=color)
    ax2.plot(best_model.history['Entropy'], color=color, linestyle='--', label='Entropy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Energy and Entropy Dynamics with Forced Pyramid Transitions (Best Model)')
    plt.show()

    # Final Metrics
    final_metrics = pd.DataFrame({
        'Geometric Space': best_model.history['Space'],
        'Time Dimension': best_model.history['Time']
    })
    print(final_metrics.tail(10))
