# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import RobustScaler

SEED = 42

# Import the data and scale the time and amount columns
credit_df = pd.read_csv("data/creditcard.csv")
scaler = RobustScaler().fit(credit_df[["Time", "Amount"]])
credit_df[["Time", "Amount"]] = scaler.transform(credit_df[["Time", "Amount"]])

# Shuffle the dataset and split into training and test sets
credit_df = credit_df.sample(frac=1, random_state=42).reset_index(drop=True)
X = credit_df.drop(columns=['Class'])
y = credit_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Loss functions
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))


# Initializer to improve gradient time
def xavier_init(size):
    return np.random.randn(*size) * np.sqrt(2 / (size[0] + size[1]))
    
class Autoencoder:
    def __init__(self, input_dim, hidden_dims):
        self.weights = []
        self.biases = []
        
        # Encoder layers
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.weights.append(xavier_init((prev_dim, h_dim)))
            self.biases.append(np.zeros((1, h_dim)))
            prev_dim = h_dim

        # Decoder layers (symmetrical)
        for h_dim in reversed(hidden_dims[:-1]):
            self.weights.append(xavier_init((prev_dim, h_dim)))
            self.biases.append(np.zeros((1, h_dim)))
            prev_dim = h_dim
        
        # Output layer
        self.weights.append(xavier_init((prev_dim, input_dim)))
        self.biases.append(np.zeros((1, input_dim)))

    def forward(self, x):
        self.activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            x = relu(x)
            self.activations.append(x)
        return x

    def backward(self, x, learning_rate=0.01):
        delta = (self.activations[-1] - x) * relu_derivative(self.activations[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(self.activations[i].T, delta) / x.shape[0]
            grad_b = np.mean(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.activations[i])
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b

    def train(self, x, epochs, learning_rate=0.01):
        x = np.array(x)
        for epoch in range(epochs):
            output = self.forward(x)
            loss = mean_squared_error(x, output)
            self.backward(x, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return loss

def per_sample_mse(model, generator):
    batch_losses = []
    for x_batch, y_batch in generator:
        y_pred = model.forward(x_batch)
        loss = mean_squared_error(y_pred.squeeze(),y_batch)
        batch_losses.append(loss)

    return batch_losses


def data_generator(X, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], X[i:i+batch_size]

hidden_dims = [100, 20]
model = Autoencoder(X_train.shape[1], hidden_dims)
batch_size = 32
generator = data_generator(X_train, batch_size)
losses = per_sample_mse(model, generator)

# Evaluate the model
train_errors = np.mean((X_train - model.forward(X_train))**2, axis=1)
threshold = np.percentile(train_errors, 97.5)
test_errors = np.mean((X_test - model.forward(X_test))**2, axis=1)
y_pred = test_errors > threshold

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:")
print(round(roc_auc_score(y_test, test_errors),4))


"""
Some numbers to further investigate the model's predictions

# Define false positives and false negatives
false_positives = np.sum((y_pred == 1) & (y_test == 0))
false_negatives = np.sum((y_pred == 0) & (y_test == 1))

print(len(y_test), false_positives, false_negatives)

# Show example with prediction
print(X_test.iloc[0], y_test[0], y_pred[0])"""
