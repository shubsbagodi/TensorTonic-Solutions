import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid function."""
    # Clipping limits z to prevent overflow/underflow in np.exp
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    """
    Trains a binary logistic regression classifier using gradient descent.
    
    Args:
        X (list or np.ndarray): Input features of shape (N, D)
        y (list or np.ndarray): Target labels (0 or 1) of shape (N,)
        lr (float): Learning rate
        steps (int): Number of gradient descent iterations
        
    Returns:
        tuple: Learned weights (w) of shape (D,) and bias (b) as a float
    """
    # Convert inputs to NumPy arrays in case they are lists
    X = np.array(X)
    y = np.array(y)
    
    N, D = X.shape
    
    # 1. Initialize weights as zeros and bias as 0.0
    w = np.zeros(D)
    b = 0.0
    
    # 2. Training Loop
    for _ in range(steps):
        # --- Forward Pass ---
        # Calculate linear combinations
        z = np.dot(X, w) + b
        
        # Apply sigmoid activation to get probabilities
        p = _sigmoid(z)
        
        # --- Compute Gradients ---
        # The derivative of the Binary Cross-Entropy loss with respect to z
        dz = p - y
        
        # Gradients for weights and bias
        dw = (1 / N) * np.dot(X.T, dz)
        db = (1 / N) * np.sum(dz)
        
        # --- Parameter Update ---
        w -= lr * dw
        b -= lr * db
        
    return w, float(b)

# --- Example Usage ---
if __name__ == "__main__":
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    
    w, b = train_logistic_regression(X, y, lr=0.1, steps=500)
    print(f"Learned parameters:\nw: {w}\nb: {b}")
    
    # Check predictions
    predictions = _sigmoid(np.dot(X, w) + b) >= 0.5
    print(f"Predictions: {predictions.astype(int)}")