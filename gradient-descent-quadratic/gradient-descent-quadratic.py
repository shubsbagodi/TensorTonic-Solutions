def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimizes a 1-D quadratic function f(x) = ax^2 + bx + c using gradient descent.
    """
    # Initialize x with the starting value
    x = float(x0)
    
    # Perform gradient descent for the specified number of steps
    for _ in range(steps):
        # Compute the derivative of f(x) = ax^2 + bx + c, which is f'(x) = 2ax + b
        grad = 2 * a * x + b
        
        # Update x by moving in the opposite direction of the gradient
        x = x - lr * grad
        
    return x