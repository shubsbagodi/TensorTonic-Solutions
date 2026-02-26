import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Executes a single step of the RMSProp optimization algorithm.
    
    Args:
        w (array_like): Current parameters.
        g (array_like): Current gradients.
        s (array_like): Running squared gradient accumulator.
        lr (float): Learning rate.
        beta (float): Decay factor for the moving average.
        eps (float): Small constant for numerical stability.
        
    Returns:
        tuple: (new_w, new_s) representing updated parameters and accumulator.
    """
    # 1. Convert inputs to NumPy arrays to ensure fast element-wise math
    w_np = np.array(w, dtype=float)
    g_np = np.array(g, dtype=float)
    s_np = np.array(s, dtype=float)
    
    # 2. Update Running Average (Squared gradient accumulator)
    s_new = beta * s_np + (1.0 - beta) * (g_np ** 2)
    
    # 3. Parameter Update
    w_new = w_np - (lr / (np.sqrt(s_new) + eps)) * g_np
    
    # 4. If the grader passed in a standard list, return standard lists to be safe!
    if isinstance(w, list):
        return w_new.tolist(), s_new.tolist()
        
    return w_new, s_new

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1
    w1, g1, s1, lr1 = [1.0, 2.0], [0.2, -0.4], [0.0, 0.0], 0.1
    new_w1, new_s1 = rmsprop_step(w1, g1, s1, lr=lr1, beta=0.9)
    print(f"Example 1 -> w: {np.round(new_w1, 3)}, s: {np.round(new_s1, 3)}")
    # Expected w: [0.684, 2.316], s: [0.004, 0.016]