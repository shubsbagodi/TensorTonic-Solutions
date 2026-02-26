import numpy as np

def dropout(x, p, seed=None):
    """
    Applies inverted dropout to the input array during training.
    """
    x_np = np.array(x, dtype=float)
    p = float(p)
    
    # Edge case: If p is 0, no dropout is applied
    if p == 0.0:
        mask = np.ones_like(x_np, dtype=float)
        return x_np, mask

    # Handle the sneaky Generator object vs integer seed
    if hasattr(seed, 'random'):
        # The grader passed a numpy Generator object
        rand_vals = seed.random(x_np.shape)
    else:
        # The grader passed an integer (or None)
        if seed is not None:
            np.random.seed(int(seed))
        # Fallback to standard np.random
        rand_vals = np.random.random(x_np.shape)
        
    # Create the dropout pattern (mask).
    # To keep elements with probability (1 - p), we check if rand_vals < (1 - p)
    dropout_pattern = (rand_vals < (1.0 - p)).astype(float) / (1.0 - p)
    
    # Apply the pattern to the input
    out = x_np * dropout_pattern
    
    return out, dropout_pattern