import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    """
    Computes the sinusoidal positional encoding matrix.
    """
    # Initialize the positional encoding matrix
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Create a column vector for positions: [0, 1, 2, ..., seq_len - 1]^T
    position = np.arange(seq_len)[:, np.newaxis]
    
    # Compute the division terms for the angles
    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)
    
    # Compute the angles for all positions and even dimension pairs via broadcasting
    angles = position / div_term
    
    # Apply sine to all even-indexed columns: 0, 2, 4...
    pe[:, 0::2] = np.sin(angles)
    
    # Apply cosine to all odd-indexed columns: 1, 3, 5...
    # slicing safely drops the un-paired last column if d_model is odd
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
    
    return pe