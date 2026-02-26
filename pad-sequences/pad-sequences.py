import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pads or truncates a list of sequences to a uniform length.
    
    Args:
        seqs (list of lists): The input token ID sequences.
        pad_value (int): The value to use for padding.
        max_len (int or None): The target length for all sequences.
        
    Returns:
        np.ndarray: A 2D array of padded sequences of shape (N, max_len).
    """
    # Handle empty input case
    if not seqs:
        return np.empty((0, 0), dtype=int)
        
    # Determine the target max_len if not explicitly provided
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
        
    N = len(seqs)
    
    # Pre-allocate the NumPy array filled entirely with the pad_value
    padded_array = np.full((N, max_len), fill_value=pad_value, dtype=int)
    
    # Copy original sequences into the padded array (with truncation if necessary)
    for i, seq in enumerate(seqs):
        copy_len = min(len(seq), max_len)
        if copy_len > 0:
            padded_array[i, :copy_len] = seq[:copy_len]
            
    return padded_array

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Auto-detected max length
    seqs1 = [[1, 2, 3], [4, 5], [6]]
    print("Example 1 Output:")
    print(pad_sequences(seqs1, pad_value=0))
    # Expected: [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
    
    print("\nExample 2 Output:")
    # Example 2: Explicit max length with truncation
    seqs2 = [[1, 2, 3, 4], [5, 6]]
    print(pad_sequences(seqs2, pad_value=-1, max_len=3))
    # Expected: [[1, 2, 3], [5, 6, -1]]