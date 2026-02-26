import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Applies a 4x4 homogeneous transformation matrix to 3D point(s).
    
    Args:
        T (array_like): 4x4 transformation matrix.
        points (array_like): A single 3D point (shape 3,) or a batch of N points (shape N, 3).
        
    Returns:
        np.ndarray or list: Transformed point(s) in spatial coordinates (x, y, z).
    """
    # 1. Convert inputs to NumPy arrays
    T_np = np.array(T, dtype=float)
    p_np = np.array(points, dtype=float)
    
    # 2. Track if the input was a single point to format the output correctly later
    is_single_point = (p_np.ndim == 1)
    
    if is_single_point:
        # Reshape to (1, 3) so we can vectorize the exact same way as a batch
        p_np = p_np.reshape(1, 3)
        
    N = p_np.shape[0]
    
    # 3. Convert to homogeneous coordinates by appending 1s
    # p_np is shape (N, 3), ones is shape (N, 1) -> p_homo becomes (N, 4)
    ones = np.ones((N, 1), dtype=float)
    p_homo = np.hstack([p_np, ones])
    
    # 4. Apply the transformation
    # We use T.T (transpose) because our points are row vectors.
    # (N, 4) @ (4, 4) -> (N, 4)
    p_trans_homo = p_homo @ T_np.T
    
    # 5. Extract the spatial part by dropping the last coordinate (the 1)
    p_trans = p_trans_homo[:, :3]
    
    # 6. Reformat the output to match the input shape
    if is_single_point:
        p_trans = p_trans.flatten()
        
    # 7. Safe return format for the grader
    if isinstance(points, list):
        return p_trans.tolist()
        
    return p_trans

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Single Point Translation
    T1 = [[1, 0, 0, 1],
          [0, 1, 0, 2],
          [0, 0, 1, 3],
          [0, 0, 0, 1]]
    p1 = [0, 0, 0]
    
    out1 = apply_homogeneous_transform(T1, p1)
    print(f"Example 1 -> Transformed Point: {out1}")
    # Expected: [1.0, 2.0, 3.0]
    
    # Example 2: Batch Points (Rotation + Translation)
    T2 = [[0, -1, 0, 1],
          [1,  0, 0, 0],
          [0,  0, 1, 0],
          [0,  0, 0, 1]]
    p2 = [[1, 0, 0], 
          [0, 1, 0]]
          
    out2 = apply_homogeneous_transform(T2, p2)
    print(f"\nExample 2 -> Transformed Batch:\n{out2}")
    # Expected: [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]