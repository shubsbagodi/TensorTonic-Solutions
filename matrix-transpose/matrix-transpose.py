import numpy as np

def matrix_transpose(A):
    """
    Transposes a 2D matrix manually by swapping rows to columns.
    """
    A = np.array(A)
    N, M = A.shape
    
    # Initialize the transposed matrix with zeros
    transposed = np.zeros((M, N), dtype=A.dtype)
    
    # Manual element-by-element indexing: (A^T)[j, i] = A[i, j]
    for i in range(N):
        for j in range(M):
            transposed[j, i] = A[i, j]
            
    return transposed