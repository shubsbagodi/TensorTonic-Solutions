import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Performs one step of the Value Iteration algorithm (Bellman optimality backup).
    
    Args:
        values (list): Current value estimates for each state, V(s).
        transitions (list): 3D array of transition probabilities, T(s, a, s').
        rewards (list): 2D array of immediate rewards, R(s, a).
        gamma (float): Discount factor.
        
    Returns:
        list: Updated value estimates for each state, V'(s).
    """
    # 1. Convert inputs to NumPy arrays for fast, vectorized operations
    V = np.array(values, dtype=float)
    T = np.array(transitions, dtype=float)
    R = np.array(rewards, dtype=float)
    
    # 2. Compute Expected Future Values
    # T is shape (States, Actions, States), V is shape (States,)
    # T @ V performs the dot product over the last axes, resulting in shape (States, Actions)
    # This represents the sum over s' of T(s, a, s') * V(s')
    expected_future_values = T @ V
    
    # 3. Compute Q-values for all states and actions
    # Q(s, a) = R(s, a) + gamma * expected_future_value
    Q = R + gamma * expected_future_values
    
    # 4. Take the maximum Q-value across actions for each state
    # This results in an array of shape (States,)
    V_new = np.max(Q, axis=1)
    
    # 5. Return strictly as a standard Python list of floats
    return V_new.tolist()

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1
    v1 = [0, 0]
    t1 = [[[0.8, 0.2], [0.3, 0.7]], 
          [[0.5, 0.5], [0.1, 0.9]]]
    r1 = [[1, 2], 
          [-1, 0]]
    g1 = 0.9
    
    print(f"Example 1 Output: {value_iteration_step(v1, t1, r1, g1)}")
    # Expected: [2.0, 0.0]
    
    # Example 2
    v2 = [0, 0, 0]
    t2 = [[[0, 1, 0], [0, 0, 1]], 
          [[1, 0, 0], [0, 0, 1]], 
          [[0, 1, 0], [1, 0, 0]]]
    r2 = [[1, 2], 
          [3, 0], 
          [-1, 5]]
    g2 = 0.9
    
    print(f"Example 2 Output: {value_iteration_step(v2, t2, r2, g2)}")
    # Expected: [2.0, 3.0, 5.0]