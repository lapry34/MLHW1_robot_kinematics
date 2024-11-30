import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10, ignore_small_threshold=1e-2):
    """
    Calculate MAPE with logic to handle small true values.
    
    Parameters:
    - y_true: Array of true values
    - y_pred: Array of predicted values
    - epsilon: Small constant to avoid division by zero
    - ignore_small_threshold: Threshold below which values are ignored in MAPE calculation
    
    Returns:
    - MAPE: Mean Absolute Percentage Error as a percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Ignore cases where the true values are too small
    significant_mask = np.abs(y_true) > ignore_small_threshold
    
    if not significant_mask.any():
        return 0.0  # Return 0 if all values are insignificant
    
    # Compute the Absolute Percentage Error for significant values
    ape = np.abs((y_true[significant_mask] - y_pred[significant_mask]) /
                 np.maximum(np.abs(y_true[significant_mask]), epsilon))
    
    # Mean APE
    mape = np.mean(ape) * 100
    return mape