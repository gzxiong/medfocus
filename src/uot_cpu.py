import numpy as np
import ot

def unbalanced_optimal_transport(source, target, source_weights=None, target_weights=None, 
                                  reg=0.1, reg_m=1.0, method='sinkhorn'):
    """
    Compute unbalanced optimal transport between source and target distributions.
    
    Parameters:
    -----------
    source : array-like, shape (n_samples_source, n_features)
        Source samples
    target : array-like, shape (n_samples_target, n_features)
        Target samples
    source_weights : array-like, shape (n_samples_source,), optional
        Weights for source samples (default: uniform)
    target_weights : array-like, shape (n_samples_target,), optional
        Weights for target samples (default: uniform)
    reg : float, optional
        Entropy regularization parameter (default: 0.1)
    reg_m : float, optional
        Marginal relaxation parameter (default: 1.0)
    method : str, optional
        Method to use: 'sinkhorn' or 'lbfgsb' (default: 'sinkhorn')
    
    Returns:
    --------
    transport_plan : array-like, shape (n_samples_source, n_samples_target)
        Optimal transport plan
    """
    source = np.asarray(source)
    target = np.asarray(target)
    
    # Initialize uniform weights if not provided
    if source_weights is None:
        source_weights = np.ones(source.shape[0]) / source.shape[0]
    else:
        source_weights = np.asarray(source_weights)
        source_weights = source_weights / source_weights.sum()
    
    if target_weights is None:
        target_weights = np.ones(target.shape[0]) / target.shape[0]
    else:
        target_weights = np.asarray(target_weights)
        target_weights = target_weights / target_weights.sum()
    
    # Compute cost matrix (squared Euclidean distance)
    cost_matrix = ot.dist(source, target, metric='sqeuclidean')
    
    # Compute unbalanced OT
    if method == 'sinkhorn':
        transport_plan = ot.unbalanced.sinkhorn_unbalanced(
            source_weights, target_weights, cost_matrix, 
            reg=reg, reg_m=reg_m
        )
    elif method == 'lbfgsb':
        transport_plan = ot.unbalanced.lbfgsb_unbalanced(
            source_weights, target_weights, cost_matrix, 
            reg_m=reg_m
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sinkhorn' or 'lbfgsb'.")
    
    return transport_plan


def unbalanced_wasserstein_distance(source, target, source_weights=None, target_weights=None,
                                      reg=0.1, reg_m=1.0, method='sinkhorn'):
    """
    Compute unbalanced Wasserstein distance between source and target distributions.
    
    Parameters:
    -----------
    source : array-like, shape (n_samples_source, n_features)
        Source samples
    target : array-like, shape (n_samples_target, n_features)
        Target samples
    source_weights : array-like, shape (n_samples_source,), optional
        Weights for source samples
    target_weights : array-like, shape (n_samples_target,), optional
        Weights for target samples
    reg : float, optional
        Entropy regularization parameter
    reg_m : float, optional
        Marginal relaxation parameter
    method : str, optional
        Method to use: 'sinkhorn' or 'lbfgsb'
    
    Returns:
    --------
    distance : float
        Unbalanced Wasserstein distance
    """
    transport_plan = unbalanced_optimal_transport(
        source, target, source_weights, target_weights, reg, reg_m, method
    )
    
    source = np.asarray(source)
    target = np.asarray(target)
    cost_matrix = ot.dist(source, target, metric='sqeuclidean')
    
    distance = np.sum(transport_plan * cost_matrix)
    return distance