def ordinal_weighted_ce(preds, dtrain):
    labels = dtrain.get_label().astype(int)
    n, K = preds.shape[0], preds.shape[1]
    
    # Softmax
    preds_exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    probs = preds_exp / np.sum(preds_exp, axis=1, keepdims=True)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
    
    # Base categorical grad/hess
    target_onehot = np.eye(K)[labels]  # (n, K)
    base_grad = probs - target_onehot
    base_hess = probs * (1 - probs)
    
    # Ordinal weights: distance from true label
    pred_classes = np.arange(K)
    dists = np.abs(labels[:, None] - pred_classes[None, :])  # (n, K)
    weights = 1.0 + dists  # e.g., linear penalty, min 1.0
    
    # Weighted
    grad = base_grad * weights
    hess = base_hess * weights  # Hessian approx; conservative
    
    return grad.flatten(), hess.flatten()
