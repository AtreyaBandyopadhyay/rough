def ordinal_weighted_ce(preds, dtrain, order_penalty=True):
    labels = dtrain.get_label().astype(int)
    n = len(labels)
    num_class = preds.shape[1]
    preds_exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    probs = preds_exp / np.sum(preds_exp, axis=1, keepdims=True)
    probs = np.clip(probs, 1e-8, 1 - 1e-8)
    
    target_onehot = np.zeros((n, num_class))
    target_onehot[np.arange(n), labels] = 1
    
    if order_penalty:
        # Weight by distance matrix (e.g., absolute distance)
        dist_matrix = np.abs(np.arange(num_class)[:, None] - np.arange(num_class))
        weighted_probs = probs * dist_matrix  # Or on loss
        loss_weights = np.sum(weighted_probs * target_onehot, axis=1)[:, None]  # Per sample weight
        grad = (probs - target_onehot) * loss_weights
        hess = probs * (1 - probs) * loss_weights[:, None]  # Approximate
    else:
        grad = probs - target_onehot
        hess = probs * (1 - probs)
    
    return grad.flatten(), hess.flatten()


