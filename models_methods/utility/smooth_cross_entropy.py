import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, ground_truth, smoothing=0.1):
    """"
    Compute the cross-entropy loss with label smoothing. Useful if the ground truth is not entirely reliable or if 
    overfitting is a concern.
    
    pred: Predicted probabilities from the model
    ground_truth: True labels
    smoothing: Smoothing factor
    log_prob: Log probabilities
    """
    n_class = pred.size(1)

    # create a tensor with the same shape as pred filled with the value of smoothing divided by n_class - 1
    one_hot = torch.full_like(pred, 
                              fill_value= smoothing / (n_class - 1))
    # sets the probability of the correct class to 1.0 - smoothing 
    one_hot.scatter_(dim=1, 
                     index=ground_truth.unsqueeze(1), 
                     value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
