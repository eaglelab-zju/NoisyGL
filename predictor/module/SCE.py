import torch
import torch.nn.functional as F

def symmetric_cross_entropy(y_true, y_pred, alpha, beta):
    y_pred_clamped = torch.clamp(y_pred, min=1e-4, max=1.0)
    y_true_log_prob = torch.gather(y_pred_clamped, 1, y_true.view(-1, 1)).view(-1)
    y_true_log_prob = torch.clamp(y_true_log_prob, min=1e-4, max=1.0)

    loss_1 = F.cross_entropy(y_pred, y_true)
    loss_2 = -torch.log(y_true_log_prob)

    loss = alpha * loss_1 + beta * loss_2
    return loss.mean()
