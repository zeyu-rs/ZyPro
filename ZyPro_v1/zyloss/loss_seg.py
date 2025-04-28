import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=1 - 0.5, beta=0.5, epsilon=0.000001):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)

        # TP
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        # FN
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        # FP
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)

        return 1 - ((true_pos + self.epsilon) /
                    (true_pos + self.alpha * false_neg + self.beta * false_pos + self.epsilon))


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-6, bce_weight=0.5):
        super(ComboLoss, self).__init__()
        self.tversky = TverskyLoss(alpha, beta, epsilon)
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, y_pred, y_true):
        # 如果 y_pred 是 logits，记得加 sigmoid
        # y_pred = torch.sigmoid(y_pred)

        tversky_loss = self.tversky(y_pred, y_true)
        bce_loss = self.bce(y_pred, y_true)

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * tversky_loss

