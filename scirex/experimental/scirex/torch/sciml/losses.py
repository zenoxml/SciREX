
import torch
import torch.nn as nn

class LpLoss(nn.Module):
    """
    Relative Lp loss, i.e., ||x - y||_p / ||y||_p
    """
    def __init__(self, d=1, p=2, size_average=True, reduction='mean'):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are parameters
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        # Calculate absolute difference norm (numerator)
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0) # (h,h)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction == 'mean':
            return torch.mean(all_norms)
        elif self.reduction == 'sum':
            return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        # Calculate relative difference norm
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)

        if self.reduction == 'mean':
            return torch.mean(diff_norms/y_norms)
        elif self.reduction == 'sum':
            return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def forward(self, pred, y, **kwargs):
        # x: input prediction
        # y: target
        return self.rel(pred, y)


class H1Loss(nn.Module):
    """
    H1 Sobolev norm loss 
    ||x - y||_H1 = ||x - y||_L2 + ||D(x - y)||_L2
    """
    def __init__(self, d=1, reduction='mean'):
        super(H1Loss, self).__init__()
        self.d = d
        self.reduction = reduction

    def forward(self, pred, y, **kwargs):
        # Assuming 2D grid for now based on FNO context
        # pred: (B, C, H, W)
        # y: (B, C, H, W)
        
        # L2 part
        diff = pred - y
        l2 = torch.norm(diff.view(diff.size(0), -1), 2, 1)
        y_l2 = torch.norm(y.view(y.size(0), -1), 2, 1)
        
        # H1 part (Gradient)
        # Simple finite difference approximation for derivative
        # Only implementing for d=2 case as standard in FNO examples
        
        if self.d == 2:
            # Gradients for diff
            diff_h = diff[:, :, 1:, :] - diff[:, :, :-1, :]
            diff_w = diff[:, :, :, 1:] - diff[:, :, :, :-1]
            
            # Gradients for y
            y_h = y[:, :, 1:, :] - y[:, :, :-1, :]
            y_w = y[:, :, :, 1:] - y[:, :, :, :-1]
            
            # Norms of gradients
            diff_h_norm = torch.norm(diff_h.view(diff_h.size(0), -1), 2, 1)
            diff_w_norm = torch.norm(diff_w.view(diff_w.size(0), -1), 2, 1)
            
            y_h_norm = torch.norm(y_h.view(y_h.size(0), -1), 2, 1)
            y_w_norm = torch.norm(y_w.view(y_w.size(0), -1), 2, 1)
            
            # Combine L2 and Gradient norms
            diff_norm = l2 + diff_h_norm + diff_w_norm
            y_norm = y_l2 + y_h_norm + y_w_norm
            
        else:
             # Fallback to L2 if not 2D or implemented
             return torch.mean(l2/y_l2) if self.reduction == 'mean' else torch.sum(l2/y_l2)

        if self.reduction == 'mean':
            return torch.mean(diff_norm / y_norm)
        elif self.reduction == 'sum':
            return torch.sum(diff_norm / y_norm)
        return diff_norm / y_norm
