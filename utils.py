
import torch
from sklearn.metrics import confusion_matrix
import cvxpy as cp
import numpy as np

from typing import Optional
from torch.optim.optimizer import Optimizer

class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        text{lr} = text{init_lr} times text{lr_mult} times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()

        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                if 'lr_mult' not in param_group:
                    param_group['lr_mult'] = 1.
                param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


    

def direct_alignment( domain_idx, src_feature, tgt_feature, src_truth_label, tgt_pseudo_label, tgt_y_estimated, \
                   num_classes, args, src_centroid, tgt_centroid):
    """
    Updates the centroids for source and target domains and calculates semantic loss
    Args:
        domain_idx: index of the source domain
        src_feature: features from source domain
        tgt_feature: features from target domain
        src_truth_label: true labels from source domain
        tgt_pseudo_label: pseudo labels from target domain
        tgt_y_estimated: estimated target label distribution
    """
    
    n, d = src_feature.shape
    s_labels, t_labels = src_truth_label, tgt_pseudo_label.long()
    ones = torch.ones_like(s_labels, dtype=torch.float32)
    zeros = torch.zeros(num_classes).to(src_feature.device)
    s_n_classes = zeros.scatter_add(0, s_labels, ones)
    t_n_classes = zeros.scatter_add(0, t_labels, ones)
    zeros = torch.zeros(num_classes, d).to(src_feature.device)
    s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), src_feature)
    t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), tgt_feature)
    cls_ones = torch.ones_like(s_n_classes.view(num_classes, 1))
    current_s_centroid = torch.div(s_sum_feature, torch.max(s_n_classes.view(num_classes, 1), cls_ones))
    current_t_centroid = torch.div(t_sum_feature, torch.max(t_n_classes.view(num_classes, 1), cls_ones))   
    new_src_centroid = (1-args.decay) * src_centroid[domain_idx, :, :] + args.decay * current_s_centroid
    new_tgt_centroid = (1-args.decay) * tgt_centroid + args.decay * current_t_centroid
    s_loss = torch.mean(torch.pow(new_src_centroid - new_tgt_centroid, 2), dim=1)
    direct_loss = torch.sum(torch.mul(tgt_y_estimated, s_loss)) 
    src_centroid[domain_idx, :, :] = new_src_centroid.detach()

    return direct_loss,  src_centroid, new_tgt_centroid.detach()



def BBSL(C,y_t,y_s):

    """
    C confusion matrix (C defined in the sckit learn should be transpose)
    y_t predicted tar label distribution
    y_s ground truth src label distribution

    """

    n = len(y_s)

    # Define and solve the CVXPY problem.
    alpha = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(alpha @ C - y_t)),
                      [alpha @ y_s == 1,
                       alpha >= 0])
    prob.solve(solver="SCS")
    # prob.solve(solver="OSQP")
    # prob.solve()

    alpha_opt = alpha.value

    return alpha_opt




def conv_optimize(args, my_confuse_matrix, src_true, tgt_pseudo_label, sample_weight,num_classes=5):
    momentum = 0.9
    src_true = src_true.cpu()
    tgt_pseudo_label = tgt_pseudo_label / np.sum(tgt_pseudo_label)
    new_sample_weight = sample_weight.clone()
    for domain_idx in range(args.N_users):
        if domain_idx == args.test_subject: 
            continue
        Con_s = my_confuse_matrix[:, :, domain_idx]
        Con_s = Con_s /np.sum(Con_s)  #(np.sum(Con_s) + 1e-10)  # Add small epsilon to prevent division by zero
        src_true_s = src_true[domain_idx, :]
        
        result = BBSL(C=Con_s, y_t=tgt_pseudo_label, y_s=src_true_s)
        #result = NLLSL(C=Con_s, y_t=tgt_pseudo_label, y_s=src_true_s)
        
        result = np.clip(result, 0.1, 10.0)
        result = result * (num_classes / result.sum())

        # Gradual update using momentum
        current_weight = sample_weight[domain_idx, :].cpu().numpy()
        updated_weight = momentum * current_weight + (1 - momentum) * result
        
        new_sample_weight[domain_idx, :] = torch.tensor(
            updated_weight, requires_grad=False).to(src_true.device)

    return new_sample_weight





def update_confusion_matrix( args, logits_t_weak, logits_s, y_s, domain_y_s, num_classes, epoch_confuse_matrix):

        for domain_idx in range(args.N_users):
            if domain_idx == args.test_subject: 
                continue
            domain_mask = domain_y_s == domain_idx
            if domain_mask.any():
                domain_preds = torch.argmax(logits_s[domain_mask], dim=1).cpu().numpy()
                domain_labels = y_s[domain_mask].cpu().numpy()
                epoch_confuse_matrix[:, :, domain_idx] += confusion_matrix(
                    domain_labels, domain_preds,
                    labels=list(range(num_classes)))
        
        tgt_pred = torch.argmax(logits_t_weak, dim=1).cpu().numpy()
        batch_tgt_pseudo_label = np.zeros([num_classes])
        for cls_idx in range(num_classes):
            batch_tgt_pseudo_label[cls_idx] = np.count_nonzero(
                tgt_pred == cls_idx)
        
        return epoch_confuse_matrix, batch_tgt_pseudo_label



