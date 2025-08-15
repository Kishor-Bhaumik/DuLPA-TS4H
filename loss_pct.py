from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ProtoLoss']


class Bidirectional_aligner(nn.Module):
    """
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self,args, num_classes: int, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(Bidirectional_aligner, self).__init__()
        self.nav_t = 1
        self.s_par = args.s_par
        self.beta = args.beta
        self.prop = (torch.ones((num_classes,1))*(1/num_classes))
        self.eps = 1e-6
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop+ self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Update proportions
        self.prop = self.prop.to(mu_s.device)

        sim_mat = torch.matmul(mu_s, f_t.T)
        old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
        s_dist_old = F.softmax(old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        self.prop = self.update_prop(prop)

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, self.prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        source_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        target_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*self.prop.squeeze(1)).sum()
        loss = source_loss + target_loss
        return loss




def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def mcc_loss(logits_t_weak, entropy_temperature, num_classes):
        train_bs= logits_t_weak.size(0)
        outputs_target_temp = logits_t_weak / entropy_temperature
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = Entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
            target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        unsup_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / num_classes

        return unsup_loss

def ce_loss(logits, targets, use_hard_labels=True, reduction='none', weight=None):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, weight=weight, reduction=reduction )
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def supervised_loss(N_users, test_subject, logits_s, y_s, domain_y_s, weights):

        losses_sup =[]
        for domain_idx in range(N_users):
            if domain_idx == test_subject: 
                continue
            domain_mask = domain_y_s == domain_idx
            if domain_mask.any():
                domain_labels = y_s[domain_mask]
                losses_sup.append((ce_loss(logits_s[domain_mask], domain_labels, reduction='none',\
                                           weight=weights[domain_idx, :]  )).mean())
        
        return torch.stack(losses_sup).mean()
