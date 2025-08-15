import torch
import torch.nn as nn

class BLUPPrototypeManager(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_domains = args.N_users
        self.num_classes = args.num_classes
        self.feature_dim = args.domain_feature_dim
        self.momentum = args.momentum
        self.eps =  1e-6
        
        # EMA statistics with bias correction tracking
        self.register_buffer('feature_sums', torch.zeros(self.num_domains, self.num_classes, self.feature_dim))
        self.register_buffer('feature_sq_sums', torch.zeros(self.num_domains, self.num_classes, self.feature_dim))
        self.register_buffer('sample_counts', torch.zeros(self.num_domains, self.num_classes))
        
        # Track EMA step count for bias correction
        self.register_buffer('ema_steps', torch.zeros(self.num_domains, self.num_classes))
        
        # Domain-specific prototypes
        self.register_buffer('domain_prototypes', torch.zeros(self.num_domains, self.num_classes, self.feature_dim))
        
        # Global prototypes (BLUP estimates)
        self.register_buffer('global_prototypes', torch.zeros(self.num_classes, self.feature_dim))
        
        # Reliability scores for analysis
        self.register_buffer('reliability_scores', torch.zeros(self.num_domains, self.num_classes))

    def update_domain_prototype(self, domain_idx, features, labels):
        """Update domain-specific prototype with bias-corrected EMA"""
        if features.size(0) == 0:
            return
            
        for class_idx in range(self.num_classes):
            class_mask = (labels == class_idx)
            if not class_mask.any():
                continue
                
            class_features = features[class_mask]
            n_samples = class_features.size(0)
            
            # Compute batch statistics
            feature_sum = class_features.sum(dim=0)
            feature_sq_sum = (class_features ** 2).sum(dim=0)
            
            # Update EMA statistics
            if self.sample_counts[domain_idx, class_idx] == 0:
                # Initialize
                self.feature_sums[domain_idx, class_idx] = feature_sum
                self.feature_sq_sums[domain_idx, class_idx] = feature_sq_sum
                self.sample_counts[domain_idx, class_idx] = n_samples
                self.ema_steps[domain_idx, class_idx] = 1
            else:
                # EMA update
                self.feature_sums[domain_idx, class_idx] = (
                    self.momentum * self.feature_sums[domain_idx, class_idx] + 
                    (1 - self.momentum) * feature_sum
                )
                self.feature_sq_sums[domain_idx, class_idx] = (
                    self.momentum * self.feature_sq_sums[domain_idx, class_idx] + 
                    (1 - self.momentum) * feature_sq_sum
                )
                self.sample_counts[domain_idx, class_idx] = (
                    self.momentum * self.sample_counts[domain_idx, class_idx] + 
                    (1 - self.momentum) * n_samples
                )
                self.ema_steps[domain_idx, class_idx] += 1
            
            # Bias-corrected mean for domain prototype
            step = self.ema_steps[domain_idx, class_idx]
            bias_correction = 1 - self.momentum ** step
            
            corrected_sum = self.feature_sums[domain_idx, class_idx] / bias_correction
            corrected_count = self.sample_counts[domain_idx, class_idx] / bias_correction
            
            self.domain_prototypes[domain_idx, class_idx] = corrected_sum / torch.clamp(corrected_count, min=1.0)

    def compute_variance_components(self, class_idx, test_domain_idx=None):
        """Compute unbiased within and between variance components for BLUP with EMA bias correction"""
        
        # Get domains to use (exclude test domain)
        available_domains = []
        domain_means = []
        within_vars = []
        domain_counts = []
        
        for domain_idx in range(self.num_domains):
            if domain_idx == test_domain_idx:
                continue
                
            count_raw = self.sample_counts[domain_idx, class_idx]
            if count_raw < 2:  # Need at least 2 samples for unbiased variance
                continue
                
            # Apply bias correction to EMA statistics
            step = self.ema_steps[domain_idx, class_idx]
            bias_correction = 1 - self.momentum ** step
            
            # Bias-corrected first and second moments
            m1_corrected = self.feature_sums[domain_idx, class_idx] / bias_correction
            m2_corrected = self.feature_sq_sums[domain_idx, class_idx] / bias_correction
            count_corrected = self.sample_counts[domain_idx, class_idx] / bias_correction
            
            # Effective sample size for variance correction
            # This accounts for the fact that EMA reduces effective sample size
            n_eff = (1 - self.momentum**step) / (1 - self.momentum)
            
            if n_eff < 2:
                continue
                
            # FIXED: Proper unbiased within-domain variance calculation
            mean_x = m1_corrected / count_corrected
            
            # First compute population variance from bias-corrected moments
            var_pop = (m2_corrected / count_corrected) - (mean_x ** 2)
            
            # Convert to unbiased sample variance accounting for EMA effective sample size
            within_var = var_pop * (n_eff / (n_eff - 1))
            within_var = torch.clamp(within_var, min=self.eps)
            
            available_domains.append(domain_idx)
            domain_means.append(mean_x)
            within_vars.append(within_var)
            domain_counts.append(count_corrected)
        
        if len(available_domains) < 2:
            # Not enough domains for between-variance estimation
            return torch.full((self.feature_dim,), self.eps), torch.full((self.feature_dim,), self.eps), []
        
        domain_means_tensor = torch.stack(domain_means)
        within_vars_tensor = torch.stack(within_vars)
        
        # FIXED: Proper unbiased between-domain variance calculation
        k = domain_means_tensor.shape[0]
        population_mean = domain_means_tensor.mean(dim=0)
        between_var = ((domain_means_tensor - population_mean) ** 2).sum(dim=0) / (k - 1)
        
        # Non-negative safeguard for PSD covariance
        between_var = torch.clamp(between_var, min=0.0) + self.eps
        
        # Average within-domain variance weighted by sample counts
        domain_counts_tensor = torch.stack(domain_counts)
        weights = domain_counts_tensor / domain_counts_tensor.sum()
        pooled_within_var = (within_vars_tensor * weights.unsqueeze(-1)).sum(dim=0)
        pooled_within_var = torch.clamp(pooled_within_var, min=self.eps)
        
        return pooled_within_var, between_var, available_domains

    def update_global_prototypes(self, test_domain_idx=None):
        """Update global prototypes using BLUP with properly corrected variance components"""
        
        updated_prototypes = self.global_prototypes.clone()
        all_blup_weights = torch.zeros(self.num_domains, self.num_classes)
        
        for class_idx in range(self.num_classes):
            within_var, between_var, available_domains = self.compute_variance_components(
                class_idx, test_domain_idx)
            
            if len(available_domains) == 0:
                continue
            
            # Collect domain prototypes and compute BLUP weights
            domain_prototypes_list = []
            blup_weights = []
            
            for domain_idx in available_domains:
                domain_prototype = self.domain_prototypes[domain_idx, class_idx]
                domain_prototypes_list.append(domain_prototype)
                
                # BLUP weight: inverse variance weighting
                # w_i ∝ 1 / (σ²_i + τ²)
                total_var = within_var + between_var  # Element-wise
                weight = 1.0 / total_var  # Element-wise
                weight_scalar = weight.mean()  # Collapse to scalar for this implementation
                
                blup_weights.append(weight_scalar)
                all_blup_weights[domain_idx, class_idx] = weight_scalar
            
            if len(domain_prototypes_list) > 0:
                domain_prototypes_tensor = torch.stack(domain_prototypes_list)
                blup_weights_tensor = torch.stack(blup_weights)
                
                # Normalize weights
                blup_weights_normalized = blup_weights_tensor / blup_weights_tensor.sum()
                
                # Compute BLUP estimate
                global_prototype = (domain_prototypes_tensor * blup_weights_normalized.unsqueeze(-1)).sum(dim=0)
                updated_prototypes[class_idx] = global_prototype
                
                # Store reliability scores (normalized weights)
                for i, domain_idx in enumerate(available_domains):
                    self.reliability_scores[domain_idx, class_idx] = blup_weights_normalized[i]
        
        self.global_prototypes = updated_prototypes 
        return updated_prototypes, all_blup_weights

    def get_reliability_scores(self):
        """Get reliability scores for analysis"""
        return self.reliability_scores.clone()

    def get_global_prototypes(self):
        """Get current global prototypes"""
        return self.global_prototypes.clone()

    def get_domain_prototypes(self):
        """Get current domain prototypes"""
        return self.domain_prototypes.clone()
    
'''
class BLUPPrototypeManager(nn.Module):
    """
    Implements Best Linear Unbiased Predictor (BLUP) for domain adaptation.
    
    Uses mixed-effects modeling to estimate global prototypes by:
    1. Treating each user's prototype as a noisy estimate of the true global prototype
    2. Computing BLUP weights based on within-user and between-user variance
    3. Forming global prototypes through principled statistical shrinkage
    
    No heuristic confidence weighting - purely based on feature variance patterns.
    """
    def __init__(self, args):
        super(BLUPPrototypeManager, self).__init__()
        
        num_classes = args.num_classes
        feature_dim = args.domain_feature_dim
        num_domains = args.N_users
        
        # Domain-specific prototype storage
        self.register_buffer('domain_prototypes', torch.zeros(num_domains, num_classes, feature_dim))
        
        # BLUP variance estimation buffers
        self.register_buffer('feature_sums', torch.zeros(num_domains, num_classes, feature_dim))
        self.register_buffer('feature_sq_sums', torch.zeros(num_domains, num_classes, feature_dim))
        self.register_buffer('sample_counts', torch.zeros(num_domains, num_classes))
        
        # Global prototypes (output)
        self.register_buffer('global_prototypes', torch.zeros(num_classes, feature_dim))
        
        # BLUP weights cache (computed less frequently)
        self.register_buffer('blup_weights', torch.ones(num_domains, num_classes))
        self.register_buffer('within_vars', torch.ones(num_domains, num_classes, feature_dim))
        self.register_buffer('between_vars', torch.ones(num_classes, feature_dim))
        
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.eps = 1e-6
        self.momentum = args.momentum

    def update_domain_prototype(self, domain_idx, features, labels, logits=None):
        """
        Accumulate feature statistics for BLUP computation.
        Called frequently during training to build up statistics.
        """
        for cls_idx in range(self.num_classes):
            mask = (labels == cls_idx)
            if not mask.any():
                continue
                
            class_features = features[mask]  # [n_samples, feature_dim]
            n_samples = class_features.shape[0]
            
            # Update running statistics for BLUP
            feature_sum = class_features.sum(dim=0)  # [feature_dim]
            feature_sq_sum = (class_features ** 2).sum(dim=0)  # [feature_dim]
            
            # Momentum-based update of statistics
            self.feature_sums[domain_idx, cls_idx] = (
                self.momentum * self.feature_sums[domain_idx, cls_idx] + 
                (1 - self.momentum) * feature_sum
            )
            self.feature_sq_sums[domain_idx, cls_idx] = (
                self.momentum * self.feature_sq_sums[domain_idx, cls_idx] + 
                (1 - self.momentum) * feature_sq_sum
            )
            self.sample_counts[domain_idx, cls_idx] = (
                self.momentum * self.sample_counts[domain_idx, cls_idx] + 
                (1 - self.momentum) * n_samples
            )
            
            # Update domain prototype (simple mean)
            domain_mean = feature_sum / n_samples
            self.domain_prototypes[domain_idx, cls_idx] = (
                self.momentum * self.domain_prototypes[domain_idx, cls_idx] + 
                (1 - self.momentum) * domain_mean
            )
    
    def compute_variance_components(self, test_domain_idx=None):
        """
        Compute BLUP variance components:
        - σ²ᵢ,c: within-user variance for user i, class c
        - τ²c: between-user variance for class c
        """
        # Domains to consider (exclude test domain)
        if test_domain_idx is not None:
            valid_domains = [i for i in range(self.num_domains) if i != test_domain_idx]
        else:
            valid_domains = list(range(self.num_domains))
        
        for cls_idx in range(self.num_classes):
            domain_means = []
            within_variances = []
            
            for domain_idx in valid_domains:
                count = self.sample_counts[domain_idx, cls_idx]
                if count < 2:  # Need at least 2 samples for variance
                    continue
                    
                # Compute within-domain variance: σ²ᵢ,c
                sum_x = self.feature_sums[domain_idx, cls_idx]
                sum_x2 = self.feature_sq_sums[domain_idx, cls_idx]
                mean_x = sum_x / count
                
                # Variance = E[X²] - E[X]²
                within_var = (sum_x2 / count) - (mean_x ** 2)
                within_var = torch.clamp(within_var, min=self.eps)  # Avoid negative variance
                
                self.within_vars[domain_idx, cls_idx] = within_var
                within_variances.append(within_var)
                domain_means.append(self.domain_prototypes[domain_idx, cls_idx])
            
            if len(domain_means) < 2:  # Need at least 2 domains for between-variance
                self.between_vars[cls_idx] = torch.ones_like(self.between_vars[cls_idx]) * self.eps
                continue
            
            # Compute between-domain variance: τ²c
            domain_means_tensor = torch.stack(domain_means)  # [n_domains, feature_dim]
            population_mean = domain_means_tensor.mean(dim=0)  # [feature_dim]
            between_var = ((domain_means_tensor - population_mean) ** 2).mean(dim=0)
            between_var = torch.clamp(between_var, min=self.eps)
            
            self.between_vars[cls_idx] = between_var
    
    def compute_blup_weights(self, test_domain_idx=None):
        """
        Compute BLUP weights: wᵢ,c ∝ 1/(σ²ᵢ,c + τ²c)
        """
        if test_domain_idx is not None:
            valid_domains = [i for i in range(self.num_domains) if i != test_domain_idx]
        else:
            valid_domains = list(range(self.num_domains))
        
        for cls_idx in range(self.num_classes):
            for domain_idx in valid_domains:
                count = self.sample_counts[domain_idx, cls_idx]
                if count < 1:
                    self.blup_weights[domain_idx, cls_idx] = 0.0
                    continue
                
                # BLUP weight formula: wᵢ,c = 1/(σ²ᵢ,c + τ²c)
                within_var = self.within_vars[domain_idx, cls_idx].mean()  # Average across features
                between_var = self.between_vars[cls_idx].mean()  # Average across features
                
                total_var = within_var + between_var
                weight = 1.0 / (total_var + self.eps)
                self.blup_weights[domain_idx, cls_idx] = weight
            
            # Normalize weights for class cls_idx
            class_weights = self.blup_weights[valid_domains, cls_idx]
            total_weight = class_weights.sum()
            if total_weight > self.eps:
                self.blup_weights[valid_domains, cls_idx] = class_weights / total_weight
    
    def update_global_prototypes(self, test_domain_idx=None):
        """
        Pure BLUP approach: Compute global prototypes using statistical shrinkage.
        Called less frequently (e.g., end of epoch).
        
        Returns:
            global_prototypes: Tensor of shape [num_classes, feature_dim]
            blup_weights: Tensor of shape [num_domains, num_classes] for analysis
        """
        # Step 1: Compute variance components
        self.compute_variance_components(test_domain_idx)
        
        # Step 2: Compute BLUP weights
        self.compute_blup_weights(test_domain_idx)
        
        # Step 3: Form global prototypes using BLUP weights
        if test_domain_idx is not None:
            valid_domains = [i for i in range(self.num_domains) if i != test_domain_idx]
        else:
            valid_domains = list(range(self.num_domains))
        
        for cls_idx in range(self.num_classes):
            weighted_prototype = torch.zeros_like(self.global_prototypes[cls_idx])
            total_weight = 0.0
            
            for domain_idx in valid_domains:
                weight = self.blup_weights[domain_idx, cls_idx]
                if weight > self.eps:
                    weighted_prototype += weight * self.domain_prototypes[domain_idx, cls_idx]
                    total_weight += weight
            
            if total_weight > self.eps:
                self.global_prototypes[cls_idx] = weighted_prototype / total_weight
            else:
                # Fallback: average of domain prototypes
                valid_prototypes = [self.domain_prototypes[i, cls_idx] 
                                  for i in valid_domains 
                                  if self.sample_counts[i, cls_idx] > 0]
                if valid_prototypes:
                    self.global_prototypes[cls_idx] = torch.stack(valid_prototypes).mean(dim=0)
        
        return self.global_prototypes.clone(), self.blup_weights.clone()
    
    def get_reliability_scores(self):
        """
        Get reliability scores for each domain-class pair.
        Higher scores indicate more reliable estimates.
        """
        reliability = torch.zeros_like(self.blup_weights)
        
        for cls_idx in range(self.num_classes):
            class_weights = self.blup_weights[:, cls_idx]
            max_weight = class_weights.max()
            if max_weight > self.eps:
                reliability[:, cls_idx] = class_weights / max_weight
        
        return reliability



class BayesianPrototypeManager(nn.Module):
    """
    Implements Hierarchical Bayesian Prototype Modeling for domain adaptation.
    
    This approach models class prototypes with:
    1. A global prior across all domains (captures class essence)
    2. Domain-specific adaptations (captures domain-specific variations)
    3. Uncertainty modeling through learned variances
    
    This allows knowledge sharing between domains while preserving domain-specific information.
    """
    def __init__(self, args):
        super(BayesianPrototypeManager, self).__init__()
        
        num_classes = args.num_classes
        feature_dim =  args.domain_feature_dim
        num_domains = args.N_users
        self.register_buffer('global_means', torch.zeros(num_classes, feature_dim))
        self.register_buffer('global_log_vars', torch.zeros(num_classes, feature_dim))
        
        # Domain-specific adaptations (offsets from global prototype)
        self.register_buffer('domain_offsets', torch.zeros(num_domains, num_classes, feature_dim))
        self.register_buffer('domain_log_vars', torch.zeros(num_domains, num_classes, feature_dim))
        # Counters for tracking updates
        self.register_buffer('counts', torch.zeros(num_domains, num_classes))
        
        self.num_domains = num_domains
        self.num_classes = num_classes
        
        self.feature_dim = args.domain_feature_dim
        self.prior_strength = args.prior_strength  # Controls influence of the prior
        self.momentum = args.momentum
        self.eps = 1e-6
        
    def get_domain_prototype(self, domain_idx, class_idx):
        """Get the domain-specific prototype for a class"""
        return self.global_means[class_idx] + self.domain_offsets[domain_idx, class_idx]
    
    def get_domain_variance(self, domain_idx, class_idx):
        """Get the domain-specific variance for a class"""
        global_var = torch.exp(self.global_log_vars[class_idx])
        domain_var = torch.exp(self.domain_log_vars[domain_idx, class_idx])
        return global_var * domain_var  # Combine global and domain variance
    
    def sample_prototype(self, domain_idx, class_idx):
        """Sample a prototype from the distribution (useful during training)"""
        mean = self.get_domain_prototype(domain_idx, class_idx)
        std = torch.sqrt(self.get_domain_variance(domain_idx, class_idx))
        return mean + torch.randn_like(mean) * std
    
    def calculate_confidence(self, features, logits):
        """Calculate confidence scores for weighting feature contributions"""
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        
        # Entropy as uncertainty measure
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1)
        entropy_normalized = entropy / torch.log(torch.tensor(probs.shape[1], dtype=torch.float32, device=features.device))
        
        # Confidence = high probability and low entropy
        confidence = max_probs * (1 - entropy_normalized)
        return confidence
    
    def update_domain_prototype(self, domain_idx, features, labels, logits):
        """Update domain-specific prototype with new observations"""
        confidence = self.calculate_confidence(features, logits)
        
        
        for cls_idx in range(self.num_classes):
            mask = (labels == cls_idx)
            if not mask.any():
                continue
                
            class_features = features[mask]
            class_conf = confidence[mask].detach()
            
            # Weighted mean of features
            norm_conf = class_conf / (class_conf.sum() + self.eps)
            weighted_mean = torch.sum(class_features * norm_conf.unsqueeze(1), dim=0)
            
            # Weighted variance calculation
            delta = class_features - weighted_mean.unsqueeze(0)
            weighted_var = torch.sum(delta * delta * norm_conf.unsqueeze(1), dim=0) / (norm_conf.sum() + self.eps)
            
            # Current count for Bayesian update
            count = self.counts[domain_idx, cls_idx].item()
            
            if count > 0:
                # Bayesian update - domain offset
                prior_mean = self.global_means[cls_idx].detach()
                domain_offset = weighted_mean - prior_mean
                
                # Update with momentum to smooth changes
                current_offset = self.domain_offsets[domain_idx, cls_idx].detach()
                updated_offset = self.momentum * current_offset + (1 - self.momentum) * domain_offset
                self.domain_offsets[domain_idx, cls_idx] = updated_offset.detach()
                
                # Update log variance (uncertainty)
                current_log_var = self.domain_log_vars[domain_idx, cls_idx].detach()
                log_var = torch.log(weighted_var + self.eps)
                updated_log_var = self.momentum * current_log_var + (1 - self.momentum) * log_var
                self.domain_log_vars[domain_idx, cls_idx] = updated_log_var.detach()
            else:
                # First observation for this domain-class pair
                self.domain_offsets[domain_idx, cls_idx] = (weighted_mean - self.global_means[cls_idx]).detach()
                self.domain_log_vars[domain_idx, cls_idx] = torch.log(weighted_var + self.eps).detach()
            
            self.counts[domain_idx, cls_idx] += 1
    
    def update_global_prototypes(self, classifier_weights, test_domain_idx=None):
        """Update global prototypes by combining information across domains"""
        # Domains to consider (exclude test domain)
        if test_domain_idx is not None:
            domains = [i for i in range(self.num_domains) if i != test_domain_idx]
        else:
            domains = list(range(self.num_domains))
        
        for cls_idx in range(self.num_classes):
            # Collect domain prototypes and their precision (inverse variance)
            valid_means = []
            valid_precisions = []
            
            for domain_idx in domains:
                if self.counts[domain_idx, cls_idx] > 0:
                    # Get domain prototype for this class
                    domain_mean = self.get_domain_prototype(domain_idx, cls_idx).detach()
                    domain_var = self.get_domain_variance(domain_idx, cls_idx).detach()
                    domain_precision = 1.0 / (domain_var + self.eps)
                    
                    valid_means.append(domain_mean)
                    valid_precisions.append(domain_precision)
            
            if not valid_means:  # No valid domains for this class
                # Use classifier weights as fallback
                self.global_means[cls_idx] = classifier_weights[cls_idx].detach()
                continue
            
            # Stack domain information
            stacked_means = torch.stack(valid_means)
            stacked_precisions = torch.stack(valid_precisions)
            
            # Bayesian fusion of Gaussians
            # For each feature dimension: combine domain estimates weighting by precision
            total_precision = torch.sum(stacked_precisions, dim=0)  # [feature_dim]
            
            # Correct shape handling for weighted mean calculation
            # First compute precision-weighted sum properly
            weighted_sum = torch.zeros_like(self.global_means[cls_idx])  # [feature_dim]
            for i in range(len(stacked_means)):
                # Weight each domain's mean by its precision
                domain_precision = stacked_precisions[i]  # [feature_dim]
                domain_mean = stacked_means[i]  # [feature_dim]
                weighted_sum += domain_mean * domain_precision
            
            # Divide by total precision to get weighted mean
            weighted_mean = weighted_sum / (total_precision + self.eps)  # [feature_dim]
            
            # Posterior variance is inverse of summed precisions
            posterior_var = 1.0 / (total_precision + self.eps)  # [feature_dim]
            
            # Calculate scalar certainty based on average variance
            certainty = torch.exp(-torch.mean(posterior_var))  # scalar
            
            # Mix with classifier weights - shape is now correct
            global_mean = certainty * weighted_mean + (1 - certainty) * classifier_weights[cls_idx].detach()  # [feature_dim]
            
            # Update global means and log variances
            self.global_means[cls_idx] = global_mean.detach()
            self.global_log_vars[cls_idx] = torch.log(posterior_var + self.eps).detach()
        
        return self.global_means.clone(), torch.exp(self.global_log_vars).clone()
    
    
    def get_certainty(self):
        """Get certainty scores for each prototype (inversely related to variance)"""
        mean_vars = torch.exp(self.global_log_vars).mean(dim=1)
        certainty = torch.exp(-mean_vars)
        return certainty


'''