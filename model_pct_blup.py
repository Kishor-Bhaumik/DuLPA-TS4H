#new unsup loss is
#epoch lelvel prototuype update
import numpy as np
import torch
import torch.optim as optim
from torchmetrics import F1Score, AUROC
import pytorch_lightning as pl
from utils import  conv_optimize,update_confusion_matrix,direct_alignment,StepwiseLR
from prototype_learner import  BLUPPrototypeManager
from nets import ResNetFeatureExtractor, FC_layer, Domain_classifier_DG
from loss_pct import Bidirectional_aligner,  mcc_loss, supervised_loss
from pytorch_lightning.utilities import CombinedLoader

class MODEL(pl.LightningModule):
    def __init__(self, args): 
        super(MODEL, self).__init__()
        self.save_hyperparameters()
        self.args = args
        num_classes = args.num_classes  
        self.feature_extractor =ResNetFeatureExtractor(args.num_channel) 
        
        self.bp_loss = Bidirectional_aligner(args,num_classes)
        self.global_prototype_manager = BLUPPrototypeManager(args)
        
        self.activity_classifier = FC_layer(args) 
        self.register_buffer('alldomain_prototypes', torch.ones(num_classes, args.domain_feature_dim))
        self.num_classes = num_classes
        self.f1_macro = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.auroc_macro = AUROC(num_classes=num_classes, average='macro', task='multiclass')

        self.epoch_confuse_matrix   = np.zeros((num_classes, num_classes, args.N_users))
        self.epoch_tgt_pseudo_label = np.zeros(num_classes)
        self.register_buffer('epoch_src_true', torch.zeros(args.N_users, num_classes))
        self.register_buffer('sample_weight', torch.ones([args.N_users, num_classes]))
        self.register_buffer('src_centroid', torch.zeros([args.N_users, num_classes, args.domain_feature_dim]))
        self.register_buffer('tgt_centroid', torch.zeros([num_classes, args.domain_feature_dim]))

        self.criterion = torch.nn.CrossEntropyLoss()
        self.domain_classifier = Domain_classifier_DG(args.N_users, args.domain_feature_dim, dropout_rate=args.dropout_rate)

        self.training_step_outputs = []
        
        # FIX 1: Initialize global_batch_counter
        self.global_batch_counter = 0

        self.register_buffer('epoch_features', None)  # Will be initialized in on_train_epoch_start
        self.register_buffer('epoch_labels', None)    # Will be initialized in on_train_epoch_start
        self.register_buffer('epoch_domains', None)   # Will be initialized in on_train_epoch_start
        self.register_buffer('epoch_logits', None)    # Will be initialized in on_train_epoch_start
            
        self.beta_scheduler = StepwiseLR(
            optimizer=None,  # No optimizer, just for beta scheduling
            init_lr=args.beta, 
            gamma=args.scheduler_gamma, 
            decay_rate=args.decay
        )

    def on_train_epoch_start(self):
        # Reset accumulators at start of epoch
        self.epoch_confuse_matrix[:] = 0
        self.epoch_tgt_pseudo_label[:]= 0
        self.epoch_src_true.zero_()
        
        # FIX 2: Reset global batch counter at epoch start
        self.global_batch_counter = 0

        train_dataloader = self.trainer.train_dataloader
        source_dataloader = train_dataloader["source"]
        source_dataset = source_dataloader.dataset

        for domain_idx in range(self.args.N_users):
            if domain_idx == self.args.test_subject:
                continue
            domain_labels = torch.tensor([y for x, y, d in source_dataset if d == domain_idx])
            
            for cls_idx in range(self.num_classes):
                self.epoch_src_true[domain_idx, cls_idx] = (domain_labels == cls_idx).sum()
            # Normalize immediately
            self.epoch_src_true[domain_idx] = self.epoch_src_true[domain_idx] / len(domain_labels)

        self.epoch_features = None
        self.epoch_labels = None
        self.epoch_domains = None
        self.epoch_logits = None

    def on_train_epoch_end(self):
        self.sample_weight = conv_optimize(
            self.args, 
            self.epoch_confuse_matrix,
            self.epoch_src_true,
            self.epoch_tgt_pseudo_label,
            self.sample_weight)

        # FIX 3: Use the helper function consistently
        if self.args.use_proto and self.epoch_features is not None:
            if self._should_update_global_prototypes(is_epoch_end=True):
                # Update domain-specific prototypes with accumulated epoch data
                for domain_idx in range(self.args.N_users):
                    if domain_idx == self.args.test_subject:
                        continue
                        
                    domain_mask = self.epoch_domains == domain_idx
                    if domain_mask.any():
                        domain_features = self.epoch_features[domain_mask]
                        domain_labels = self.epoch_labels[domain_mask]
                        
                        # Update domain-specific prototypes with accumulated epoch data
                        self.global_prototype_manager.update_domain_prototype(
                            domain_idx, domain_features, domain_labels)
                
                # Update global prototypes using pure BLUP (no classifier weights needed)
                prototypes_updated, blup_weights = self.global_prototype_manager.update_global_prototypes(
                    test_domain_idx=self.args.test_subject)

                self.alldomain_prototypes = prototypes_updated
        
        # Clear accumulated outputs and data
        self.training_step_outputs.clear()

    def forward(self, x_source, x_target):
        # Extract features
        x_feat_all = self.feature_extractor(torch.cat((x_source, x_target), dim=0))
        logits_all = self.activity_classifier(x_feat_all)
        logits_source, logits_target = torch.split(logits_all, [x_source.shape[0], x_target.shape[0]], dim=0)
        domain_logits = self.domain_classifier(x_feat_all, constant=1, Reverse=True)

        source_feat, target_feat = torch.split(x_feat_all, [x_source.shape[0], x_target.shape[0]], dim=0)

        return logits_source, logits_target, source_feat,target_feat, domain_logits
              
    def ramp_up(self):
        p = float(self.current_epoch) / self.trainer.max_epochs
        constant = 2. / (1. + np.exp(-10 * p)) - 1  
        return constant

    def _should_update_global_prototypes(self, is_epoch_end=False):
        """Helper function to determine if global prototypes should be updated"""
        if is_epoch_end:
            return True
        
        # FIX 4: Add safety check and handle case when proto_update_freq is None
        if self.args.proto_update_freq is not None:
            return self.global_batch_counter % self.args.proto_update_freq == 0
        
        # If no batch-level update frequency is specified, don't update during training steps
        return False
    
    def training_step(self, batch, batch_idx):
        x_source, y_source, domain_y_source = batch["source"]
        x_target, _, domain_y_target = batch["target"]

        # Increment global batch counter
        self.global_batch_counter += 1

        logits_source, logits_target,source_feat,target_feat, domain_logits= self.forward(x_source, x_target )
        unsup_loss= mcc_loss(logits_target, self.args.entropy_temperature, self.num_classes)
        sup_loss = supervised_loss(self.args.N_users, self.args.test_subject, logits_source, y_source, domain_y_source, self.sample_weight)

        confuse_matrix,batch_tgt_pseudo_label = update_confusion_matrix(self.args, logits_target, logits_source, y_source, domain_y_source,\
            self.num_classes, self.epoch_confuse_matrix)
        self.epoch_confuse_matrix = confuse_matrix
        self.epoch_tgt_pseudo_label += batch_tgt_pseudo_label

        tgt_pred = torch.argmax(logits_target, dim=1)
        L_direct = []
        
        for domain_idx in range(self.args.N_users):
            if domain_idx == self.args.test_subject:
                continue
            tgt_label_estimated_distri = self.epoch_src_true[domain_idx, :] * self.sample_weight[domain_idx, :]  

            direct_loss, src_centroid, tgt_centroid = direct_alignment(domain_idx=domain_idx, src_feature=source_feat[domain_y_source == domain_idx], \
                tgt_feature=target_feat,src_truth_label=y_source[domain_y_source == domain_idx],tgt_pseudo_label=tgt_pred , \
                            tgt_y_estimated=tgt_label_estimated_distri, num_classes=self.num_classes, args=self.args,\
                                src_centroid=self.src_centroid, tgt_centroid=self.tgt_centroid )
            
            self.src_centroid = src_centroid
            self.tgt_centroid = tgt_centroid
            L_direct.append(direct_loss)

        L_direct_ = torch.stack(L_direct).mean()
        all_domain_labels = torch.cat((domain_y_source, domain_y_target), dim=0)
        domain_loss = self.criterion( domain_logits , all_domain_labels)

        pred_s = torch.argmax(logits_source, dim=1)
        f1_train = self.f1_macro(pred_s, y_source)

        # Accumulate features for epoch-level processing
        if self.epoch_features is None:
            self.epoch_features = source_feat.detach().clone()
            self.epoch_labels = y_source.detach().clone()
            self.epoch_domains = domain_y_source.detach().clone()
            self.epoch_logits = logits_source.detach().clone()
        else:
            self.epoch_features = torch.cat([self.epoch_features, source_feat.detach()], dim=0)
            self.epoch_labels = torch.cat([self.epoch_labels, y_source.detach()], dim=0)
            self.epoch_domains = torch.cat([self.epoch_domains, domain_y_source.detach()], dim=0)
            self.epoch_logits = torch.cat([self.epoch_logits, logits_source.detach()], dim=0)

        # Update domain prototypes with current batch (lightweight, responsive updates)
        if self.args.use_proto:
            for domain_idx in range(self.args.N_users):
                if domain_idx == self.args.test_subject:
                    continue
                domain_mask = domain_y_source == domain_idx
                if domain_mask.any():
                    self.global_prototype_manager.update_domain_prototype(
                        domain_idx, source_feat[domain_mask].detach(), y_source[domain_mask].detach())

            # Check if we should update global prototypes based on batch frequency
            if self._should_update_global_prototypes(is_epoch_end=False):
                prototypes_updated, _ = self.global_prototype_manager.update_global_prototypes(
                    test_domain_idx=self.args.test_subject)
                self.alldomain_prototypes = prototypes_updated

        if self.current_epoch == 0 and batch_idx == 0:
            prototypes_updated = self.activity_classifier.get_weights().clone().detach()
        else:
            prototypes_updated = self.alldomain_prototypes

        self.beta_scheduler.step()
        self.bp_loss.beta = self.beta_scheduler.get_lr()
        bidirectional_loss = self.bp_loss(prototypes_updated, target_feat) 

        unsup_weight_loss = self.args.unsup_loss_weight
        supervised_weight_loss = self.args.supervised_loss_weight
        
        ramp_const = self.ramp_up()
        direct_loss_weight = ramp_const 
        domain_loss_weight = ramp_const 
        bidirect_loss_weight = ramp_const 
        
        total_loss = supervised_weight_loss*sup_loss + unsup_weight_loss*unsup_loss +  direct_loss_weight*L_direct_ +\
            domain_loss_weight*domain_loss + bidirect_loss_weight*bidirectional_loss

        self.log("train_f1", f1_train,on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": total_loss, "f1": f1_train}

    def train_dataloader(self):
        if hasattr(self, 'train_loader_dict'):
            combined_loader = CombinedLoader(self.train_loader_dict, mode="max_size_cycle")
            return combined_loader
        return None
        
    def test_dataloader(self):
        if hasattr(self, 'test_loader'):
            return self.test_loader
        return None
    
    def test_step(self, batch, batch_idx):
        x_t, y, _ = batch
        x_t = self.feature_extractor(x_t)
        logits_t= self.activity_classifier(x_t)
        preds = torch.argmax(logits_t, dim=1)
        f1_val = self.f1_macro(preds, y)
        auroc_val = self.auroc_macro(logits_t, y)
    
        self.log("test_f1_macro", f1_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auroc_macro", auroc_val, on_step=False, on_epoch=True, prog_bar=True)
        return {"f1": f1_val, "auroc": auroc_val}

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr,  weight_decay=0.01)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step_size, gamma=self.args.scheduler_gamma)
        return [optimizer], [scheduler]

