import torch
from tqdm import tqdm
import os
import sys
from typing import Dict, Tuple
from core.WILD.utils_wild import (
    visualize_reconstructions,
    select_diverse_sample_batch
)
from core.utils import process_batch, get_model_name, _calculate_metrics
from core.WILD.trainer import WILDTrainer

class StagedWILDTrainer(WILDTrainer):
    """
    Staged trainer for NVAE with capacity annealing to ensure proper disentanglement.
    
    Stage 1: Train za, zy first with zay capacity = 0 (force them to encode their respective factors)
    Stage 2: Gradually introduce zay capacity with strong KL + independence penalty
    Stage 3: Full training with all latent variables active
    """
    
    def __init__(self, model, optimizer, device, args, patience=5, scheduler=None):
        super().__init__(model, optimizer, device, args, patience, scheduler=scheduler)
        
        # Staged training parameters
        self.stage1_epochs = args.stage1_epochs if hasattr(args, 'stage1_epochs') and args.stage1_epochs is not None else max(10, args.epochs // 3)
        self.stage2_epochs = args.stage2_epochs if hasattr(args, 'stage2_epochs') and args.stage2_epochs is not None else max(10, args.epochs // 3)
        self.stage3_epochs = args.epochs - self.stage1_epochs - self.stage2_epochs

        # Validate stage epoch configuration
        if self.stage1_epochs < 0:
            raise ValueError(f"stage1_epochs cannot be negative: {self.stage1_epochs}")
        if self.stage2_epochs < 0:
            raise ValueError(f"stage2_epochs cannot be negative: {self.stage2_epochs}")
        if self.stage3_epochs < 0:
            raise ValueError(
                f"Invalid stage configuration: Stage epochs exceed total epochs!\n"
                f"  stage1_epochs ({self.stage1_epochs}) + stage2_epochs ({self.stage2_epochs}) = "
                f"{self.stage1_epochs + self.stage2_epochs}\n"
                f"  but total epochs = {args.epochs}\n"
                f"  → stage3_epochs would be {self.stage3_epochs} (negative!)\n\n"
                f"Please either:\n"
                f"  1. Reduce --stage1_epochs or --stage2_epochs, OR\n"
                f"  2. Increase total --epochs to at least {self.stage1_epochs + self.stage2_epochs}"
            )

        # Warning for edge case where stage3 has zero epochs
        if self.stage3_epochs == 0:
            print(f"⚠️  WARNING: Stage 3 has 0 epochs. Training will only use Stages 1 and 2.")
        
        # Learning rate scheduling for stability
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.warmup_epochs = 3  # Warm up for first 3 epochs
        
        # Capacity annealing parameters (toggleable)
        self.use_zay_annealing = getattr(args, 'use_zay_annealing', True)  # Default True for backward compatibility
        if self.use_zay_annealing:
            self.zay_capacity_start = 0.0
            self.zay_capacity_end = 1.0
        else:
            # No annealing: constant full capacity
            self.zay_capacity_start = 1.0
            self.zay_capacity_end = 1.0

        # Independence penalty parameters
        self.independence_penalty_weight = args.independence_penalty if hasattr(args, 'independence_penalty') else 10.0
        self.use_independence_penalty = args.use_independence_penalty if hasattr(args, 'use_independence_penalty') else True
        
        # Track current stage
        self.current_stage = 1
        self.stage_epoch = 0
        
        print(f"🎯 Staged Training Configuration:")
        print(f"  Stage 1 (warm-up): {self.stage1_epochs} epochs")
        print(f"  Stage 2 (penalty): {self.stage2_epochs} epochs")
        print(f"  Stage 3 (full): {self.stage3_epochs} epochs")

        print(f"\n🔧 Zay Capacity Annealing:")
        if self.use_zay_annealing:
            print(f"  Status: ENABLED")
            print(f"  Stage 1: Suppressed (0.1)")
            print(f"  Stage 2: Ramp from {self.zay_capacity_start:.1f} → {self.zay_capacity_end:.1f}")
            print(f"  Stage 3: Full capacity (1.0)")
        else:
            print(f"  Status: DISABLED")
            print(f"  Stage 1: Suppressed (0.1)")
            print(f"  Stage 2-3: Full capacity (1.0)")

        print(f"\n🎯 Independence Penalty:")
        if self.use_independence_penalty:
            print(f"  Status: ENABLED")
            print(f"  Weight: {self.independence_penalty_weight}")
            print(f"  Applied in: Stage 2")
        else:
            print(f"  Status: DISABLED")
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Main training loop with staged approach."""

        # Validate dataloaders are not empty before training
        if len(train_loader) == 0:
            raise ValueError(
                "❌ Train loader is empty (0 batches). This likely means:\n"
                "   - OOD filtering removed all samples from training set\n"
                "   - Batch size is larger than the dataset\n"
                f"   - Check your data filtering settings and batch_size={self.args.batch_size}"
            )

        if len(val_loader) == 0:
            raise ValueError(
                "❌ Validation loader is empty (0 batches). This likely means:\n"
                "   - OOD filtering removed all samples from validation set\n"
                "   - Batch size is larger than the dataset\n"
                f"   - Check your data filtering settings and batch_size={self.args.batch_size}"
            )

        for epoch in range(num_epochs):
            # Determine current stage
            if epoch < self.stage1_epochs:
                self.current_stage = 1
                self.stage_epoch = epoch
            elif epoch < self.stage1_epochs + self.stage2_epochs:
                self.current_stage = 2
                self.stage_epoch = epoch - self.stage1_epochs
            else:
                self.current_stage = 3
                self.stage_epoch = epoch - self.stage1_epochs - self.stage2_epochs
            
            # Training phase with stage-specific parameters
            self.model.train()
            
            # Learning rate warm-up for stability
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                current_lr = self.initial_lr * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                # Reset to initial learning rate after warmup
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.initial_lr
            
            if self.beta_annealing:
                trn_current_beta = self.get_current_beta(epoch)
            else:
                trn_current_beta = self.beta_scale
                
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{self.args.epochs} [Stage {self.current_stage}] - LR: {current_lr:.6f}, Beta: {trn_current_beta:.4f}')
            
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, current_beta=trn_current_beta)

            # Validation phase (use fixed beta_scale for comparable metrics across epochs/stages)
            # This ensures validation loss is consistent for early stopping and performance tracking
            val_loss, val_metrics = self._validate(val_loader, epoch, current_beta=self.beta_scale)
            
            print(f'  Train Loss: {train_loss:.4f}')
            for k, v in train_metrics.items():
                print(f'  Train {k}: {v:.4f}')
                
            print(f'  Val Loss: {val_loss:.4f}')
            for k, v in val_metrics.items():
                print(f'  Val {k}: {v:.4f}')
            
            # Visualize latent spaces after each epoch
            self.visualize_latent_epoch(val_loader, epoch)
            
            # Early stopping check
            if self._check_early_stopping(val_loss, epoch, num_epochs, val_metrics):
                break
            # Best model is automatically saved in _check_early_stopping when validation improves

        self.epochs_trained = epoch + 1

    def _train_epoch(self, train_loader, epoch, current_beta) -> Tuple[float, Dict[str, float]]:
        """Training epoch with stage-specific loss computation."""
        train_loss = 0
        train_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0, 'independence_penalty': 0}
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1} [Stage {self.current_stage}]")
        
        for batch_idx, batch in train_pbar:
            x, y, hospital_id = process_batch(batch, self.device, dataset_type='wild')
            self.optimizer.zero_grad()

            # Stage-specific loss computation
            loss = self._compute_staged_loss(y, x, hospital_id, current_beta)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"❌ NaN loss detected in stage {self.current_stage}, skipping batch")
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            train_loss += loss.item()
            
            # Calculate metrics for every batch to be consistent with validation
            batch_metrics = _calculate_metrics(self.model, y, x, hospital_id, 'train')
            # Add independence penalty to metrics if enabled
            if self.use_independence_penalty:
                batch_metrics['independence_penalty'] = self._compute_independence_penalty(y, x, hospital_id).item()
            else:
                batch_metrics['independence_penalty'] = 0.0
            
            for k, v in batch_metrics.items():
                train_metrics_sum[k] += v
            
            train_pbar.set_postfix(loss=loss.item(), stage=self.current_stage)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}
        
        # Visualize reconstructions
        trn_sample_batch = select_diverse_sample_batch(train_loader, data_type='train', samples_per_domain=10)
        image_dir = os.path.join(self.reconstructions_dir, f'train_stage{self.current_stage}_epoch_{epoch}.png')
        visualize_reconstructions(self.model, epoch+1, trn_sample_batch, image_dir, args=self.args)

        return avg_train_loss, avg_train_metrics

    def _compute_staged_loss(self, y, x, a, current_beta):
        """Compute loss based on current training stage."""
        
        # Check for NaN in inputs
        if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(a).any():
            print("⚠️  NaN detected in inputs!")
            print(f"x has NaN: {torch.isnan(x).any()}")
            print(f"y has NaN: {torch.isnan(y).any()}")  
            print(f"a has NaN: {torch.isnan(a).any()}")
        
        # Get model outputs
        try:
            x_recon, z, qz, pzy, pzx, pza, pzay, y_hat, a_hat, zy, zx, zay, za = self.model.forward(y, x, a)
        except ValueError as e:
            print(f"❌ Error in model forward pass: {e}")
            print(f"Stage: {self.current_stage}, Epoch: {self.stage_epoch}")
            print(f"Input shapes - x: {x.shape}, y: {y.shape}, a: {a.shape}")
            print(f"Input ranges - x: [{x.min():.4f}, {x.max():.4f}]")
            raise e
        
        # Base reconstruction loss (always present)
        x_recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction='sum')
        
        # Compute KL divergences for each latent variable
        log_prob_z = qz.log_prob(z)
        log_prob_zy = log_prob_z[:, self.model.zy_index_range[0]:self.model.zy_index_range[1]]
        log_prob_zx = log_prob_z[:, self.model.zx_index_range[0]:self.model.zx_index_range[1]]
        log_prob_za = log_prob_z[:, self.model.za_index_range[0]:self.model.za_index_range[1]]

        # Handle DIVA mode (no zay component)
        if hasattr(self.model, 'diva') and self.model.diva:
            log_prob_zay = torch.zeros_like(log_prob_zy)
        else:
            log_prob_zay = log_prob_z[:, self.model.zay_index_range[0]:self.model.zay_index_range[1]]

        kl_zy = torch.sum(log_prob_zy - pzy.log_prob(zy))
        kl_zx = torch.sum(log_prob_zx - pzx.log_prob(zx))
        kl_za = torch.sum(log_prob_za - pza.log_prob(za))

        # KL for zay (0 for DIVA models)
        if hasattr(self.model, 'diva') and self.model.diva:
            kl_zay = torch.tensor(0.0, device=zy.device)
        else:
            kl_zay = torch.sum(log_prob_zay - pzay.log_prob(zay))
        
        # Classification losses
        y_target = y.long() if len(y.shape) == 1 else y.max(dim=1)[1]
        a_target = a.long() if len(a.shape) == 1 else a.max(dim=1)[1]
        y_cross_entropy = torch.nn.functional.cross_entropy(y_hat, y_target, reduction='sum')
        a_cross_entropy = torch.nn.functional.cross_entropy(a_hat, a_target, reduction='sum')
        
        # Stage-specific loss computation
        if self.current_stage == 1:
            # Stage 1: Foundation - suppress zay to force za/zy to learn their factors
            # Suppress zay by setting its capacity to near zero
            zay_capacity = 0.1  # Small but non-zero for numerical stability

            total_loss = (self.model.recon_weight * x_recon_loss +
                         current_beta * (self.model.beta_1 * kl_zy +
                                       self.model.beta_2 * kl_zx +
                                       self.model.beta_3 * zay_capacity * kl_zay +
                                       self.model.beta_4 * kl_za) +
                         self.model.alpha_1 * y_cross_entropy +
                         self.model.alpha_2 * a_cross_entropy)
            
        elif self.current_stage == 2:
            # Stage 2: Apply capacity annealing (if enabled) + independence penalty (if enabled)

            # Compute zay capacity based on annealing setting
            if self.use_zay_annealing:
                # Linearly ramp from zay_capacity_start to zay_capacity_end
                if self.stage2_epochs > 1:
                    progress = self.stage_epoch / (self.stage2_epochs - 1)
                else:
                    # Single epoch stage: immediately use end capacity
                    progress = 1.0
                progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
                zay_capacity = self.zay_capacity_start + progress * (self.zay_capacity_end - self.zay_capacity_start)
            else:
                # No annealing: constant full capacity
                zay_capacity = 1.0

            total_loss = (self.model.recon_weight * x_recon_loss +
                         current_beta * (self.model.beta_1 * kl_zy + 
                                       self.model.beta_2 * kl_zx + 
                                       self.model.beta_3 * zay_capacity * kl_zay +
                                       self.model.beta_4 * kl_za) +
                         self.model.alpha_1 * y_cross_entropy + 
                         self.model.alpha_2 * a_cross_entropy)
            
            # Add independence penalty if enabled
            if self.use_independence_penalty:
                independence_penalty = self._compute_independence_penalty(y, x, a)
                total_loss += self.independence_penalty_weight * independence_penalty
            
        else:  # Stage 3
            # Stage 3: Full training with all latent variables
            total_loss = (self.model.recon_weight * x_recon_loss + 
                         current_beta * (self.model.beta_1 * kl_zy + 
                                       self.model.beta_2 * kl_zx + 
                                       self.model.beta_3 * kl_zay + 
                                       self.model.beta_4 * kl_za) +
                         self.model.alpha_1 * y_cross_entropy + 
                         self.model.alpha_2 * a_cross_entropy)
        
        return total_loss
    
    def _compute_independence_penalty(self, y, x, a):
        """
        Compute independence penalty to prevent zay from capturing za/zy information.
        This encourages zay to only capture synergistic information.

        Note: Gradients are required for this penalty to influence training.
        Returns 0 for DIVA models (no zay component).
        """
        # Early return for DIVA models (no zay component to penalize)
        if hasattr(self.model, 'diva') and self.model.diva:
            return torch.tensor(0.0, device=x.device)

        # Get latent distribution and sample (use actual samples, not just encoder means)
        qz_loc, qz_scale = self.model.qz(x)
        qz = torch.distributions.Normal(qz_loc, qz_scale)
        z = qz.rsample()  # Sample with reparameterization for gradients

        # Extract latent components from sampled z
        zy = z[:, self.model.zy_index_range[0]:self.model.zy_index_range[1]]
        za = z[:, self.model.za_index_range[0]:self.model.za_index_range[1]]
        zay = z[:, self.model.zay_index_range[0]:self.model.zay_index_range[1]]

        # Check for NaN in sampled latents (can occur with numerical instability)
        if torch.isnan(zy).any() or torch.isnan(za).any() or torch.isnan(zay).any():
            print("⚠️  Warning: NaN detected in latent samples, returning zero independence penalty")
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Compute mutual information approximation using correlation
        # Penalty = |corr(zay, zy)| + |corr(zay, za)|

        # Normalize features with robust division by zero protection
        min_std = 1e-6  # Larger epsilon for numerical stability
        zy_norm = (zy - zy.mean(dim=0)) / zy.std(dim=0).clamp(min=min_std)
        za_norm = (za - za.mean(dim=0)) / za.std(dim=0).clamp(min=min_std)
        zay_norm = (zay - zay.mean(dim=0)) / zay.std(dim=0).clamp(min=min_std)

        # Compute correlations (with division by zero protection)
        batch_size = max(1, zay_norm.shape[0])  # Ensure minimum of 1 to prevent division by zero
        corr_zay_zy = torch.mean(torch.abs(torch.sum(zay_norm * zy_norm, dim=0) / batch_size))
        corr_zay_za = torch.mean(torch.abs(torch.sum(zay_norm * za_norm, dim=0) / batch_size))

        independence_penalty = corr_zay_zy + corr_zay_za

        return independence_penalty

    def _validate(self, val_loader, epoch, current_beta) -> Tuple[float, Dict[str, float]]:
        """Validation with stage-aware loss computation."""
        self.model.eval()
        val_loss = 0
        val_metrics_sum = {'recon_mse': 0, 'y_accuracy': 0, 'a_accuracy': 0, 'independence_penalty': 0}
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                       desc=f"Epoch {epoch+1} [Val-Stage {self.current_stage}]")
        
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                x, y, hospital_id = process_batch(batch, self.device, dataset_type='wild')

                # Use staged loss for validation too
                loss = self._compute_staged_loss(y, x, hospital_id, current_beta)
                val_loss += loss.item()
                
                batch_metrics = _calculate_metrics(self.model, y, x, hospital_id, 'val')
                if self.use_independence_penalty:
                    batch_metrics['independence_penalty'] = self._compute_independence_penalty(y, x, hospital_id).item()
                else:
                    batch_metrics['independence_penalty'] = 0.0
                
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v
                
                val_pbar.set_postfix(loss=loss.item(), stage=self.current_stage)
        
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        
        return val_loss, val_metrics 