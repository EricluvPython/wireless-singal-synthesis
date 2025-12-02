#!/usr/bin/env python3
"""
Physics-guided loss functions for MNIST experiments

These loss functions mimic the physics priors from wireless RSS data:
1. L_vert: Vertical attenuation (row-wise intensity matching)
2. L_occ: Occupancy (column-wise brightness probability matching)
3. L_smooth: Smoothness penalty (spatial gradient regularization)
4. L_handover: Row transition consistency (center row transition count)

All functions operate on batches of images in shape [B, 1, 28, 28] and are fully differentiable.
"""

import torch
import torch.nn.functional as F


def vertical_attenuation_loss(x_hat, labels, priors, device='cuda'):
    """
    L_vert: Vertical attenuation loss
    
    Compares row-wise mean pixel intensity of generated images to data statistics.
    For images: computes mean per row
    For RSS vectors: computes mean per feature
    
    Args:
        x_hat: Generated data (B, 1, H, W) for images or (B, 1, 1, num_features) for RSS
        labels: Class labels (B,) 
        priors: Dict containing 'mu_data' of shape (num_classes, H) or (num_classes, num_features)
        device: Device for computation
    
    Returns:
        Scalar loss value
    """
    B = x_hat.shape[0]
    
    # Handle both image format (B, 1, H, W) and RSS vector format (B, 1, 1, num_features)
    if x_hat.dim() == 4 and x_hat.shape[2] == 1:
        # RSS vector format: (B, 1, 1, num_features) -> (B, num_features)
        mu_gen = x_hat.squeeze(1).squeeze(1)
    else:
        # Image format: (B, 1, H, W) -> (B, H)
        # Compute row-wise mean
        mu_gen = x_hat.squeeze(1).mean(dim=2)  # Mean over columns -> row averages
    
    # Get expected means for each label
    mu_data = priors['mu_data'].to(device)  # (num_classes, H) or (num_classes, num_features)
    mu_expected = mu_data[labels]  # (B, H) or (B, num_features)
    
    # L1 loss
    loss = F.l1_loss(mu_gen, mu_expected)
    
    return loss


def occupancy_loss(x_hat, labels, priors, threshold=0.0, device='cuda'):
    """
    L_occ: Occupancy loss (brightness probability per column)
    
    For images: compute probability that each column contains bright pixels
    For RSS: compute probability that each cell tower is "visible" (RSS > threshold)
    
    Args:
        x_hat: Generated data (B, 1, H, W) for images or (B, 1, 1, num_features) for RSS
        labels: Class labels (B,)
        priors: Dict containing 'p_occ_data' of shape (num_classes, W) or (num_classes, num_features)
        threshold: Brightness/RSS threshold (default 0.0 for standardized data, 0.5 for images)
        device: Device for computation
    
    Returns:
        Scalar loss value
    """
    B = x_hat.shape[0]
    
    # Auto-detect threshold based on data format
    if x_hat.dim() == 4 and x_hat.shape[2] == 1:
        # RSS vector format - use 0.0 for standardized data
        threshold = 0.0
    elif threshold == 0.0:
        # Image format - use 0.5 as default
        threshold = 0.5
    
    # Use soft approximation instead of hard binarization to maintain gradients
    temperature = 10.0
    bright = torch.sigmoid((x_hat - threshold) * temperature)
    
    # Handle both image and RSS vector formats
    if x_hat.dim() == 4 and x_hat.shape[2] == 1:
        # RSS vector format: (B, 1, 1, num_features) -> (B, num_features)
        has_bright = bright.squeeze(1).squeeze(1)  # Already per-feature
    else:
        # Image format: (B, 1, H, W) -> (B, W)
        # For each column, get maximum brightness (soft max over rows)
        bright_squeezed = bright.squeeze(1)  # (B, H, W)
        soft_max_brightness = torch.logsumexp(bright_squeezed * temperature, dim=1) / temperature  # (B, W)
        has_bright = torch.sigmoid(soft_max_brightness * temperature)
    
    # Average across batch to get occupancy probability per feature/column
    p_occ_gen = has_bright.mean(dim=0)  # (W,) or (num_features,)
    
    # Get expected occupancy for the labels in this batch
    p_occ_data = priors['p_occ_data'].to(device)  # (num_classes, W) or (num_classes, num_features)
    
    # Average the expected occupancies across the batch labels
    p_occ_expected = p_occ_data[labels].mean(dim=0)  # (W,) or (num_features,)
    
    # MSE loss instead of KL (more stable with gradients)
    loss = F.mse_loss(p_occ_gen, p_occ_expected)
    
    return loss


def smoothness_loss(x_hat, priors, device='cuda'):
    """
    L_smooth: Smoothness penalty
    
    Penalize gradients/variations that exceed the threshold tau.
    For images: penalize spatial gradients
    For RSS: penalize variations between adjacent features
    
    Args:
        x_hat: Generated data (B, 1, H, W) for images or (B, 1, 1, num_features) for RSS
        priors: Dict containing 'tau' (scalar threshold)
        device: Device for computation
    
    Returns:
        Scalar loss value
    """
    tau = priors['tau']
    
    # Handle both image and RSS vector formats
    if x_hat.dim() == 4 and x_hat.shape[2] == 1:
        # RSS vector format: (B, 1, 1, num_features) -> variations between features
        x_squeezed = x_hat.squeeze(1).squeeze(1)  # (B, num_features)
        delta = torch.abs(x_squeezed[:, 1:] - x_squeezed[:, :-1])  # (B, num_features-1)
        penalty = F.relu(delta - tau)
        loss = penalty.sum() / x_hat.shape[0]
    else:
        # Image format: compute horizontal and vertical differences
        delta_h = torch.abs(x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1])
        delta_v = torch.abs(x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :])
        
        penalty_h = F.relu(delta_h - tau)
        penalty_v = F.relu(delta_v - tau)
        
        loss = (penalty_h.sum() + penalty_v.sum()) / x_hat.shape[0]
    
    return loss


def handover_loss(x_hat, labels, priors, threshold=0.0, device='cuda'):
    """
    L_handover: Transition consistency loss
    
    For images: counts transitions in the center row after binarization
    For RSS: counts variations between adjacent features
    
    Args:
        x_hat: Generated data (B, 1, H, W) for images or (B, 1, 1, num_features) for RSS
        labels: Class labels (B,)
        priors: Dict containing 'N_mean' and 'N_std' of shape (num_classes,)
        threshold: Binarization threshold (default 0.0 for standardized data, 0.5 for images)
        device: Device for computation
    
    Returns:
        Scalar loss value
    """
    B = x_hat.shape[0]
    temperature = 10.0
    
    # Auto-detect threshold based on data format
    if x_hat.dim() == 4 and x_hat.shape[2] == 1:
        # RSS vector format - use 0.0 for standardized data
        threshold = 0.0
    elif threshold == 0.0:
        # Image format - use 0.5 as default
        threshold = 0.5
    
    # Handle both image and RSS vector formats
    if x_hat.dim() == 4 and x_hat.shape[2] == 1:
        # RSS vector format: (B, 1, 1, num_features) -> (B, num_features)
        values = x_hat.squeeze(1).squeeze(1)
        binary_values = torch.sigmoid((values - threshold) * temperature)
        diffs = torch.abs(binary_values[:, 1:] - binary_values[:, :-1])
    else:
        # Image format: extract center row
        img_size = x_hat.shape[2]
        center_row = img_size // 2
        center_rows = x_hat[:, 0, center_row, :]
        binary_rows = torch.sigmoid((center_rows - threshold) * temperature)
        diffs = torch.abs(binary_rows[:, 1:] - binary_rows[:, :-1])
    
    # Sum "soft" transitions per sample
    N_gen = diffs.sum(dim=1)  # (B,)
    
    # Get expected transitions for each label
    N_mean = priors['N_mean'].to(device)  # (num_classes,)
    N_std = priors['N_std'].to(device)    # (num_classes,)
    
    N_expected_mean = N_mean[labels]  # (B,)
    
    # Use simple L1 or MSE loss instead of normalized error
    # Normalized error causes issues when std is very small (near 0)
    # which is common for RSS data where transitions are minimal
    loss = F.mse_loss(N_gen, N_expected_mean)
    
    return loss


# Track if we've already printed loss status messages
_loss_status_printed = {'L_occ': False, 'L_handover': False}

def compute_physics_losses(x_hat, labels, priors, device='cuda'):
    """
    Compute all four physics-guided losses
    
    Automatically detects and skips uninformative losses based on priors.
    
    Args:
        x_hat: Generated/predicted images (B, 1, 28, 28) or RSS vectors (B, 1, 1, num_features)
        labels: Digit/class labels (B,)
        priors: Dict containing all precomputed priors
        device: Device for computation
    
    Returns:
        Dict with individual loss components (uninformative losses set to 0)
    """
    global _loss_status_printed
    losses = {}
    
    # Vertical attenuation - always compute (most informative)
    losses['L_vert'] = vertical_attenuation_loss(x_hat, labels, priors, device)
    
    # Occupancy - skip if all values are near 1.0 (uninformative)
    p_occ_data = priors['p_occ_data']
    if p_occ_data.min() < 0.95:  # Has some variation
        losses['L_occ'] = occupancy_loss(x_hat, labels, priors, device=device)
        if not _loss_status_printed['L_occ']:
            print(f"[DEBUG] L_occ enabled: p_occ range [{p_occ_data.min():.3f}, {p_occ_data.max():.3f}]")
            _loss_status_printed['L_occ'] = True
    else:
        losses['L_occ'] = torch.tensor(0.0, device=device)
        if not _loss_status_printed['L_occ']:
            print(f"[INFO] L_occ disabled: p_occ all ~1.0 (range [{p_occ_data.min():.3f}, {p_occ_data.max():.3f}])")
            _loss_status_printed['L_occ'] = True
    
    # Smoothness - always compute (general regularizer)
    losses['L_smooth'] = smoothness_loss(x_hat, priors, device)
    
    # Handover - skip if mean transitions are very low (< 0.1) everywhere
    N_mean = priors['N_mean']
    if N_mean.max() > 0.1:  # Has meaningful transitions
        losses['L_handover'] = handover_loss(x_hat, labels, priors, device=device)
        if not _loss_status_printed['L_handover']:
            print(f"[DEBUG] L_handover enabled: N_mean range [{N_mean.min():.3f}, {N_mean.max():.3f}]")
            _loss_status_printed['L_handover'] = True
    else:
        losses['L_handover'] = torch.tensor(0.0, device=device)
        if not _loss_status_printed['L_handover']:
            print(f"[INFO] L_handover disabled: N_mean all <0.1 (range [{N_mean.min():.3f}, {N_mean.max():.3f}])")
            _loss_status_printed['L_handover'] = True
    
    return losses


def test_losses():
    """Test that all loss functions work correctly"""
    print("Testing physics loss functions...")
    
    # Create dummy data with gradients enabled
    B, C, H, W = 4, 1, 28, 28
    x_hat = torch.rand(B, C, H, W, requires_grad=True)
    labels = torch.randint(0, 10, (B,))
    
    # Create dummy priors
    priors = {
        'mu_data': torch.rand(10, 28),
        'p_occ_data': torch.rand(10, 28),
        'tau': 0.1,
        'N_mean': torch.rand(10) * 5,
        'N_std': torch.rand(10) * 2,
    }
    
    # Test each loss
    losses = compute_physics_losses(x_hat, labels, priors, device='cpu')
    
    for name, value in losses.items():
        assert value.dim() == 0, f"{name} should be scalar"
        assert value.requires_grad, f"{name} should require gradients"
        print(f"✓ {name}: {value.item():.4f}")
    
    # Test backward pass
    total_loss = sum(losses.values())
    total_loss.backward()
    assert x_hat.grad is not None, "Gradients should flow back to x_hat"
    
    print("\n✓ All physics loss functions working correctly!")
    print("✓ Gradients flow correctly!")


if __name__ == "__main__":
    test_losses()
