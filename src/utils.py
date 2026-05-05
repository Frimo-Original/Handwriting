import torch


def mdn_loss(pi, mu, sigma, rho, target_xy, reduction="mean"):
    target = target_xy.unsqueeze(1).expand_as(mu)

    mu_x = mu[:, :, 0]
    mu_y = mu[:, :, 1]
    sigma_x = torch.clamp(sigma[:, :, 0], min=1e-4, max=1e4)
    sigma_y = torch.clamp(sigma[:, :, 1], min=1e-4, max=1e4)
    rho = torch.clamp(rho, -0.999, 0.999)

    dx = target[:, :, 0] - mu_x
    dy = target[:, :, 1] - mu_y
    one_minus_rho2 = torch.clamp(1.0 - rho**2, min=1e-5)

    z = (dx / sigma_x) ** 2
    z = z + (dy / sigma_y) ** 2
    z = z - 2.0 * rho * dx * dy / (sigma_x * sigma_y)
    z = z / one_minus_rho2

    log_norm = -0.5 * z
    log_norm = log_norm - torch.log(
        2.0 * torch.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho2)
    )

    log_prob = torch.logsumexp(torch.log(torch.clamp(pi, min=1e-8)) + log_norm, dim=1)
    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def sample_mdn(pi, mu, sigma, rho):
    batch_size, n_mix = pi.shape
    idx = torch.multinomial(pi, 1)
    idx_expanded = idx.unsqueeze(-1).expand(-1, 1, 2)
    mean = torch.gather(mu, 1, idx_expanded).squeeze(1)
    std = torch.gather(sigma, 1, idx_expanded).squeeze(1)
    corr = torch.gather(rho, 1, idx).squeeze(1)

    eps = torch.randn(batch_size, 2, device=pi.device, dtype=pi.dtype)
    x = mean[:, 0] + std[:, 0] * eps[:, 0]
    y = mean[:, 1] + std[:, 1] * (
        corr * eps[:, 0] + torch.sqrt(torch.clamp(1.0 - corr**2, min=1e-5)) * eps[:, 1]
    )
    return torch.stack([x, y], dim=1)
