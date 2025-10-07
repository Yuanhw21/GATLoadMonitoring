import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim


def project_to_aggregate(device_power: torch.Tensor, residual: Optional[torch.Tensor], aggregate: torch.Tensor, include_residual: bool, eps: float = 1e-6):
    """Project predictions so that device sums match the aggregate load."""
    device_sum = device_power.sum(dim=1)

    if include_residual and residual is not None:
        target_device_sum = torch.clamp(aggregate - residual, min=0.0)
        scale = torch.where(device_sum > eps, target_device_sum / (device_sum + eps), torch.zeros_like(device_sum))
        device_adj = device_power * scale.unsqueeze(-1)
        residual_adj = aggregate - device_adj.sum(dim=1)
        return device_adj, residual_adj

    scale = torch.where(device_sum > eps, aggregate / (device_sum + eps), torch.zeros_like(device_sum))
    device_adj = device_power * scale.unsqueeze(-1)
    return device_adj, None


def compute_losses(outputs: Dict[str, torch.Tensor], target_raw: torch.Tensor, aggregate: torch.Tensor, device_states: torch.Tensor, include_residual: bool, weights: Dict[str, float], pos_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    device_target = target_raw[:, 1:] if include_residual else target_raw
    device_loss = F.smooth_l1_loss(outputs["device_power"], device_target)

    if include_residual and outputs["residual"] is not None:
        residual_target = target_raw[:, 0]
        residual_loss = F.smooth_l1_loss(outputs["residual"], residual_target)
        power_loss = (device_loss + residual_loss) * 0.5
    else:
        power_loss = device_loss

    on_logits = outputs["on_logits"]
    if pos_weight is not None:
        onoff_loss = F.binary_cross_entropy_with_logits(
            on_logits,
            device_states,
            pos_weight=pos_weight.to(on_logits.device, dtype=on_logits.dtype),
        )
    else:
        onoff_loss = F.binary_cross_entropy_with_logits(on_logits, device_states)

    total_pred = outputs["device_power"].sum(dim=1)
    if include_residual and outputs["residual"] is not None:
        total_pred = total_pred + outputs["residual"]
    consistency_loss = torch.mean(torch.abs(total_pred - aggregate))

    total_loss = (
        weights.get("power", 1.0) * power_loss
        + weights.get("onoff", 1.0) * onoff_loss
        + weights.get("consistency", 0.1) * consistency_loss
    )

    return {
        "total": total_loss,
        "power": power_loss.detach(),
        "onoff": onoff_loss.detach(),
        "consistency": consistency_loss.detach(),
    }


def train_epoch(model, dataloader, optimizer, device, edge_index, include_residual, loss_weights, grad_clip, pos_weight: Optional[torch.Tensor] = None):
    model.train()
    edge_index = edge_index.to(device)
    running = {"total": 0.0, "power": 0.0, "onoff": 0.0, "consistency": 0.0}
    num_samples = 0

    for features, _, target_raw, aggregate, device_states in dataloader:
        features = features.to(device)
        target_raw = target_raw.to(device)
        aggregate = aggregate.to(device)
        device_states = device_states.to(device)

        optimizer.zero_grad()
        outputs = model(features, edge_index)
        losses = compute_losses(outputs, target_raw, aggregate, device_states, include_residual, loss_weights, pos_weight)
        losses["total"].backward()
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = features.size(0)
        num_samples += batch_size
        for key in running:
            running[key] += losses[key].item() * batch_size

    return {key: value / max(1, num_samples) for key, value in running.items()}


def evaluate(model, dataloader, device, edge_index, include_residual, loss_weights, project: bool = True, pos_weight: Optional[torch.Tensor] = None):
    model.eval()
    edge_index = edge_index.to(device)

    running = {"total": 0.0, "power": 0.0, "onoff": 0.0, "consistency": 0.0}
    num_samples = 0

    preds, targets, aggregates, residuals, on_probs = [], [], [], [], []

    with torch.no_grad():
        for features, _, target_raw, aggregate, device_states in dataloader:
            features = features.to(device)
            target_raw = target_raw.to(device)
            aggregate = aggregate.to(device)
            device_states = device_states.to(device)

            outputs = model(features, edge_index)
            losses = compute_losses(outputs, target_raw, aggregate, device_states, include_residual, loss_weights, pos_weight)

            batch_size = features.size(0)
            num_samples += batch_size
            for key in running:
                running[key] += losses[key].item() * batch_size

            device_power = outputs["device_power"]
            residual_pred = outputs["residual"]
            if project:
                device_power, residual_adj = project_to_aggregate(device_power, residual_pred, aggregate, include_residual)
            else:
                residual_adj = residual_pred

            device_target = target_raw[:, 1:] if include_residual else target_raw

            preds.append(device_power.cpu())
            targets.append(device_target.cpu())
            aggregates.append(aggregate.cpu())
            if include_residual:
                residuals.append(residual_adj.cpu())
            on_probs.append(torch.sigmoid(outputs["on_logits"]).cpu())

    results = {key: value / max(1, num_samples) for key, value in running.items()}
    results["predictions"] = torch.cat(preds, dim=0).numpy() if preds else np.empty((0,))
    results["targets"] = torch.cat(targets, dim=0).numpy() if targets else np.empty((0,))
    results["aggregates"] = torch.cat(aggregates, dim=0).numpy() if aggregates else np.empty((0,))
    results["on_prob"] = torch.cat(on_probs, dim=0).numpy() if on_probs else np.empty((0,))
    if include_residual:
        results["residuals"] = torch.cat(residuals, dim=0).numpy() if residuals else np.empty((0,))
    else:
        results["residuals"] = None

    return results


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    edge_index,
    loss_weights=None,
    scheduler=None,
    grad_clip: Optional[float] = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
):
    loss_weights = loss_weights or {"power": 1.0, "onoff": 0.5, "consistency": 0.2}
    history = []
    best_state = None
    best_loss = float("inf")

    include_residual = model.metadata.include_residual

    for epoch in range(1, num_epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            edge_index,
            include_residual,
            loss_weights,
            grad_clip,
            pos_weight,
        )
        val_stats = evaluate(
            model,
            val_loader,
            device,
            edge_index,
            include_residual,
            loss_weights,
            project=True,
            pos_weight=pos_weight,
        )

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["total"])
            else:
                scheduler.step()

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"train_total={train_stats['total']:.4f} "
            f"val_total={val_stats['total']:.4f} "
            f"val_power={val_stats['power']:.4f} "
            f"val_onoff={val_stats['onoff']:.4f} "
            f"val_cons={val_stats['consistency']:.4f}",
            flush=True,
        )

        history.append({"epoch": epoch, "train": train_stats, "val": {k: v for k, v in val_stats.items() if isinstance(v, float)}})

        if val_stats["total"] < best_loss:
            best_loss = val_stats["total"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval = evaluate(
        model,
        val_loader,
        device,
        edge_index,
        include_residual,
        loss_weights,
        project=True,
        pos_weight=pos_weight,
    )
    return model, history, final_eval
