"""
wandb_train.py
==============
Main training script for the ADNI Multimodal Neural CDE.
Logs all metrics to Weights & Biases.

Usage:
    python wandb_train.py                          # train baseline
    python wandb_train.py --model multimodal       # train multimodal
    python wandb_train.py --epochs 5 --subset 50   # quick test run

Requires: jax, equinox, diffrax, optax, wandb, numpy, pandas
"""

import argparse
import time
import os
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb

from ncde.data_loader import build_datasets, Normalizer
from ncde.model import create_model


# ═══════════════════════════════════════════════════════════════════════════
# Loss function
# ═══════════════════════════════════════════════════════════════════════════

def masked_mse_loss(
    model: eqx.Module,
    time: jax.Array,
    features: jax.Array,
    mask: jax.Array,
    target: jax.Array,
    lengths: jax.Array,
) -> jax.Array:
    """
    Compute masked MSE loss over a batch.

    Only computes loss at time steps with observed ADAS13 (the first feature
    column mask, since ADAS13 is feature index 0).

    Args:
        model: the NCDE model
        time: (B, T) time points
        features: (B, T, F) features
        mask: (B, T, F) observation masks
        target: (B, T) targets
        lengths: (B,) actual sequence lengths

    Returns:
        Scalar mean loss
    """
    batch_size = time.shape[0]

    def _single_loss(t, f, m, tgt, length):
        """Loss for a single subject."""
        preds = model(t, f, m, length)  # (T,)

        # Mask: only count observed ADAS13 (column 0) and within actual length
        time_mask = jnp.arange(t.shape[0]) < length
        adas_observed = m[:, 0] > 0.5  # ADAS13 is feature 0
        valid = time_mask & adas_observed

        sq_errors = (preds - tgt) ** 2
        # Mean over valid positions (avoid division by zero)
        n_valid = jnp.maximum(valid.sum(), 1)
        return (sq_errors * valid).sum() / n_valid

    # vmap over the batch
    losses = jax.vmap(_single_loss)(time, features, mask, target, lengths)
    return losses.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

@eqx.filter_jit
def compute_predictions_batch(
    model: eqx.Module,
    time: jax.Array,
    features: jax.Array,
    mask: jax.Array,
    lengths: jax.Array,
) -> jax.Array:
    """Run model on a batch and return predictions."""
    def _single(t, f, m, length):
        return model(t, f, m, length)
    return jax.vmap(_single)(time, features, mask, lengths)


def evaluate_model(
    model: eqx.Module,
    batches: list[dict],
    normalizer: Normalizer,
) -> dict:
    """
    Evaluate the model on a set of batches.

    Returns dict with MAE, RMSE, R², correlation (all in original ADAS13 scale).
    """
    all_preds = []
    all_targets = []
    all_valid = []

    for batch in batches:
        preds = compute_predictions_batch(
            model, batch["time"], batch["features"], batch["mask"], batch["lengths"]
        )
        # Collect predictions and targets
        for i in range(batch["time"].shape[0]):
            length = int(batch["lengths"][i])
            pred_seq = np.array(preds[i, :length])
            tgt_seq = np.array(batch["target"][i, :length])
            mask_seq = np.array(batch["mask"][i, :length, 0])  # ADAS13 mask

            valid_idx = mask_seq > 0.5
            if valid_idx.sum() > 0:
                all_preds.append(pred_seq[valid_idx])
                all_targets.append(tgt_seq[valid_idx])

    if len(all_preds) == 0:
        return {"mae": float("inf"), "rmse": float("inf"), "r2": 0.0, "corr": 0.0}

    # Concatenate and unnormalize
    preds_cat = np.concatenate(all_preds)
    tgt_cat = np.concatenate(all_targets)

    # Unnormalize to original ADAS13 scale
    preds_orig = preds_cat * normalizer.tgt_std + normalizer.tgt_mean
    tgt_orig = tgt_cat * normalizer.tgt_std + normalizer.tgt_mean

    # Metrics
    errors = preds_orig - tgt_orig
    mae = float(np.nanmean(np.abs(errors)))
    rmse = float(np.sqrt(np.nanmean(errors ** 2)))

    # R²
    ss_res = float(np.nansum(errors ** 2))
    tgt_mean = float(np.nanmean(tgt_orig))
    ss_tot = float(np.nansum((tgt_orig - tgt_mean) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    # Pearson correlation
    if len(preds_orig) > 1:
        corr = float(np.corrcoef(preds_orig, tgt_orig)[0, 1])
    else:
        corr = 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr, "n_predictions": len(preds_orig)}


# ═══════════════════════════════════════════════════════════════════════════
# Learning rate schedule: cosine annealing with warmup
# ═══════════════════════════════════════════════════════════════════════════

def create_lr_schedule(
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> optax.Schedule:
    """
    Create a learning rate schedule with linear warmup + cosine decay.

    Args:
        base_lr: peak learning rate after warmup
        warmup_epochs: number of warmup epochs
        total_epochs: total training epochs
        steps_per_epoch: number of optimizer steps per epoch

    Returns:
        optax Schedule
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = (total_epochs - warmup_epochs) * steps_per_epoch

    warmup_fn = optax.linear_schedule(
        init_value=base_lr * 0.01,
        end_value=base_lr,
        transition_steps=max(warmup_steps, 1),
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=max(decay_steps, 1),
        alpha=base_lr * 0.01,  # minimum LR
    )
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Training step (JIT-compiled)
# ═══════════════════════════════════════════════════════════════════════════

@eqx.filter_jit
def train_step(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict,
) -> tuple:
    """
    Single gradient descent step.

    Args:
        model: current model
        opt_state: optimizer state
        optimizer: optax optimizer
        batch: dict with time, features, mask, target, lengths

    Returns:
        (updated_model, updated_opt_state, loss_value)
    """
    loss_fn = lambda m: masked_mse_loss(
        m, batch["time"], batch["features"], batch["mask"], batch["target"], batch["lengths"]
    )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_loss_batch(model: eqx.Module, batch: dict) -> jax.Array:
    """Compute loss on a batch (no gradients)."""
    return masked_mse_loss(
        model, batch["time"], batch["features"], batch["mask"], batch["target"], batch["lengths"]
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    """Main training function with W&B logging."""

    # ─── Paths ────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent
    seq_dir = str(project_root / "outputs" / "sequences")
    ckpt_dir = project_root / "ncde" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ─── Initialize W&B ──────────────────────────────────────────────────
    run = wandb.init(
        entity="multincde_daml",
        project="multimodal_adni",
        name=f"{args.model}_ncde_h{args.hidden_dim}_lr{args.lr}",
        config={
            "model_type": args.model,
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "vf_width": args.vf_width,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "grad_clip_norm": args.grad_clip,
            "seed": args.seed,
            "subset": args.subset,
        },
        tags=[args.model, "ncde", "adni", "adas13"],
    )

    print(f"\n{'='*60}")
    print(f"  ADNI Multimodal Neural CDE Training")
    print(f"  Model: {args.model}")
    print(f"  W&B run: {run.name}")
    print(f"{'='*60}\n")

    # ─── Load data ───────────────────────────────────────────────────────
    train_batches, val_batches, test_batches, normalizer = build_datasets(
        seq_dir=seq_dir,
        batch_size=args.batch_size,
        subset=args.subset if args.subset > 0 else None,
    )

    feature_dim = int(train_batches[0]["features"].shape[2])
    wandb.config.update({"feature_dim": feature_dim, "n_train_batches": len(train_batches)})

    # ─── Create model ────────────────────────────────────────────────────
    key = jax.random.PRNGKey(args.seed)
    model = create_model(
        model_type=args.model,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        vf_width=args.vf_width,
        key=key,
    )

    # Count parameters
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"  Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ─── Optimizer: AdamW + grad clipping + LR schedule ──────────────────
    steps_per_epoch = len(train_batches)
    lr_schedule = create_lr_schedule(
        base_lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ─── Training loop ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_metrics = {}
    patience_counter = 0
    global_step = 0

    print(f"\n  Starting training for {args.epochs} epochs...")
    print(f"  Batches per epoch: {steps_per_epoch}")
    print(f"  Patience: {args.patience} epochs\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── Train epoch ──────────────────────────────────────────────────
        train_losses = []

        # Reshuffle batches each epoch
        rng = np.random.default_rng(args.seed + epoch)
        batch_order = rng.permutation(len(train_batches))

        for batch_idx in batch_order:
            batch = train_batches[batch_idx]
            model, opt_state, loss = train_step(model, opt_state, optimizer, batch)
            train_losses.append(float(loss))
            global_step += 1

            # Log per-step metrics
            if global_step % 10 == 0:
                wandb.log({
                    "train/step_loss": float(loss),
                    "train/learning_rate": float(lr_schedule(global_step)),
                    "global_step": global_step,
                }, step=global_step)

        train_loss_mean = float(np.mean(train_losses))

        # ── Validation epoch ─────────────────────────────────────────────
        val_losses = []
        for batch in val_batches:
            val_loss = eval_loss_batch(model, batch)
            val_losses.append(float(val_loss))
        val_loss_mean = float(np.mean(val_losses))

        epoch_time = time.time() - epoch_start

        # ── Compute full metrics every 5 epochs or at end ────────────────
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = evaluate_model(model, val_batches, normalizer)
            metric_str = (
                f"  MAE={val_metrics['mae']:.2f}  RMSE={val_metrics['rmse']:.2f}  "
                f"R²={val_metrics['r2']:.4f}  r={val_metrics['corr']:.4f}"
            )

            wandb.log({
                "val/mae": val_metrics["mae"],
                "val/rmse": val_metrics["rmse"],
                "val/r2": val_metrics["r2"],
                "val/correlation": val_metrics["corr"],
            }, step=global_step)
        else:
            metric_str = ""

        # ── Log epoch metrics ────────────────────────────────────────────
        wandb.log({
            "train/epoch_loss": train_loss_mean,
            "val/epoch_loss": val_loss_mean,
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
        }, step=global_step)

        # Print progress
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss_mean:.4f}  val_loss={val_loss_mean:.4f} | "
            f"{epoch_time:.1f}s{metric_str}"
        )

        # ── Early stopping check ─────────────────────────────────────────
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            patience_counter = 0

            # Save best model checkpoint
            ckpt_path = str(ckpt_dir / f"best_{args.model}.eqx")
            eqx.tree_serialise_leaves(ckpt_path, model)

            if metric_str:
                best_val_metrics = val_metrics

            print(f"        ▸ New best! Saved to {ckpt_path}")
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ─── Final evaluation on test set ────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Loading best model for test evaluation...")

    # Reload the best model
    best_model = create_model(
        model_type=args.model,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        vf_width=args.vf_width,
        key=key,
    )
    ckpt_path = str(ckpt_dir / f"best_{args.model}.eqx")
    best_model = eqx.tree_deserialise_leaves(ckpt_path, best_model)

    # Test metrics
    test_metrics = evaluate_model(best_model, test_batches, normalizer)
    val_metrics_final = evaluate_model(best_model, val_batches, normalizer)

    print(f"\n  +{'='*50}+")
    print(f"  |  Final Results ({args.model.upper()} NCDE)")
    print(f"  +{'='*50}+")
    print(f"  |  Val  MAE: {val_metrics_final['mae']:>8.2f}  RMSE: {val_metrics_final['rmse']:>8.2f}")
    print(f"  |  Val  R²:  {val_metrics_final['r2']:>8.4f}  Corr: {val_metrics_final['corr']:>8.4f}")
    print(f"  |  Test MAE: {test_metrics['mae']:>8.2f}  RMSE: {test_metrics['rmse']:>8.2f}")
    print(f"  |  Test R²:  {test_metrics['r2']:>8.4f}  Corr: {test_metrics['corr']:>8.4f}")
    print(f"  |  Predictions: {test_metrics['n_predictions']}")
    print(f"  +{'='*50}+\n")

    # Log final metrics to W&B
    wandb.run.summary.update({
        "test/mae": test_metrics["mae"],
        "test/rmse": test_metrics["rmse"],
        "test/r2": test_metrics["r2"],
        "test/correlation": test_metrics["corr"],
        "val_final/mae": val_metrics_final["mae"],
        "val_final/rmse": val_metrics_final["rmse"],
        "val_final/r2": val_metrics_final["r2"],
        "val_final/correlation": val_metrics_final["corr"],
        "model_type": args.model,
        "n_params": n_params,
    })

    wandb.finish()
    print("  W&B run finished. ✓")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train ADNI Multimodal Neural CDE with W&B logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "multimodal"],
                        help="Model type: 'baseline' (clinical only) or 'multimodal' (with imaging encoder)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="CDE hidden state dimension")
    parser.add_argument("--embed_dim", type=int, default=32,
                        help="Imaging encoder embedding dimension (multimodal only)")
    parser.add_argument("--vf_width", type=int, default=128,
                        help="Vector field MLP width")

    # Training
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")

    # Data
    parser.add_argument("--subset", type=int, default=0,
                        help="Use only this many subjects per split (0 = all)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
