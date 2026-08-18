"""Train and evaluate the CANN material model."""

from pathlib import Path

import torch
from torch import nn

from defs.cann import CANN
from defs.data import (
    compute_normalization_stats,
    create_dataloader,
    denormalize_energy,
    generate_cann_data,
    normalize_data,
)
from defs.training import evaluate_model, train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device("cpu")
    dtype = torch.float64

    mu = 384.614
    lam = 576.923

    min_I1 = 0.08
    max_I1 = 200.0
    min_J = 0.04
    max_J = 100.0

    train_samples = 300
    test_samples = 30

    input_dim = 2
    hidden_dim = 32
    output_dim = 1

    batch_size = 8
    shuffle = True
    drop_last = False

    epochs = 20
    learning_rate = 0.001

    model_dir = Path("models")
    model_file = model_dir / "cann_neo_hookean.pt"

    train_x_raw, train_y_raw = generate_cann_data(
        n_samples=train_samples,
        min_I1=min_I1,
        max_I1=max_I1,
        min_J=min_J,
        max_J=max_J,
        mu=mu,
        lam=lam,
        device=data_device,
        dtype=dtype,
    )

    test_x_raw, test_y_raw = generate_cann_data(
        n_samples=test_samples,
        min_I1=min_I1,
        max_I1=max_I1,
        min_J=min_J,
        max_J=max_J,
        mu=mu,
        lam=lam,
        device=data_device,
        dtype=dtype,
    )

    stats = compute_normalization_stats(train_x_raw, train_y_raw)

    train_x, train_y = normalize_data(
        train_x_raw,
        train_y_raw,
        stats,
    )

    test_x, test_y = normalize_data(
        test_x_raw,
        test_y_raw,
        stats,
    )

    train_loader = create_dataloader(
        train_x,
        train_y,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    model = CANN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device=device, dtype=dtype)

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model, train_losses = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
    )

    prediction_norm, target_norm, test_loss = evaluate_model(
        model=model,
        x=test_x,
        target=test_y,
        loss_fn=loss_fn,
        device=device,
    )

    prediction = denormalize_energy(prediction_norm, stats)
    target = denormalize_energy(target_norm, stats)

    physical_mse = torch.mean((prediction - target) ** 2).item()

    print(f"\nNormalized test MSE: {test_loss:.6e}")
    print(f"Physical test MSE:   {physical_mse:.6e} Pa^2")

    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "mu": mu,
            "lam": lam,
            "min_I1": min_I1,
            "max_I1": max_I1,
            "min_J": min_J,
            "max_J": max_J,
            "x_mean": stats.x_mean,
            "x_std": stats.x_std,
            "y_mean": stats.y_mean,
            "y_std": stats.y_std,
            "train_losses": train_losses,
        },
        model_file,
    )

    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()