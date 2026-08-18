"""Training and evaluation functions for the CANN
"""

import torch


def train_model(model, train_loader, optimizer, loss_fn, epochs, device):
    """Train a CANN

    Parameters
    model: CANN model to train
    train_loader: DataLoader containing normalized input/target pairs
    optimizer: Optimizer used for parameter updates
    loss_fn: Loss function used for training
    epochs: Number of training epochs
    device: Device used for training

    Returns
    model: Trained model
    losses: Mean training loss for each epoch
    """
    model.to(device)
    losses = torch.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = model(x)
            loss = loss_fn(prediction.squeeze(-1), target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(train_loader)
        losses[epoch] = mean_loss

        print(
            f"Epoch {epoch + 1}/{epochs} Done, "
            f"Mean Loss: {mean_loss}"
        )

    return model, losses


def evaluate_model(model, x, target, loss_fn, device):
    """Evaluate a trained CANN on a dataset

    Parameters
    model: Trained CANN model
    x: Normalized CANN inputs
    target: Normalized analytical target values
    loss_fn: Loss function used for evaluation
    device: Device used for inference

    Returns
    prediction: Model predictions on the CPU
    target: Corresponding target values on the CPU
    loss: Evaluation loss
    """
    model.eval()

    with torch.no_grad():
        x_device = x.to(device)
        target_device = target.to(device)

        prediction = model(x_device).squeeze(-1)
        loss = loss_fn(prediction, target_device).item()

    return prediction.cpu(), target_device.cpu(), loss