import torch
from torch.optim import lr_scheduler


def train_one_epoch_seg(model, dataloader, criterion, optimizer, device):
    model = model.to(device)
    model.train()
    total_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()
        labels = labels.float() / 255.0
        #labels = labels.float()

        # Forward pass
        outputs = model(inputs)

        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(0)

        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss