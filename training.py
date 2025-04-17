import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mymodels import UNet3D
from dataset import HeartMRIDataset  # Assume you define a dataset class elsewhere
import numpy as np
import torch.nn.functional as F

# Dice loss implementation
def dice_loss(pred, target, epsilon=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()

# Hybrid loss: BCE + Dice
def hybrid_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc

# Match spatial dimensions of skip connections
def match_size_and_concat(upsampled, bypass):
    diff_d = bypass.shape[2] - upsampled.shape[2]
    diff_h = bypass.shape[3] - upsampled.shape[3]
    diff_w = bypass.shape[4] - upsampled.shape[4]

    upsampled = F.pad(upsampled, [
        diff_w // 2, diff_w - diff_w // 2,
        diff_h // 2, diff_h - diff_h // 2,
        diff_d // 2, diff_d - diff_d // 2
    ])

    return torch.cat([upsampled, bypass], dim=1)

import mymodels
mymodels.match_size_and_concat = match_size_and_concat

# Training loop
def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=50, log_dir="runs/heart_mri"):
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()  # for mixed precision
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            with torch.cuda.amp.autocast():  # mixed precision
                outputs = model(images)
                loss = hybrid_loss(outputs, masks)

            scaler.scale(loss).backward()

            if (i + 1) % 2 == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        model.eval()
        val_loss = 0.0
        dice_score_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = hybrid_loss(outputs, masks)
                val_loss += loss.item()

                # Dice score tracking
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice_score = (2. * intersection) / (union + 1e-5)
                dice_score_total += dice_score.item()

        val_loss /= len(val_loader)
        avg_dice = dice_score_total / len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Dice/Validation", avg_dice, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {avg_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model_testing.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # if early_stop_counter >= early_stop_patience:
        #     print("Early stopping triggered.")
        #     break

        if scheduler:
            scheduler.step(val_loss)

        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"

    train_dataset = HeartMRIDataset(data_dir=data_dir, split='train')
    val_dataset = HeartMRIDataset(data_dir=data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet3D(in_channels=1, out_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=50)
