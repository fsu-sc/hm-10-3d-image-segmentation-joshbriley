import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mymodels import UNet3D
from dataset import HeartMRIDataset
from training import dice_loss
import torch.nn.functional as F

# Dice coefficient (not loss)
def dice_coef(pred, target, epsilon=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

# Visualization function
def visualize_prediction(image, mask, pred, idx=0, writer=None):
    image = image.cpu().numpy()[0, 0, :, :, :]  # remove channel dim
    mask = mask.cpu().numpy()[0, 0, :, :, :]
    pred = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()[0, 0, :, :, :]

    mid_slice = image.shape[2] // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image[:, :, mid_slice], cmap='gray')
    axs[0].set_title('MRI Image')
    axs[1].imshow(mask[:, :, mid_slice], cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(pred[:, :, mid_slice], cmap='gray')
    axs[2].set_title('Prediction')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"results/sample_{idx}.png")
    plt.close()

    if writer:
        img_slice = torch.tensor(image[:, :, mid_slice]).float()
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-5)
        mask_slice = torch.tensor(mask[:, :, mid_slice]).unsqueeze(0)
        pred_slice = torch.tensor(pred[:, :, mid_slice]).unsqueeze(0)

        writer.add_image(f"Image/{idx}", img_slice.unsqueeze(0), global_step=0)
        writer.add_image(f"Mask/{idx}", mask_slice, global_step=0)
        writer.add_image(f"Prediction/{idx}", pred_slice, global_step=0)

if __name__ == "__main__":
    data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
    val_dataset = HeartMRIDataset(data_dir=data_dir, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load("runs/heart_mri/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    dice_scores = []
    os.makedirs("results", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/heart_mri/eval")

    # Log model architecture
    sample_input = torch.randn(1, 1, 64, 64, 64).to(device)
    writer.add_graph(model, sample_input)

    total_val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            image, mask = batch
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            loss = dice_loss(output, mask)
            dice = dice_coef(output, mask)
            dice_scores.append(dice)
            total_val_loss += loss.item()

            writer.add_scalar("DiceScore/val", dice, i)
            writer.add_scalar("Loss/val", loss.item(), i)

            if i < 5:
                visualize_prediction(image, mask, output, idx=i, writer=writer)

    avg_val_dice = np.mean(dice_scores)
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Average Dice Score on Validation Set: {avg_val_dice:.4f}")
    print(f"Average Loss on Validation Set: {avg_val_loss:.4f}")

    writer.add_scalar("DiceScore/val_avg", avg_val_dice, 0)
    writer.add_scalar("Loss/val_avg", avg_val_loss, 0)
    writer.close()