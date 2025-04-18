import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Set data path
data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
images_dir = os.path.join(data_dir, "imagesTr")
labels_dir = os.path.join(data_dir, "labelsTr")

# List image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

print(f"Number of training images: {len(image_files)}")

# Load a sample image and label (change index to 1 for a different image)
sample_image_path = os.path.join(images_dir, image_files[1])
sample_label_path = os.path.join(labels_dir, label_files[1])

img = nib.load(sample_image_path)
label = nib.load(sample_label_path)

img_data = img.get_fdata()
label_data = label.get_fdata()

# Print image shape and voxel spacing
print("Image shape:", img_data.shape)
print("Voxel spacing:", img.header.get_zooms())

# Display sample slices from axial, sagittal, coronal views
def show_slices(image, label, output_path="slices.png"):
    mid_slices = [s//2 for s in image.shape]

    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    axs[0, 0].imshow(image[mid_slices[0], :, :], cmap="gray")
    axs[0, 0].set_title("Axial Slice")
    axs[1, 0].imshow(label[mid_slices[0], :, :])
    axs[1, 0].set_title("Axial Label")

    axs[0, 1].imshow(image[:, mid_slices[1], :], cmap="gray")
    axs[0, 1].set_title("Sagittal Slice")
    axs[1, 1].imshow(label[:, mid_slices[1], :])
    axs[1, 1].set_title("Sagittal Label")

    axs[0, 2].imshow(image[:, :, mid_slices[2]], cmap="gray")
    axs[0, 2].set_title("Coronal Slice")
    axs[1, 2].imshow(label[:, :, mid_slices[2]])
    axs[1, 2].set_title("Coronal Label")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure instead of showing it
    plt.close()

show_slices(img_data, label_data, output_path="sample_slices.png")
