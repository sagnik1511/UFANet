import os
import torch
from termcolor import cprint
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import Resize


def predict_single_patch(model, patch, patch_num, grid_shape, store_directory, device, figure_size):
    curr_batch_size = 1 if len(patch.shape) == 3 else patch.shape[0]
    max_el = grid_shape[0] * grid_shape[1]
    if device != "cpu":
        cprint("Patch loaded on device...", "blue")
        patch = patch.cuda()
    with torch.no_grad():
        predictions = model(patch)
        real_grid = patch[:min(curr_batch_size, max_el), ...]
        mask_grid = predictions[:min(curr_batch_size, max_el), ...]
        mask_grid = torch.cat([mask_grid, mask_grid, mask_grid], dim=1)
    reshaped_real_grid = Resize((mask_grid.shape[-1], mask_grid.shape[-1]))(real_grid)
    image_grid = make_grid(torch.cat([reshaped_real_grid, mask_grid], dim=0), nrow=grid_shape[0])
    if device != "cpu":
        image_grid = image_grid.cpu()
    image_grid = image_grid.permute(1, 2, 0)
    filename = os.path.join(store_directory, f"patch_{patch_num}.png")
    plt.figure(figsize=figure_size)
    plt.imshow(image_grid)
    plt.savefig(filename)


def predict(model, data_loader, store_num_patches, grid_shape, store_directory, device, figure_size):
    if device != "cpu":
        cprint("Model loaded on device...", "blue")
        model = model.cuda()
    for index, patch in enumerate(data_loader):
        predict_single_patch(model, patch, index+1, grid_shape, store_directory, device, figure_size)
        if index + 1 == store_num_patches:
            break
