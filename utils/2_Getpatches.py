import numpy as np
import os
import glob
import sys
sys.path.append('../')
from configs.config import get_config

def clean_directory(directory):
    """Remove all .npy files in the specified directory."""
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    existing_files = glob.glob(os.path.join(directory, '*.npy'))
    if existing_files:
        for file in existing_files:
            os.remove(file)
        print(f'Cleared directory: {directory}')
        
def normalize_data(data, method='zscore'):
    """Normalize the data using specified method ('minmax' or 'zscore')."""
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    else:
        raise ValueError("Unsupported normalization method. Use 'minmax' or 'zscore'.")

def extract_patches(data, patch_size, overlap=1):
    """Extract patches from 2D data with zero-padding at edges."""
    nr, nt = data.shape
    patches = []
    
    step = max(1, patch_size // overlap)  # Ensure step is at least 1
    
    # Iterate through all possible starting positions with step size
    for i in range(0, nr, step):
        for j in range(0, nt, step):
            # Calculate the end indices for the current patch
            i_end = i + patch_size
            j_end = j + patch_size
            
            # Create a zero-initialized patch
            patch = np.zeros((patch_size, patch_size), dtype=data.dtype)
            
            # Determine the valid data region to copy
            data_i_start = max(i, 0)
            data_i_end = min(i_end, nr)
            data_j_start = max(j, 0)
            data_j_end = min(j_end, nt)
            
            # Calculate the corresponding region in the patch
            patch_i_start = data_i_start - i
            patch_i_end = patch_i_start + (data_i_end - data_i_start)
            patch_j_start = data_j_start - j
            patch_j_end = patch_j_start + (data_j_end - data_j_start)
            
            # Copy valid data to the patch
            if data_i_start < nr and data_j_start < nt:
                patch[patch_i_start:patch_i_end, patch_j_start:patch_j_end] = \
                    data[data_i_start:data_i_end, data_j_start:data_j_end]
            
            # 先切片再归一化
            # normalize_data(patch)
            patches.append(patch)
    
    return patches
        

# def augment_patch(img_patch, lbl_patch, config):
#     """Apply random flips and noise to a patch."""
#     aug_cfg = config.data.augmentation
    
#     # Random horizontal flip
#     if aug_cfg.get('horizontal_flip', False) and np.random.rand() < aug_cfg.horizontal_flip_prob:
#         img_patch = np.fliplr(img_patch)
#         lbl_patch = np.fliplr(lbl_patch)
    
#     # Random vertical flip
#     if aug_cfg.get('vertical_flip', False) and np.random.rand() < aug_cfg.vertical_flip_prob:
#         img_patch = np.flipud(img_patch)
#         lbl_patch = np.flipud(lbl_patch)
    
#     # Add Gaussian noise (only to image)
#     if aug_cfg.get('add_guassion_noise', False) and np.random.rand() < aug_cfg.guassion_noise_prob:
#         noise_std = np.random.uniform(aug_cfg.guassion_noise_low, aug_cfg.guassion_noise_high)
#         noise = np.random.normal(
#             loc=0.0,
#             scale=noise_std,
#             size=img_patch.shape
#         ).astype(img_patch.dtype)
#         img_patch += noise
        
#         if aug_cfg.get('add_sp_noise', False) and np.random.rand() < aug_cfg.sp_noise_prob:
#             sp_ratio = aug_cfg.get('sp_ratio', 0.01)
#             num_total = img_patch.size
#             num_salt = int(num_total * sp_ratio / 2)
#             num_pepper = int(num_total * sp_ratio / 2)

#             # 随机生成 salt 像素的索引
#             coords_salt = tuple(np.random.randint(0, s, num_salt) for s in img_patch.shape)
#             img_patch[coords_salt] = img_patch.max()

#             # 随机生成 pepper 像素的索引
#             coords_pepper = tuple(np.random.randint(0, s, num_pepper) for s in img_patch.shape)
#             img_patch[coords_pepper] = img_patch.min()
    
    
#     return img_patch, lbl_patch
        

def process_seismic_data(image_path, label_path, save_image_path, save_label_path, config):
    """Load seismic data, extract patches, and save them as .npy files."""
    clean_directory(save_image_path)
    clean_directory(save_label_path)

    # Load seismic data
    image_files = sorted(glob.glob(os.path.join(image_path, '*.npy')))
    label_files = sorted(glob.glob(os.path.join(label_path, '*.npy')))

    if len(image_files) != len(label_files):
        raise ValueError("Mismatch between the number of images and labels")

    for index, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        # Load (nr, nt) data for this shot
        image_data = np.load(image_file)
        label_data = np.load(label_file)

        # Extract patches
        image_patches = extract_patches(image_data, config.data.img_size, config.data.overlap)
        label_patches = extract_patches(label_data, config.data.img_size, config.data.overlap)
        # print(len(image_patches))

        # Save patches
        for i, (img_patch, lbl_patch) in enumerate(zip(image_patches, label_patches)):
            
            patch_index = index * len(image_patches) + i
            
            img_name = os.path.join(save_image_path, f"{patch_index}.npy")
            lbl_name = os.path.join(save_label_path, f"{patch_index}.npy")
            
            np.save(img_name, img_patch)
            np.save(lbl_name, lbl_patch)
        
        print(f'Seismic shot {index} done')


if __name__ == '__main__':
    
    config_path = "/home/wwd/deeplearning/configs/config.yaml"
    config = get_config(config_path)
    
    # Paths
    image_path = config.data.npz_image_path
    label_path = config.data.npz_label_path

    save_image_path = config.data.image_path
    save_label_path = config.data.label_path

    patch_size = config.data.img_size

    # Process seismic data
    process_seismic_data(image_path, label_path, save_image_path, save_label_path, config)
