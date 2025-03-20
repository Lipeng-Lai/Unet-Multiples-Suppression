import numpy as np
import os
import glob

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
            
            patches.append(patch)
    
    return patches

def clean_directory(directory):
    """Remove all .npy files in the specified directory."""
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    existing_files = glob.glob(os.path.join(directory, '*.npy'))
    if existing_files:
        for file in existing_files:
            os.remove(file)
        print(f'Cleared directory: {directory}')
        

def process_seismic_data(image_path, label_path, save_image_path, save_label_path, patch_size):
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
        image_patches = extract_patches(image_data, patch_size)
        label_patches = extract_patches(label_data, patch_size)
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
    # Paths
    image_path = '../dataset/image/'
    label_path = '../dataset/label/'

    save_image_path = '../data/image/'
    save_label_path = '../data/label/'

    patch_size = 256

    # Process seismic data
    process_seismic_data(image_path, label_path, save_image_path, save_label_path, patch_size)
