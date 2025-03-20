import numpy as np
import os
import glob

def clean_directory(directory):
    """Remove all .npy files in the specified directory."""
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    existing_files = glob.glob(os.path.join(directory, '*.npy'))
    if existing_files:
        for file in existing_files:
            os.remove(file)
        print(f'Cleared directory: {directory}')
        
def normalize_data(data, method='minmax'):
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

def save_data(npz_path, image_path, label_path, image_file, label_file, interval=10, norm_method='zscore', pre=''):
    """Load data from .npz files and save it as .npy files in the specified directories with interval sampling."""
    
    # Load data
    image = np.load(os.path.join(npz_path, image_file))['arr_0']
    label = np.load(os.path.join(npz_path, label_file))['arr_0']

    image = normalize_data(image, method=norm_method)
    label = normalize_data(label, method=norm_method)
    
    ns = image.shape[0]
    
    # Save data at specified intervals
    for index in range(0, ns, interval):  # Step size is 'interval' (default is 5)
        np.save(os.path.join(image_path, f'{pre}_free_{index}.npy'), image[index, :, :])
        np.save(os.path.join(label_path, f'{pre}_damp_{index}.npy'), label[index, :, :])
        print(f'{pre} seismic shot {index} saved')

if __name__ == "__main__":
    # Define save paths
    npz_path = '../dataset/'
    image_path = '../dataset/image/'
    label_path = '../dataset/label/'
    
    # Clean existing files in the directories
    clean_directory(image_path)
    clean_directory(label_path)

    # ----------------------------------syn1 data----------------------------------
    image_file1 = 'data_free_syn1.npz'
    label_file1 = 'data_damp_syn1.npz'
    save_data(npz_path, image_path, label_path, image_file1, label_file1, interval=20, norm_method='zscore', pre='syn1')
    # ----------------------------------syn1 data----------------------------------
    
    
    # ----------------------------------syn2 data----------------------------------
    image_file2 = 'data_free_syn2.npz'
    label_file2 = 'data_damp_syn2.npz'
    save_data(npz_path, image_path, label_path, image_file2, label_file2, interval=20, norm_method='zscore', pre='syn2')
    # ----------------------------------syn2 data----------------------------------
    
    
    # ----------------------------------field1 data----------------------------------
    image_file3 = 'SEAM_data.npz'
    label_file3 = 'SEAM_primaries.npz'
    save_data(npz_path, image_path, label_path, image_file3, label_file3, interval=5, norm_method='zscore', pre='field1')
    # ----------------------------------field1 data----------------------------------
    
    
    
     
