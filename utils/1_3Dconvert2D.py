import numpy as np
import os
import glob
import sys
sys.path.append('../')
from utils.Read_SEGY import ReadSEGYData
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

def save_data(npz_path, image_path, label_path, image_file, label_file, interval=10, norm_method='zscore', pre=''):
    """Load data from .npz files and save it as .npy files in the specified directories with interval sampling."""
    
    # Load data
    if pre == 'field3':
        image = np.load(os.path.join(npz_path, image_file))['p'].transpose(0, 2, 1)
        label = np.load(os.path.join(npz_path, label_file))['p'].transpose(0, 2, 1)
    elif pre == 'field5':
        image = ReadSEGYData(os.path.join(npz_path, image_file))
        label = ReadSEGYData(os.path.join(npz_path, label_file))
        np.save(os.path.join(image_path, f'{pre}_free.npy'), image)
        np.save(os.path.join(label_path, f'{pre}_damp.npy'), label)
        return
    else:
        image = np.load(os.path.join(npz_path, image_file))['arr_0']
        label = np.load(os.path.join(npz_path, label_file))['arr_0']

    # 归一化
    # image = normalize_data(image, method=norm_method)
    # label = normalize_data(label, method=norm_method)
    
    ns = image.shape[0]
    
    # Save data at specified intervals
    for index in range(0, ns, interval):
        np.save(os.path.join(image_path, f'{pre}_free_{index}.npy'), image[index, :, :])
        np.save(os.path.join(label_path, f'{pre}_damp_{index}.npy'), label[index, :, :])
        print(f'{pre} seismic shot {index} saved')

if __name__ == "__main__":
    config_path = "/home/wwd/deeplearning/configs/config.yaml"
    config = get_config(config_path)
    # npz_path = '../dataset/'
    npz_path = config.data.npz_path
    image_path = config.data.npz_image_path
    label_path = config.data.npz_label_path
    
    # Clean existing files in the directories
    clean_directory(image_path)
    clean_directory(label_path)

    # ----------------------------------syn1 data----------------------------------
    # image_file1 = 'syn_data_modify.npz'
    # label_file1 = 'syn_primaries.npz'
    # save_data(npz_path, image_path, label_path, image_file1, label_file1, config.data.syn_shot_interval, norm_method='zscore', pre='syn1')
    # ----------------------------------syn1 data----------------------------------
    
    
    # ----------------------------------syn2 data----------------------------------
    # image_file2 = 'data_free_syn2.npz'
    # label_file2 = 'data_damp_syn2.npz'
    # save_data(npz_path, image_path, label_path, image_file2, label_file2, config.data.syn_shot_interval, norm_method='zscore', pre='syn2')
    # ----------------------------------syn2 data----------------------------------
    
    
    # ----------------------------------field1 data----------------------------------
    # image_file3 = 'SEAM1_data.npz'
    # label_file3 = 'SEAM1_primaries.npz'
    # save_data(npz_path, image_path, label_path, image_file3, label_file3, config.data.SEAM_shot_interval, norm_method='zscore', pre='field1')
    # ----------------------------------field1 data----------------------------------
    
    # ----------------------------------field2 data----------------------------------
    # image_file4 = 'SEAM2_data.npz'
    # label_file4 = 'SEAM2_primaries.npz'
    # save_data(npz_path, image_path, label_path, image_file4, label_file4, config.data.SEAM_shot_interval, norm_method='zscore', pre='field2')
    # ----------------------------------field2 data----------------------------------
    
    # ----------------------------------field3 data----------------------------------
    # image_file5 = 'input_full_volvesynth.npz'
    # label_file5 = 'input_nofs_full_volvesynth.npz'
    # save_data(npz_path, image_path, label_path, image_file5, label_file5, config.data.Volve_shot_interval, norm_method='zscore', pre='field3')
    # ----------------------------------field3 data----------------------------------
    
    # ----------------------------------field4 data----------------------------------
    # image_file6 = 'Sigsbee_fs.npz'
    # label_file6 = 'Sigsbee_nfs.npz'
    # save_data(npz_path, image_path, label_path, image_file6, label_file6, config.data.Sigsbee_shot_interval, norm_method='zscore', pre='field4')
    # ----------------------------------field4 data----------------------------------
    
    # ----------------------------------field5 data----------------------------------
    image_file7 = 'Sigsbee2B_ZeroOffset_fs.sgy'
    label_file7 = 'Sigsbee2B_ZeroOffset_nfs.sgy'
    save_data(npz_path, image_path, label_path, image_file7, label_file7, 1, norm_method='zscore', pre='field5')
    # ----------------------------------field5 data----------------------------------
    
    
