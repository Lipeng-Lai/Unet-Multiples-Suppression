from .dncnn import DnCNN
from .hrnet import HRNet2D
from .resunet import ResUNet2D
from .unet import UNet2D
from .unetpp import Unetpp
from .transunet import TransUNet
import sys
sys.path.append('..')
from configs.config import get_config

def build_model(config):
    model_type = config.model.name
    if model_type == "DnCNN":
        model = DnCNN(
            in_channels=config.model.in_chans,
        )
    elif model_type == "HRNet2D":
        model = HRNet2D(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            base_channel=16,
        )
    elif model_type == "ResUNet2D":
        model = ResUNet2D(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
        )
    elif model_type == "UNet2D":
        model = UNet2D(
            in_channels=config.model.in_chans,
        )
    elif model_type == "Unetpp":
        model = Unetpp(
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
        )
    elif model_type == "TransUNet":
        model = TransUNet()
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    return model

if __name__ == '__main__':
    config_path = "/home/wwd/deeplearning/configs/config.yaml"
    config = get_config(config_path)
    build_model(config)
