import torch
from ..models.ufanet import UFANet


def load_checkpoint(path, config):
    model = UFANet(config.REAL_CHANNELS, config.MASK_CHANNELS, config.BASE_FILTER_DIM,
                  config.DEPTH, config.USE_ATTN, config.USE_FAM)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    return model
