from termcolor import cprint
from ..training.predictor import predict
from ..utils.training import create_data_loader
from ..config.test import defaults as CFG
from ..training.load_checkpoint import load_checkpoint


def run_scheduler(config):
    data_loader = create_data_loader(config=config, shuffle=config.SHUFFLE, drop_last=config.DROP_LAST, evaluate=True)
    cprint("Data Loader created...", "blue")
    model = load_checkpoint(path=config.CHECKPOINT_PATH, config=config)
    cprint("model loaded from checkpoint path...", "blue")
    predict(model=model,
            data_loader=data_loader,
            store_num_patches=config.NUM_PATCHES,
            grid_shape=config.GRID_SHAPE,
            store_directory=config.STORE_RESULTS_DIRECTORY,
            device=config.DEVICE,
            figure_size=config.FIG_SHAPE)


if __name__ == '__main__':
    run_scheduler(CFG)
