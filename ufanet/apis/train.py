from torch.optim import Adam
from torch.nn import MSELoss
from ..models import UFANet
from ..config.train import defaults as CFG
from ..training.trainer import train_model
from ..utils.training import create_data_loader, fetch_metric_dict


def run_scheduler(config):
    train_dl = create_data_loader(config, True, config.SHUFFLE, config.DROP_LAST)
    val_dl = create_data_loader(config, False, config.SHUFFLE, config.DROP_LAST)
    model = UFANet(config.REAL_CHANNELS, config.MASK_CHANNELS,
                  config.BASE_FILTER_DIM, config.DEPTH, config.USE_ATTN, config.USE_FAM)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    loss_fn = MSELoss()
    metric_dict = fetch_metric_dict()
    train_model(model=model,
                train_loader=train_dl,
                val_loader=val_dl,
                num_epochs=config.NUM_EPOCHS,
                loss_fn=loss_fn,
                optim=optimizer,
                metrics=metric_dict,
                device=config.DEVICE,
                track_result_counter=config.TRACK_RESULT_COUNTER,
                store_result_dir=config.RESULT_WAREHOUSE_DIRECTORY,
                train_kill_thresh=config.TRAIN_KILL_THRESH,
                figure_shape=config.RESULT_FIGURE_SHAPE,
                segm_threshold=config.SEGM_THRESH)


if __name__ == "__main__":
    run_scheduler(CFG)
