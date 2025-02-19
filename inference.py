import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    writer = None
    if "writer" in config:
        project_config = OmegaConf.to_container(config)
        writer = instantiate(config.writer, None, project_config)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    print(model)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )

    # save_path for model predictions
    save_path = None
    if config.inferencer.save_path is not None:
        save_path = Path(config.inferencer.save_path)
        save_path.mkdir(exist_ok=True, parents=True)

    log_step = config.inferencer.log_step if "writer" in config else None

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        writer=writer,
        metrics=metrics,
        skip_model_load=False,
        log_step=log_step,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
