from datetime import datetime
from typing import Dict, Any

import hydra
from omegaconf import DictConfig

from gnssir.processing import arcs2splines, arcsplot, snr2arcs


@hydra.main(config_path="configs/", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for GNSS-IR processing tasks.

    This function handles different processing tasks based on the configuration:
    - snr2arcs: Convert SNR data to arcs
    - arcsplot: Plot the arcs
    - arcs2splines: Convert arcs to splines

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing processing parameters

    Raises
    ------
    ValueError
        If the specified task is not recognized
    """
    print(f"site is {cfg.site_name}")

    pyargs = load_cfg(cfg)

    if cfg.task == "snr2arcs":
        snr2arcs(**pyargs)
    elif cfg.task == "arcsplot":
        arcsplot(**pyargs)
    elif cfg.task == "arcs2splines":
        arcs2splines(**pyargs)
    else:
        print(f"input function '{cfg.task}' not recognized ")
        print("e.g., set task=snr2arcs from command line")


def load_cfg(cfg: DictConfig) -> Dict[str, Any]:
    """
    Load and process configuration parameters.

    Converts date strings in the configuration to datetime objects.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    -------
    Dict[str, Any]
        Processed configuration dictionary with datetime objects
    """
    pyargs = dict(cfg)
    for dt in ["sdt", "edt"]:
        pyargs[dt] = datetime.strptime(pyargs[dt], "%y-%m-%d %H:%M")
    return pyargs


if __name__ == "__main__":
    main()
