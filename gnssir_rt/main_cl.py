from datetime import datetime

import hydra
from omegaconf import DictConfig

from gnssir_rt.processing import arcs2splines, arcsplot, snr2arcs


@hydra.main(config_path="configs/", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print(f"site is {cfg.site_name}")

    pyargs = load_cfg(cfg)

    if cfg.run == "snr2arcs":
        snr2arcs(**pyargs)
    elif cfg.run == "arcsplot":
        arcsplot(**pyargs)
    elif cfg.run == "arcs2splines":
        arcs2splines(**pyargs)
    else:
        print(f"input function '{cfg.run}' not recognized ")
        print("e.g., set run=snr2arcs from command line")


def load_cfg(cfg):
    pyargs = dict(cfg)
    for dt in ["sdt", "edt"]:
        pyargs[dt] = datetime.strptime(pyargs[dt], "%y-%m-%d %H:%M")
    return pyargs


if __name__ == "__main__":
    main()
