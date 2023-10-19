from pytorch_lightning.cli import LightningCLI

from comer.datamodule import FintamathDatamodule
from comer.lit_comer import LitCoMER

if __name__ == "__main__":
    cli = LightningCLI(
        LitCoMER,
        FintamathDatamodule,
        save_config_overwrite=True,
    )
