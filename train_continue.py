from pytorch_lightning.cli import LightningCLI

from comer.datamodule import FintamathDatamodule
from comer.lit_comer import LitCoMER

ckpt = "pretrained/pretrained.ckpt"

if __name__ == "__main__":
    cli = LightningCLI(
        LitCoMER,
        FintamathDatamodule,
        save_config_overwrite=True,
        run=False,
    )
    cli.trainer.fit(
        cli.model.load_from_checkpoint(ckpt), datamodule=cli.datamodule, ckpt_path=ckpt
    )
