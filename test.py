import os

import typer
from comer.datamodule import FintamathDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def main():
    # generate output latex in result.zip
    ckp_path = f"{os.path.dirname(os.path.realpath(__file__))}/pretrained/pretrained.ckpt"
    print(f"Checkpoint: {ckp_path}")

    trainer = Trainer(logger=False, gpus=1)

    dm = FintamathDatamodule(eval_batch_size=4)

    model = LitCoMER.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
