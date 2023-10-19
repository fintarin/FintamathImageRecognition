import torch

import onnx

from comer.lit_comer import LitCoMER

ckpt_path = "pretrained/pretrained.ckpt"
onnx_path = "pretrained/pretrained.onnx"
pb_path = "pretrained/pretrained.pb"

if __name__ == "__main__":
    ckpt_model = LitCoMER.load_from_checkpoint(ckpt_path)

    batch_size = 8
    height, width = 256, 256
    img = torch.randn(batch_size, 1, height, width)
    img_mask = torch.randint(0, 2, (batch_size, height, width), dtype=torch.bool)
    tgt = torch.randint(0, 2, (batch_size * 2, height), dtype=torch.long)
    
    torch.onnx.export(ckpt_model, (img, img_mask, tgt), onnx, verbose=True)
