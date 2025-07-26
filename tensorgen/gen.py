import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file
import numpy as np


class C(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = nn.Conv2d(3, 256, 3, stride=1, padding=0, dilation=1)
        self.c2 = nn.Conv2d(256, 256, 1, stride=1, padding=0, dilation=1, groups=64)
        self.c3 = nn.Conv2d(256, 10, 5, stride=3, padding=2, dilation=1, groups=2, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        c3_out = self.c3(c2_out)
        c3_gelu_out = self.gelu(c3_out)
        in_out_tensors = {
            "x": x,
            "c1_out":c1_out,
            "c2_out":c2_out,
            "c3_out":c3_out,
            "c3_gelu_out":c3_gelu_out,
        }
        save_file(in_out_tensors, "safetensors/in_out_tensors.safetensors")
        return c3_gelu_out

img = Image.open("tewelu.jpg")
x = (torch.tensor(np.array(img)[None,:].transpose(0,3,1,2)).contiguous() / 255.0).float()
model = C().float()
out = model(x)
save_file(model.state_dict(), "safetensors/model1.safetensors")

# print(model.state_dict())
print(model.c1.weight)
