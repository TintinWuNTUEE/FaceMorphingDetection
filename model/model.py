import os
import torch
import torch.nn as nn
from torchvision.models import swin_v2_s, Swin_V2_S_Weights


class mySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = swin_v2_s(Swin_V2_S_Weights.DEFAULT)
        self.swin.head = nn.Linear(in_features=768, out_features=1, bias=True)

    def forward(self, x):
        x = self.swin(x)
        return x


class ensemble_swin(nn.Module):
    def __init__(self, path, device, kf):
        super().__init__()
        self.swins = [mySwin() for i in range(self.kf)]
        self.path = path
        self.load_model()
        self.device = device
        self.kf = kf

    def predict(self, i, x):
        self.swins[i].to(self.device)
        output = self.swins[i](x)
        self.swins[i].cpu()
        return output

    def forward(self, x):
        output = torch.tensor([self.predict(i, x) for i in range(self.kf)])
        output = torch.mean(output)
        return output

    def load_model(self):
        if not os.path.isdir(self.path):
            return
        for i, ckpt_path in enumerate(os.listdir(self.path)):
            self.swins[i].load_state_dict(
                torch.load(os.path.join(self.path, ckpt_path)["model"])
            )


def get_ensemble():
    return ensemble_swin()


def get_model():
    return mySwin()


if __name__ == "__main__":
    input = torch.rand((1, 3, 256, 256))
    model = ensemble_swin("", device="cpu")
    print(model(input))
