import torch
import torch.nn as nn
from torchvision.models import swin_v2_s,Swin_V2_S_Weights
class mySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = swin_v2_s(Swin_V2_S_Weights.DEFAULT)
        self.swin.head = nn.Linear(in_features=768, out_features=1, bias=True)
    def forward(self, x):
        x = self.swin(x)
        return x
def get_model():
    return mySwin()
if __name__ == "__main__":
    input = torch.rand((1,3,256,256))
    model = mySwin()
    print(model(input))