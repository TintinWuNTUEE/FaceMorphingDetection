import torch.nn as nn
from torchvision.models import swin_v2_t,Swin_V2_T_Weights
class mySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = swin_v2_t(Swin_V2_T_Weights.DEFAULT)
        self.swin.head.out_features = 1
    def forward(self, x):
        x = self.swin(x)
        return x
def get_model():
    return mySwin()
if __name__ == "__main__":
    model = mySwin()
    print(model)