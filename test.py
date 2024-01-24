from torchsummary import summary
from model import get_model
import torch as th
if __name__ == "__main__":
    from config import ConfigFactory
    args, msg = ConfigFactory.build()
    model = get_model(args,"cuda")
    rpt = summary(model, input_size=[th.rand(19),th.rand(150,19)],device="cuda")
    print(rpt)
