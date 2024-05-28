from model import get_model
# import torch as th
# if __name__ == "__main__":
#     from config import ConfigFactory
#     args, msg = ConfigFactory.build()
#     model = get_model(args,"cuda")
#     rpt = summary(model, input_size=[th.rand(19),th.rand(150,19)],device="cuda")
#     print(rpt)

import torch as th
from train import Trainer
from config import ConfigFactory
from logger import TrainLoggerFactory
from dataset import get_air_loader_normalizer

if __name__ == "__main__":

    train_logger = TrainLoggerFactory.build()
    train_logger.model_dict_path = "logs/old_logs/25ratio/model_parameter.pth"
    args, msg = ConfigFactory.build()
    train_logger.info(msg)
    train_logger.info(f"Device num: {th.cuda.device_count()}")
    DEVICE = "cuda:0" if th.cuda.is_available() else "cpu"
    train_logger.info(f"Device: {DEVICE}")

    MODEL = get_model(args, DEVICE)
    MODEL.load_state_dict(train_logger.load_model_parameters(),strict=False)
    trainer = Trainer(args, MODEL, DEVICE)

    (
        train_loader,
        val_loader,
        test_loader,
    ), d_normalizer = get_air_loader_normalizer(args)
    loss_dict = trainer.evaluate(MODEL, val_loader, d_normalizer)
    print(loss_dict)
