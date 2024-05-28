import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset import get_air_loader_normalizer
from model import get_model, predict


class Trainer:

    def __init__(self, args, model, device) -> None:
        self.args = args
        self.model = model
        self.best_model = None
        self.device = device

        self.metric_calculator = {
            "mae": lambda error: np.abs(error).mean(),
            "mape": lambda error, label:
            (np.abs(error) / np.abs(label)).mean(),
            "mse": lambda error: np.square(error).mean(),
        }
        self.main_weight = max(args.mask_ratio, 1 - args.mask_ratio)

    @staticmethod
    def flat_dict(my_dict):
        return [
            val for pair in zip(my_dict.keys(), my_dict.values())
            for val in pair
        ]

    @staticmethod
    def dict2str(loss):
        return pd.DataFrame.from_dict(loss,
                                      orient="index").T.to_string(index=False)

    def get_loss(self, args, batch_data, model):
        loss_fn = F.mse_loss
        data_recover, curl_loss, query_label, _, mask = predict(
            batch_data, model, self.device, mm_size=args.mm_size)
        with autocast():
            loss_mask = loss_fn(data_recover[mask], query_label[mask])
            loss_non_mask = loss_fn(data_recover[~mask], query_label[~mask])
            loss = loss_mask * self.main_weight + loss_non_mask * (
                1 - self.main_weight)
            # curl_loss = curl_loss/10000
            # main_loss = th.matmul(th.stack([loss, curl_loss]), th.reciprocal(
            #     2*w.square())) + th.cumprod(w, dim=0)[-1].log()
            main_loss = loss
            main_loss = (main_loss) / args.acc_steps
        return main_loss, loss_mask, loss_non_mask, curl_loss

    def train_an_epoch(self, args, model, loop, optimizer):
        scaler = GradScaler()
        loop_len = len(loop)
        running_step = 0
        loss_dict = {"main": 0.0, "mask": 0.0, "non_mask": 0.0, "curl": 0.0}
        for batch_data in loop:
            main_loss, loss_mask, loss_non_mask, curl_loss = self.get_loss(
                args, batch_data, model)
            scaler.scale(main_loss).backward()
            running_step += 1
            if (running_step % args.acc_steps == 0) or (running_step
                                                        == loop_len):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            loop.set_postfix({'loss': "{:.4f}".format(main_loss.item())},
                             refresh=True)
            loss_dict["main"] += main_loss.item() / loop_len
            loss_dict["mask"] += loss_mask.item() / loop_len
            loss_dict["non_mask"] += loss_non_mask.item() / loop_len
            loss_dict["curl"] += curl_loss.item() / loop_len

        loss_dict["main"] = loss_dict["main"] * args.acc_steps

        return loss_dict

    def evaluate(self, model, loader, data_normalizer):
        model = model.eval()
        loss_dict = {"mae": 0.0, "mape": 0.0, "rmse": 0.0}
        with th.no_grad():
            for batch_data in tqdm(loader):
                data_recover, _, query_label, data_index, mask = predict(
                    batch_data, model, self.device, mm_size=0, training=False)
                pred_label = data_normalizer.denorm(
                    data_recover.cpu().detach().numpy(), data_index)
                label = data_normalizer.denorm(
                    query_label.cpu().detach().numpy(), data_index)
                error = (pred_label - label)[mask]
                scale_index = 1 / len(loader)
                loss_dict["mae"] += np.abs(error).mean() * scale_index
                loss_dict["mape"] += (np.abs(error) /
                                      np.abs(label[mask])).mean() * scale_index
                loss_dict["rmse"] += np.sqrt(
                    np.square(error).mean()) * scale_index
        return loss_dict

    def train(self, logger):
        args = self.args
        (
            train_loader,
            val_loader,
            test_loader,
        ), d_normalizer = get_air_loader_normalizer(args)

        model = self.model.to(self.device)
        optimizer = Adam(model.parameters(),
                         lr=args.lr,
                         weight_decay=args.w_decay)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

        best_loss = th.inf
        logger.info("Starting the training process...")
        for epoch in range(args.epochs):
            # train
            model = model.train()
            logger.info(f"Epoch {epoch}.")
            optimizer.zero_grad()
            all_loss = {}
            train_loss = self.train_an_epoch(args, model,
                                             tqdm(train_loader, desc='Train'),
                                             optimizer)
            all_loss["Train loss"] = self.dict2str(train_loss)
            # validation
            if epoch >= 35:
                val_loss = self.evaluate(model, val_loader, d_normalizer)
                all_loss["Val loss"] = self.dict2str(val_loss)
                record_criteria = val_loss["mae"]
                if record_criteria < best_loss:
                    best_loss = record_criteria
                    logger.info("Save Model Parameters...")
                    logger.save_model_parameters(model)
            logger.info("\n" + "\n".join(self.flat_dict(all_loss)))
            scheduler.step()
        logger.info("Training process completed.")
        model.load_state_dict(logger.load_model_parameters())
        test_loss = self.dict2str(
            self.evaluate(model, test_loader, d_normalizer))
        logger.info("\nTest loss\n" + test_loss)


if __name__ == "__main__":
    from config import ConfigFactory
    from logger import TrainLoggerFactory

    train_logger = TrainLoggerFactory.build()
    args, msg = ConfigFactory.build()
    train_logger.info(msg)
    train_logger.info(f"Device num: {th.cuda.device_count()}")
    DEVICE = "cuda:3" if th.cuda.is_available() else "cpu"
    train_logger.info(f"Device: {DEVICE}")

    MODEL = get_model(args, DEVICE)
    trainer = Trainer(args, MODEL, DEVICE)
    trainer.train(train_logger)
