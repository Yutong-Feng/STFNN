import logging
import os
from datetime import datetime
from torch import save, load


class TrainLogger(logging.Logger):
    def __init__(self, name: str, level, model_dict_path) -> None:
        super().__init__(name, level)
        self.model_dict_path = model_dict_path

    def save_model_parameters(self, model):
        save(model.state_dict(), self.model_dict_path)

    def load_model_parameters(self):
        return load(self.model_dict_path)


class TrainLoggerFactory:
    LOG_FOLDER = "logs"
    LOG_FILE = "training.log"
    MODEL_FILE = "model_parameter.pth"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    @classmethod
    def get_file_path(cls):
        file_path = os.path.join(
            cls.LOG_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        return file_path

    @classmethod
    def build_handler(cls, handler_type, log_file_path):
        formatter = logging.Formatter(cls.LOG_FORMAT, datefmt=cls.DATE_FORMAT)
        handler = {
            "console": logging.StreamHandler(),
            "file": logging.FileHandler(log_file_path),
        }[handler_type]

        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def build(cls) -> logging.Logger:
        file_path = cls.get_file_path()
        log_path = os.path.join(file_path, cls.LOG_FILE)
        model_path = os.path.join(
            file_path, cls.MODEL_FILE)
        
        logger = TrainLogger("training_logger", logging.INFO, model_path)
        logger.addHandler(cls.build_handler("console", log_path))
        logger.addHandler(cls.build_handler("file", log_path))
        return logger


if __name__ == "__main__":
    temp_logger = TrainLoggerFactory.build()
    temp_logger.info("Starting the training process...")
    temp_logger.info("Training process completed.")
