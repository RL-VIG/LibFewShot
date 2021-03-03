from core.config import Config
from core.trainer import Trainer

if __name__ == '__main__':
    config = Config('./config/r2d2.yaml').get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
