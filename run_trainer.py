from core.config import Config
from core.trainer import Trainer

if __name__ == '__main__':
    config = Config('./config/dn4.yaml').get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
