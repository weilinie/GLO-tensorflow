import os
from GLO_model import GLO_model
from config import get_config
from utils import prepare_dirs


config, unparsed = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
prepare_dirs(config, config.dataset)

glo_model = GLO_model(config)

if config.is_train:
    glo_model.train()

else:
    if not config.load_path:
        raise Exception("[!] You should specify `load_path` to load a pretrained model")
    glo_model.test()