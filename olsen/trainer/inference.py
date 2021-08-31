import os
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningArgumentParser

import olsen.constants as consts
from olsen.dataload.datamodule import DocumentClassificationData
from olsen.model.bert_slider import BertSlider

INFER_PATH = consts.DATASET_DIR.joinpath("anthology")
CONFIG_PATH = consts.CONFIG_DIR.joinpath("infer.yaml")
PRED_FILES = [INFER_PATH.joinpath(f) for f in os.listdir(INFER_PATH) if os.path.isfile(INFER_PATH.joinpath(f))]

parser = LightningArgumentParser(parse_as_dict=False, default_config_files=[str(CONFIG_PATH)])

parser.add_lightning_class_args(Trainer, "trainer")

parser.add_argument("--exp_name", default="nelsen_model")

parser.set_defaults({"checkpoint.monitor": "val_f1_marco",
                     "checkpoint.dirpath": str(consts.MODEL_DIR),
                     "data.pred_filenames": PRED_FILES,
                     "data.labels": consts.SECT_LABELS})

args = parser.parse_args()

model = BertSlider.load_from_checkpoint(checkpoint_path=consts.MODEL_DIR.joinpath(args.exp_name))

model_args = model.hparams.items()
dict_args = vars(args.data)
dict_args["window_size"] = model_args["window_size"]
dict_args["dilation_size"] = model_args["dilation_size"]

data = DocumentClassificationData(**dict_args)
trainer = Trainer.from_argparse_args(args.trainer)
trainer.predict(model, data)
