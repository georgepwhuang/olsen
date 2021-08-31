from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from olsen.dataload.datamodule import DocumentClassificationData
from olsen.model.bert_slider import BertSlider
from pytorch_lightning import Trainer
import olsen.constants as consts
import os

TRAIN_PATH = consts.DATASET_DIR.joinpath("sect_label")
INFER_PATH = consts.DATASET_DIR.joinpath("anthology")
CONFIG_PATH = consts.CONFIG_DIR.joinpath("train.yaml")
PRED_FILES = [str(INFER_PATH.joinpath(f)) for f in os.listdir(INFER_PATH) if os.path.isfile(INFER_PATH.joinpath(f))]

parser = LightningArgumentParser(parse_as_dict=False, default_config_files=[str(CONFIG_PATH)])

parser.add_lightning_class_args(DocumentClassificationData, "data")
parser.add_lightning_class_args(BertSlider, "model")
parser.add_lightning_class_args(Trainer, "trainer")
parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

parser.add_argument("--exp_name", default="nelsen_model")
parser.link_arguments("data.window_size", "model.window_size")
parser.link_arguments("data.dilation_gap", "model.dilation_gap")
parser.link_arguments("data.labels", "model.labels")
parser.link_arguments("exp_name", "checkpoint.filename")

parser.set_defaults({"checkpoint.monitor": "val_f1_marco",
                     "checkpoint.dirpath": str(consts.MODEL_DIR),
                     "data.train_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.train")),
                     "data.dev_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.dev")),
                     "data.test_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.test")),
                     "data.pred_filenames": PRED_FILES,
                     "data.labels": consts.SECT_LABELS})

args = parser.parse_args()

data = DocumentClassificationData(**vars(args.data))
model = BertSlider(**vars(args.model))
checkpoint_callback = ModelCheckpoint(**vars(args.checkpoint))
trainer = Trainer.from_argparse_args(args.trainer, callbacks=[checkpoint_callback])
trainer.tune(model, data)
trainer.fit(model, data)
trainer.predict(model, data)
