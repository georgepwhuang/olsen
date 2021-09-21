from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from olsen.datamod.datamodule import DocumentClassificationData
from olsen.model.bert_slider import BertSlider
from pytorch_lightning import Trainer
import olsen.constants as consts
import json

TRAIN_PATH = consts.DATASET_DIR.joinpath("sect_label")
CONFIG_PATH = consts.CONFIG_DIR.joinpath("train.yaml")

parser = LightningArgumentParser(parse_as_dict=False, default_config_files=[str(CONFIG_PATH)])

parser.add_lightning_class_args(DocumentClassificationData, "data")
parser.add_lightning_class_args(BertSlider, "model")
parser.add_lightning_class_args(Trainer, "trainer")
parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

parser.add_argument("--exp_name", default="olsen_model")
parser.link_arguments("model.window_size", "data.window_size")
parser.link_arguments("model.dilation_gap", "data.dilation_gap")
parser.link_arguments("model.transformer", "data.transformer")
parser.link_arguments("model.labels", "data.labels")
parser.link_arguments("exp_name", "checkpoint.filename")

parser.set_defaults({"checkpoint.dirpath": str(consts.MODEL_DIR),
                     "data.train_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.train")),
                     "data.dev_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.dev")),
                     "data.test_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.test")),
                     "model.labels": consts.SECT_LABELS,
                     "model.num_classes": len(consts.SECT_LABELS),
                     "trainer.default_root_dir": str(consts.LOG_DIR)})

args = parser.parse_args()

OUTPUT_PATH = consts.OUTPUT_DIR.joinpath(args.exp_name)

data = DocumentClassificationData(**vars(args.data))
model = BertSlider(**vars(args.model))
checkpoint_callback = ModelCheckpoint(**vars(args.checkpoint))
trainer = Trainer.from_argparse_args(args.trainer, callbacks=[checkpoint_callback])
trainer.fit(model, data)
test_metrics = trainer.test(model, data)
with open(str(OUTPUT_PATH)+".json", "w") as outfile:
    json.dump(test_metrics[0], outfile)
