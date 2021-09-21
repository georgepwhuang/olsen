from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from olsen.datamod.datamodule import DocumentClassificationData
from olsen.model.bert_slider import BertSlider
from pytorch_lightning import Trainer
import olsen.constants as consts
import os
import warnings
import itertools
import pandas as pd
import json
warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_PATH = consts.DATASET_DIR.joinpath("sect_label")
INFER_PATH = consts.DATASET_DIR.joinpath("anthology")
CONFIG_PATH = consts.CONFIG_DIR.joinpath("train.yaml")
labelled_files = [os.path.splitext(f)[0] + ".test" for f in os.listdir(consts.LABELLED_DATA_DIR.joinpath("anthology"))]
PRED_FILES = [str(INFER_PATH.joinpath(f)) for f in labelled_files if os.path.isfile(INFER_PATH.joinpath(f))]

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
                     "data.pred_filenames": PRED_FILES,
                     "model.labels": consts.SECT_LABELS,
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
result = trainer.predict(model, data)
flattened_result = []
for i in range(len(PRED_FILES)):
    labels = list(itertools.chain.from_iterable(result[i]))
    lines = []
    with open(PRED_FILES[i]) as fp:
        for line in fp:
            if bool(line.strip()):
                line = line.strip()
                lines.append(line)
    df = pd.DataFrame()
    df["Lines"] = lines
    df["Label"] = labels
    df.to_csv(os.path.join(OUTPUT_PATH, os.path.splitext(os.path.basename(PRED_FILES[i]))[0] + ".csv"), index=False)

