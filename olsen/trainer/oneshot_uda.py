import itertools
import json
import os
import warnings

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.plugins import DeepSpeedPlugin

import olsen.constants as consts
from olsen.datamod.datamodule import DocumentClassificationDataForSSL
from olsen.model.bert_slider_with_uda import BertSliderWithUDA

warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_PATH = consts.DATASET_DIR.joinpath("sect_label")
UNSUP_PATH = consts.DATASET_DIR.joinpath("unlabelled")
INFER_PATH = consts.DATASET_DIR.joinpath("anthology")
CONFIG_PATH = consts.CONFIG_DIR.joinpath("train.yaml")
labelled_files = [os.path.splitext(f)[0] + ".test" for f in os.listdir(consts.LABELLED_DATA_DIR.joinpath("anthology"))]
PRED_FILES = [str(INFER_PATH.joinpath(f)) for f in labelled_files if os.path.isfile(INFER_PATH.joinpath(f))]

parser = LightningArgumentParser(parse_as_dict=False, default_config_files=[str(CONFIG_PATH)])

parser.add_lightning_class_args(DocumentClassificationDataForSSL, "data")
parser.add_lightning_class_args(BertSliderWithUDA, "model")
parser.add_lightning_class_args(Trainer, "trainer")
parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
parser.add_class_arguments(DeepSpeedPlugin, "strategy")

parser.add_argument("--exp_name", default="olsen_model")
parser.link_arguments("model.window_size", "data.window_size")
parser.link_arguments("model.dilation_gap", "data.dilation_gap")
parser.link_arguments("model.transformer", "data.transformer")
parser.link_arguments("model.labels", "data.labels")
parser.link_arguments("exp_name", "checkpoint.filename")
parser.link_arguments("data.batch_size", "strategy.logging_batch_size_per_gpu")

parser.set_defaults({"checkpoint.dirpath": str(consts.MODEL_DIR),
                     "data.train_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.train")),
                     "data.dev_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.dev")),
                     "data.test_filename": str(TRAIN_PATH.joinpath("sectLabelDoc.test")),
                     "data.pred_filenames": PRED_FILES,
                     "data.unlabelled_dirname": str(UNSUP_PATH.joinpath("translated")),
                     "model.labels": consts.SECT_LABELS,
                     "model.num_classes": len(consts.SECT_LABELS),
                     "trainer.default_root_dir": str(consts.LOG_DIR)})

args = parser.parse_args()

OUTPUT_PATH = consts.OUTPUT_DIR.joinpath(args.exp_name)

data = DocumentClassificationDataForSSL(**vars(args.data))
strategy = DeepSpeedPlugin(**vars(args.strategy))
model = BertSliderWithUDA(**vars(args.model))
checkpoint_callback = ModelCheckpoint(**vars(args.checkpoint))
trainer = Trainer.from_argparse_args(args.trainer, callbacks=[checkpoint_callback], strategy=strategy,
                                     accumulate_grad_batches=sum(args.data.data_ratio))
trainer.fit(model, data)
'''
test_metrics = trainer.test(model, data)
with open(str(OUTPUT_PATH) + ".json", "w") as outfile:
    json.dump(test_metrics[0], outfile)
result = trainer.predict(model, data)
flattened_result = []
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
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
'''