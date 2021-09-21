import os
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningArgumentParser

import olsen.constants as consts
from olsen.datamod.datamodule import DocumentClassificationData
from olsen.model.bert_slider import BertSlider

import itertools
import pandas as pd

INFER_PATH = consts.DATASET_DIR.joinpath("anthology")
CONFIG_PATH = consts.CONFIG_DIR.joinpath("infer.yaml")
labelled_files = [os.path.splitext(f)[0] + ".test" for f in os.listdir(consts.LABELLED_DATA_DIR.joinpath("anthology"))]
PRED_FILES = [str(INFER_PATH.joinpath(f)) for f in labelled_files if os.path.isfile(INFER_PATH.joinpath(f))]

parser = LightningArgumentParser(parse_as_dict=False, default_config_files=[str(CONFIG_PATH)])

parser.add_lightning_class_args(DocumentClassificationData, "data")
parser.add_lightning_class_args(BertSlider, "model")
parser.add_lightning_class_args(Trainer, "trainer")

parser.add_argument("--exp_name", default="olsen_model")

parser.set_defaults({"data.pred_filenames": PRED_FILES,
                     "trainer.logger": False})

args = parser.parse_args(_skip_check=True)

OUTPUT_PATH = consts.OUTPUT_DIR.joinpath(args.exp_name)

model = BertSlider.load_from_checkpoint(checkpoint_path=consts.MODEL_DIR.joinpath(args.exp_name+".ckpt"))

model_args = model.hparams
dict_args = vars(args.data)
dict_args["window_size"] = model_args["window_size"]
dict_args["dilation_gap"] = model_args["dilation_gap"]
dict_args["transformer"] = model_args["transformer"]
dict_args["labels"] = model_args["labels"]
data = DocumentClassificationData(**dict_args)
trainer = Trainer.from_argparse_args(args.trainer)
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

