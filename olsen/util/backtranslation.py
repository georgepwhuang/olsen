from olsen.util.uda_util import BackTranslator
from olsen.constants import DATASET_DIR
import os


source = DATASET_DIR.joinpath("uda", "original")
target = DATASET_DIR.joinpath("uda", "translated")
input_files = os.listdir(source)[:2]
translator = BackTranslator(device=3)
for file in input_files:
    with open(source.joinpath(file)) as f:
        original_lines = f.readlines()
        backtran_lines = translator.back_translate(original_lines)
    result = [oline + "###" + bline for oline, bline in zip(original_lines, backtran_lines)]
    result = "\n".join(result)
    with open(target.joinpath(file), 'w') as f:
        f.write(result)
        f.write("\n")
