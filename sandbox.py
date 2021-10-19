from olsen.constants import DATASET_DIR

SOURCE = DATASET_DIR.joinpath("sect_label", "sectLabelDoc.train")
TARGET = DATASET_DIR.joinpath("sect_label", "sectLabel.text")

with open(SOURCE) as f:
    lines = f.readlines()

with open(TARGET, "w") as f:
    for line in lines:
        if bool(line.strip()):
            f.write(line.strip().split("###")[0])
            f.write("\n")
        else:
            f.write("\n")