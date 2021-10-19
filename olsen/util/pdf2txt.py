import os.path
import subprocess

import olsen.constants as const

source = const.RAW_DATA_DIR.joinpath("uda")
target = const.DATASET_DIR.joinpath("uda")
files = os.listdir(source)
PDF_BOX_JAR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pdfbox-app-2.0.24.jar")
for filepath in files:
    if ".pdf" not in str(filepath):
        continue
    writefile = os.path.splitext(filepath)[0]
    text = subprocess.run(
        ["java", "-jar", PDF_BOX_JAR, "ExtractText", source.joinpath(filepath), target.joinpath(writefile+".txt")],
    )
