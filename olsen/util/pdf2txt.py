import os.path
import subprocess

import olsen.constants as const

source = const.RAW_DATA_DIR.joinpath("unlabelled")
target = const.DATASET_DIR.joinpath("unlabelled")
files = os.listdir(source)
PDF_BOX_JAR = const.BIN_DIR.joinpath("pdf2txt/pdfbox-app-2.0.24.jar")
for filepath in files:
    if ".pdf" not in str(filepath):
        continue
    writefile = os.path.splitext(filepath)[0]
    text = subprocess.run(
        ["java", "-jar", PDF_BOX_JAR, "ExtractText", source.joinpath(filepath), target.joinpath(writefile + ".txt")],
    )
