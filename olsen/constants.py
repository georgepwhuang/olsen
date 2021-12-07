import os
import pathlib

CODE_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = pathlib.Path(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = ROOT_DIR.joinpath("data")
RAW_DATA_DIR = DATA_DIR.joinpath("raw_data")
LABELLED_DATA_DIR = DATA_DIR.joinpath("labelled_data")
DATASET_DIR = DATA_DIR.joinpath("dataset")

CONFIG_DIR = ROOT_DIR.joinpath("config")
LOG_DIR = ROOT_DIR.joinpath("logs")
MODEL_DIR = ROOT_DIR.joinpath("models")
OUTPUT_DIR = ROOT_DIR.joinpath("output")
BIN_DIR = ROOT_DIR.joinpath("bin")

SECT_LABELS = ['address', 'affiliation', 'author', 'bodyText', 'category', 'construct', 'copyright', 'email',
               'equation', 'figure', 'figureCaption', 'footnote', 'keyword', 'listItem', 'note', 'page', 'reference',
               'sectionHeader', 'subsectionHeader', 'subsubsectionHeader', 'table', 'tableCaption', 'title']
GENE_LABELS = ['abstract', 'acknowledgments', 'background', 'categories-and-subject-descriptors', 'conclusions',
               'discussions', 'evaluation', 'general-terms', 'introduction', 'method', 'references', 'related-works']
