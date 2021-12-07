import ctranslate2
import fastBPE
import os
from sacremoses import MosesTokenizer, MosesDetokenizer

from olsen.constants import DATASET_DIR, BIN_DIR

device = 0
source = DATASET_DIR.joinpath("unlabelled", "original")
target = DATASET_DIR.joinpath("unlabelled", "translated")
input_files = os.listdir(source)
normalizer = MosesTokenizer(lang='en')
joiner = MosesDetokenizer(lang='en')
bpe = fastBPE.fastBPE(str(BIN_DIR.joinpath("tokenize/bpecodes")), str(BIN_DIR.joinpath("tokenize/dict.en.txt")))
en_de = ctranslate2.Translator(str(BIN_DIR.joinpath("translate/en-de")), device="cuda", device_index=[device])
de_en = ctranslate2.Translator(str(BIN_DIR.joinpath("translate/de-en")), device="cuda", device_index=[device])
for file in input_files:
    with open(source.joinpath(file)) as f:
        original_lines = f.readlines()
        normalized = [normalizer.tokenize(line, return_str=True) for line in original_lines]
        bpe_processed = bpe.apply(normalized)
        tokenized = [line.split() for line in bpe_processed]
        translated = en_de.translate_batch(tokenized, max_batch_size=128, beam_size=1,
                                           sampling_topk=0, sampling_temperature=0.8)
        translation = [translated[i].hypotheses[0] for i in range(len(translated))]
        back_tran = de_en.translate_batch(translation, max_batch_size=128, beam_size=1,
                                          sampling_topk=0, sampling_temperature=0.8)
        back_result = [back_tran[i].hypotheses[0] for i in range(len(back_tran))]
        backtran_lines = [joiner.detokenize(line).replace(" @-@ ", "-").replace("@@ ", "").replace("@@", "") for line in
                          back_result]
    result = [oline.strip() + "###" + bline.strip() for oline, bline in zip(original_lines, backtran_lines)]
    result = "\n".join(result)
    with open(target.joinpath(file), 'w') as f:
        f.write(result)
        f.write("\n")
del en_de
del de_en
