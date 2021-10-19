from typing import List
import torch

class BackTranslator():
    def __init__(self, device: int):
        self.device = torch.device(f"cuda:{device}" if device > -1 else "cpu")

        self.en2de = torch.hub.load('pytorch/fairseq:main', 'transformer.wmt19.en-de',
                                    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                                    tokenizer='moses', bpe='fastbpe')
        self.en2de.eval()  # disable dropout
        self.en2de.to(self.device)

        self.de2en = torch.hub.load('pytorch/fairseq:main', 'transformer.wmt19.de-en',
                                    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                                    tokenizer='moses', bpe='fastbpe')
        self.de2en.eval()  # disable dropout
        self.de2en.to(self.device)


    def back_translate(self, inputs: List[str]) -> List[str]:
        result = []
        for i in range(len(inputs), 64):
            batch = inputs[i:i+64]
            translated = self.en2de.translate(batch)
            back = self.de2en.translate(translated)
            result.extend(back)
        return result


def get_tsa_thresh(schedule, global_step, num_train_steps, num_classes):
    start = 1.0 / num_classes
    end = 1
    progress = min(1.0, float(global_step) / float(num_train_steps))
    training_progress = torch.tensor(progress)
    scale = 5
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        threshold = 1 - torch.exp((-training_progress) * scale)
    elif schedule == 'root_schedule':
        threshold = torch.pow(training_progress, float(1 / scale))
    elif schedule == 'shifted_schedule':
        threshold = 0.5 + training_progress/2
    else:
        raise ValueError
    return threshold * (end - start) + start

