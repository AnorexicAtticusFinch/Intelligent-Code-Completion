import torch
import torch.nn
from labml import experiment, logger
from labml_helpers.module import Module
from labml.logger import Text, Style
from labml.utils.pytorch import get_modules
from train import Configs, TextDataset

class Predictor:

    def __init__(self, model, dataset):
        self.dataset = dataset
        self.model = model
        self.prompt = ''

    def get_all_predictions(self, char):
        self.prompt += char
        self.prompt = self.prompt[-512:]
        data = torch.tensor([[self.dataset.stoi[c]] for c in self.prompt], dtype=torch.long, device=self.model.device)
        prediction, tmp = self.model(data)
        prediction = prediction[-1, :, :]
        return prediction.detach().cpu().numpy()

    def get_prediction(self, char):
        prediction = self.get_all_predictions(char)
        best = prediction.argmax(-1).squeeze().item()
        return self.dataset.itos[best]

def calculate_keystrokes_saved(pos, text):
    ans = 0
    while pos < len(text) and text[pos] != "." and text[pos] != " " and text[pos] != "\n":
        ans += 1
        pos += 1
    return ans

class Validator:
    
    def __init__(self, model, dataset, text):
        self.text = text
        self.predictor = Predictor(model, dataset)

    def validate(self):
        line_no = 1
        logs = [(f'{line_no: 4d}: ', Text.meta), (self.text[0], Text.subtle)]
        correct = 0
        keystrokes = 0
        was_correct = False
        i = 0

        while i < (len(self.text) - 1):
            next_token = self.predictor.get_prediction(self.text[i])
            saved = 0
            if next_token == self.text[i + 1]:
                correct += 1
                saved = calculate_keystrokes_saved(i+1, self.text)
                keystrokes += saved
                if self.text[i + 1] != ' ' and self.text[i + 1] != '\n':
                    was_correct = True
                
            if self.text[i + 1] == '\n':
                logger.log(logs)
                if was_correct:
                    logger.log('\tCurrent Accuracy: ', (f'{correct / (i+1) :.2f}', Text.value))
                    logger.log('\tCurrent Keystrokes Saved: ', (f'{keystrokes :.0f}', Text.value))
                was_correct = False
                line_no += 1
                logs = [(f"{line_no: 4d}: ", Text.meta)]
            elif self.text[i + 1] == '\r':
                continue
            else:
                if next_token == self.text[i + 1] and self.text[i + 1] != ' ':
                    logs.append((self.text[i + 1], Style.underline))
                else:
                    logs.append((self.text[i + 1], Text.subtle))

            for j in range(saved):
                i += 1
                if i < (len(self.text) - 1) and self.text[i + 1] != '\n':
                    logs.append((self.text[i + 1], Text.subtle))

            if i < (len(self.text) - 1) and self.text[i + 1] == '\n' and was_correct:
                logger.log(logs)
                logger.log('\tCurrent Accuracy: ', (f'{correct / (i+1) :.2f}', Text.value))
                logger.log('\tCurrent Keystrokes Saved: ', (f'{keystrokes :.0f}', Text.value))
                was_correct = False
                line_no += 1
                logs = [(f"{line_no: 4d}: ", Text.meta)]
                i += 1

            i += 1

        logger.log(logs)
        logger.log('Total Accuracy: ', (f'{correct / (len(self.text) - 1) :.2f}', Text.value))
        logger.log('Total Keystrokes Saved: ', (f'{keystrokes :.0f}', Text.value))

def main():
    conf = Configs()
    experiment.create(name='code_completion_validate', comment='lstm model')
    conf_dict = experiment.load_configs('29ea0a701d1711eb9035080027277f49')
    experiment.configs(conf, conf_dict, 'run')
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load('29ea0a701d1711eb9035080027277f49')
    experiment.start()
    validator = Validator(conf.model, conf.text, conf.text.valid)
    validator.validate()

if __name__ == '__main__':
    main()

#29ea0a701d1711eb9035080027277f49
#7af365b009ee11ebb7ab080027277f49