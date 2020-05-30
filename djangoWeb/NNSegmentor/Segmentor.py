import torch
import torch.nn as nn
import os
from django.conf import settings
from transformers import BertModel, BertTokenizer


class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.cnn = nn.Sequential(
            nn.Conv1d(768+15, 768+15, 7, stride=1, padding=3),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(768+15, 2)

        self.dictionary = torch.load(os.path.join(settings.BASE_DIR, 'NNSegmentor/dictionary.pt'))
        self.embeddings1 = None
        self.embeddings2 = None

        self.__load_model()
        #
        self.cnn.to('cuda:0')
        self.cls.to('cuda:0')
        self.dropout.to('cuda:0')

    def forward(self, inst):
        with torch.no_grad():
            dict_tensor = self.__get_dict_tensor(inst)
            embeddings = self.__get_embeddings(inst)
            embeddings = torch.cat([embeddings, dict_tensor], 1).unsqueeze(0)
            assert embeddings.shape[2] == 768+15, 'cat error'
            cnn_out = self.cnn(self.dropout(embeddings.permute(0, 2, 1)))
            pred = self.cls(cnn_out.permute(0, 2, 1))
        return pred.squeeze()

    def __get_embeddings(self, inst):
        inst = '-' + inst + '-'
        res = []
        for i in range(1, len(inst)-1):
            if inst[i-1:i+2] in self.embeddings1:
               res.append(self.embeddings1[inst[i-1:i+2]])
            elif inst[i-2:i+2] in self.embeddings2:
                res.append(self.embeddings2[inst[i - 1:i + 2]])
            elif inst[i] in self.embeddings1:
                res.append(self.embeddings1[inst[i]])
            elif inst[i] in self.embeddings2:
                res.append(self.embeddings2[inst[i]])
            else:
                res.append([0.]*768)
                print('error')
        return torch.tensor(res).to('cuda:0')

    def __get_dict_tensor(self, inst):
        inst = '--'+inst+'--'
        res = []
        for i in range(len(inst) - 4):
            data = []
            for j in range(5):
                for k in range(j, 5):
                    data.append(1 if inst[i + j:i + k + 1] in self.dictionary else 0)
            assert len(data) == 15
            res.append(data)
        return torch.tensor(res, dtype=torch.float).to('cuda:0')

    def __load_model(self):
        self.embeddings1 = torch.load(os.path.join(settings.BASE_DIR, 'NNSegmentor/embeddings_1.pt'))
        # embeddings2 = torch.load(os.path.join(settings.BASE_DIR, 'NNSegmentor/embeddings_2.pt'))
        # self.embeddings = embeddings1.update(embeddings2)
        model_state = torch.load(os.path.join(settings.BASE_DIR, 'NNSegmentor/model.pt'))
        self.cnn.load_state_dict(model_state['cnn'])
        self.cls.load_state_dict(model_state['cls'])
        self.dropout.load_state_dict(model_state['dropo'])


