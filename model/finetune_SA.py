import torch.nn as nn
from model.bert import BERT
import config.hparams as hp


class BertForSA(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, num_labels=2):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(hp.hidden, hp.hidden)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(hp.hidden, num_labels)

        self.init_layer(self.classifier)

    def forward(self, x, pos):
        x, attn_list = self.bert(x, pos)
        pooled_h = self.activ(self.fc(x[:, 0]))
        logits = self.classifier(self.drop(pooled_h))

        return logits


    def init_layer(self, layers):
        for p in layers.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

