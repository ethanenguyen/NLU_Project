import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class EntityDetectionModel(nn.Module ):
    def __init__(self, config, n_classes, dropout, weight_name='bert-base-uncased'):
        super().__init__()
        self.n_classes = n_classes
        self.config = config
        self.bert = BertModel.from_pretrained(weight_name)
        if config == 1:
            self.relu = nn.ReLU()
        elif config == 2:
            self.batchnorm = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout( p=dropout )
        self.classification_layer = nn.Linear( 768, n_classes)

    def forward(self, X, mask):
        hidden_states, cls_output = self.bert( X, attention_mask = mask)
        hidden_states = self.dropout(hidden_states)

        if self.config == 1:
            hidden_states = self.relu(hidden_states)
        elif self.config == 2:
            hidden_states = hidden_states.permute(0,2,1)
            hidden_states = self.batchnorm(hidden_states)
            hidden_states = hidden_states.permute(0,2,1)


        return F.log_softmax(self.classification_layer( hidden_states), dim=2)
