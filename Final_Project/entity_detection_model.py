import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class EntityDetectionModel(nn.Module ):
    def __init__(self, n_classes, dropout, weight_name='bert-base-uncased'):
        super().__init__()
        self.n_classes = n_classes
        self.bert = BertModel.from_pretrained(weight_name)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout( p=dropout )
        self.classification_layer = nn.Linear( 768, n_classes)

    def forward(self, X, mask):
        hidden_states, cls_output = self.bert( X, attention_mask = mask)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return F.log_softmax(self.classification_layer( hidden_states), dim=2)
