import torch
import torch.nn as nn

from transformers import BertModel


class TabBERT(nn.Module):
    def __init__(self, embedding_sizes, n_continuous, transformer_model, device_tm):
        super().__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, embedding_dim)
                for num_classes, embedding_dim in embedding_sizes
            ]
        )

        self.dropout = nn.Dropout(0.3)

        self.bert = BertModel.from_pretrained(transformer_model).to(device_tm)
        for param in self.bert.parameters():
            param.requires_grad = False

        # Fully connected layers
        input_size = (
            sum(e[1] for e in embedding_sizes) + n_continuous + self.bert.config.hidden_size
        )
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, x_cat, x_cont):
        bert_output = self.bert(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        bert_output = bert_output.last_hidden_state[:, 0, :]  # bert_output.pooler_output

        # Pass through embedding layers
        x_embed = [emb_layer(x_cat[:, i].long()) for i, emb_layer in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, 1)
        x_embed = self.dropout(x_embed)

        # Concatenate features
        x = torch.cat((bert_output, x_embed, x_cont), dim=1)

        # Pass through fully connected layers
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.elu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return self.sigmoid(x)
