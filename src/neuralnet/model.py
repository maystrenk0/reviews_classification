import torch
import torch.nn as nn

from transformers import BertForSequenceClassification


class TabBERT(nn.Module):
    def __init__(self, embedding_sizes, n_continuous, transformer_model, tm_device):
        super().__init__()

        # # Embedding layers for categorical features
        # self.embeddings = nn.ModuleList(
        #     [
        #         nn.Embedding(num_classes, embedding_dim)
        #         for num_classes, embedding_dim in embedding_sizes
        #     ]
        # )

        # self.dropout = nn.Dropout(0.3)

        # Sentiment model (Hugging Face BERT)
        self.sentiment_model = BertForSequenceClassification.from_pretrained(
            transformer_model, num_labels=2
        ).to(tm_device)
        self.sentiment_model.train()  # Ensure it is in training mode

        # # Fully connected layers
        # input_size = sum(e[1] for e in embedding_sizes) + n_continuous + 1
        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 1)

        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, x_cat, x_cont):
        # Sentiment analysis from BERT
        sentiment_output = self.sentiment_model(
            input_ids=input_ids.long(), attention_mask=attention_mask.long()
        )
        sentiment_logits = sentiment_output.logits
        sentiment_probs = self.softmax(sentiment_logits)
        sentiment_pos = sentiment_probs[:, 1]
        return sentiment_pos
        # Pass through embedding layers
        x_embed = [emb_layer(x_cat[:, i].long()) for i, emb_layer in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, 1)
        x_embed = self.dropout(x_embed)

        # Concatenate features
        x = torch.cat((sentiment_pos.unsqueeze(1), x_embed, x_cont), dim=1)

        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return self.sigmoid(x)
