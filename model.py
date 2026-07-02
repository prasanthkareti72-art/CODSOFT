import torch
import torch.nn as nn
import torchvision.models as models
from collections import Counter
import re

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word2idx = {v: k for k, v in self.idx2word.items()}

    def __len__(self):
        return len(self.idx2word)

    @staticmethod
    def tokenize(text):
        # Convert to lowercase and extract words
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.word2idx.get(token, self.word2idx["<unk>"])
            for token in tokenized_text
        ]


class CNNEncoder(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(CNNEncoder, self).__init__()
        self.train_cnn = train_cnn
        # Using ResNet50 (pre-trained on ImageNet)
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            # Fallback to pretrained=True for backward compatibility
            self.resnet = models.resnet50(pretrained=True)
            
        # Freeze ResNet parameters if not fine-tuning
        for param in self.resnet.parameters():
            param.requires_grad = self.train_cnn

        # Replace the final fully connected classification layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Forward pass to extract features
        features = self.resnet(images)
        return self.dropout(self.relu(features))


class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # captions shape: (batch_size, seq_len)
        # We omit the <end> token from the inputs because the model shouldn't receive <end> as input
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        
        # Concatenate CNN features (batch_size, 1, embed_size) and word embeddings
        # shape: (batch_size, seq_len, embed_size)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # LSTM output shape: (batch_size, seq_len, hidden_size)
        hiddens, _ = self.lstm(inputs)
        
        # Project LSTM outputs to vocabulary size
        outputs = self.linear(hiddens)
        return outputs
