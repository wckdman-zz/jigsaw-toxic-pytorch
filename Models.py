import torch
import torch.nn as nn


class GMP(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout=0.3):
        super(GMP, self).__init__()
        label_size = 6
        emb_size = 300
        hidden = 128

        self.word_embeddings = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False

        self.rnn = nn.GRU(emb_size, hidden, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(2*hidden, label_size)
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.drop(embeds)
        embeds, hid = self.rnn(embeds)
        embeds = embeds.max(1)[0]
        x = embeds
        x = self.linear(x)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)


    
    
class GRUCnn(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout=0.3):
        super(GRUCnn, self).__init__()
        label_size = 6
        emb_size = 300
        hid1 = 128
        hid2 = 128
        
        self.word_embeddings = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False

        self.conv = nn.Conv1d(2*hid1, hid2, 2)
        self.rnn = nn.GRU(emb_size, hid1, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(hid2, label_size)
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.drop(embeds)
        embeds, hid = self.rnn(embeds)
        embeds = embeds.permute(0, 2, 1)

        cnn = self.conv(embeds)
        cnn = self.activation(cnn)
        cnn = cnn.permute(0, 2, 1)
        cnn = cnn.max(1)[0]
        x = self.linear(cnn)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)
