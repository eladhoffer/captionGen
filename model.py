import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class CaptionModel(nn.Module):

    def __init__(self, cnn, vocab, embedding_size=256, rnn_size=256, num_layers=1):
        super(CaptionModel, self).__init__()
        self.vocab = vocab
        self.cnn = cnn
        self.cnn.fc = nn.Linear(2048, embedding_size)
        self.embedder = nn.Embedding(len(self.vocab), embedding_size)
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers=num_layers)
        self.classifier = nn.Linear(rnn_size, len(vocab))

    def forward(self, imgs, captions, lengths):
        embeddings = self.embedder(captions.t()).t()

        img_feats = self.cnn(imgs).unsqueeze(0)
        embeddings = torch.cat([img_feats, embeddings], 0)
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        feats, state = self.rnn(packed_embeddings)
        pred = self.classifier(feats[0])

        return pred, state

    def generate(self, img, end_token='EOS', max_length=20):
        word2idx = {word: idx for idx, (_, word) in enumerate(self.vocab)}
        end_idx = word2idx[end_token]
        self.eval()
        state = None
        imgs = Variable(img.unsqueeze(0), volatile=True)
        embeddings = self.cnn(imgs).unsqueeze(0)
        txt = []
        while True:
            feats, state = self.rnn(embeddings, state)
            pred = self.classifier(feats.squeeze(0))
            prob, idx = pred.max(1)
            idx_num = idx.data.squeeze()[0]
            print(idx_num)
            txt.append(self.vocab[idx_num][1])
            if idx_num == end_idx or len(txt) == max_length:
                break
            embeddings = self.embedder(idx)

        return txt

    def save_checkpoint(self, filename):
        torch.save({'embedder': self.embedder.state_dict(),
                    'rnn': self.rnn.state_dict(),
                    'cnn': self.cnn.state_dict(),
                    'classifier': self.classifier.state_dict()},
                   filename)

    def load_checkpoint(self, filename):
        cpnt = torch.load(filename)
        self.embedder.load_state_dict(cpnt['embedder'])
        self.rnn.load_state_dict(cpnt['rnn'])
        self.cnn.load_state_dict(cpnt['cnn'])
        self.classifier.load_state_dict(cpnt['classifier'])

    def finetune_cnn(self, allow=True):
        for p in self.cnn.parameters():
            p.requires_grad = allow
        for p in self.cnn.fc.parameters():
            p.requires_grad = True
