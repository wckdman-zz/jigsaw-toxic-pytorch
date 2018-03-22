import random
import numpy as np
import torch
from torch.autograd import Variable

class DataLoader(object):

    def __init__(
            self, src_word2idx,
            src=None, tgt=None,
            cuda=True, batch_size=64, shuffle=True, test=False):

        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src) / batch_size))

        self._batch_size = batch_size

        self._src = src
        self._tgt = tgt

        src_idx2word = {idx: word for word, idx in src_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        return len(self._src)

    @property
    def src_vocab_size(self):
        return len(self._src_word2idx)

    @property
    def src_word2idx(self):
        return self._src_word2idx


    @property
    def src_idx2word(self):
        return self._src_idx2word


    def shuffle(self):
        if self._tgt is not None:
            paired_insts = list(zip(self._src, self._tgt))
            random.shuffle(paired_insts)
            self._src, self._tgt = zip(*paired_insts)
        else:
            random.shuffle(self._src)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):

        def pad_to_longest(insts):

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [0] * (max_len - len(inst))
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src = self._src[start_idx:end_idx]
            src_data = pad_to_longest(src)

            if self._tgt is not None:
                tgt = self._tgt[start_idx:end_idx]
                tgt = Variable(torch.FloatTensor(tgt), volatile=self.test)
            else:
                return src_data, None
            if self.cuda:
                tgt = tgt.cuda()
            return src_data, tgt

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
