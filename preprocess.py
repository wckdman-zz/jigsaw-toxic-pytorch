import argparse
import torch
import numpy as np
from gensim.models import KeyedVectors


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'


def read_instances_from_file(inst_file, max_sent_len, keep_case):

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file, encoding='utf-8') as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()

            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [word_inst]
            else:
                word_insts += [PAD_WORD]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts


def build_vocab_idx(word_insts, min_word_count):

    full_vocab = set(w for sent in word_insts for w in sent if len(sent)>0)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        PAD_WORD: 0,
        UNK_WORD: 1}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):

    return [[word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in s] for s in word_insts]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-test_src', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=400)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-embeds', action='store_true')

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)

    # Validation set
    test_src_word_insts = read_instances_from_file(
        opt.test_src, opt.max_word_seq_len, opt.keep_case)

    # Build vocabulary

    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + test_src_word_insts, opt.min_word_count)
            src_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)

    embedding_matrix = None
    
    if opt.embeds:
        print('[Info] Creating embedding matrix', opt.save_data)
        ft_model = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec')
        emb_mean, emb_std = ft_model.vectors.mean(), ft_model.vectors.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (len(src_word2idx), 300) )
        
        not_found = 0
        for k, v in src_word2idx.items():
            try:
                embedding_matrix[v] = ft_model.get_vector(k)
            except:
                not_found += 1
        print('[Info] words not found. ', not_found)
        print('[Info] Done. ', embedding_matrix.shape)

    # word to index

    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, src_word2idx)
    
    train_tgt = np.loadtxt(opt.train_tgt)

    data = {
        'settings': opt,
        'embeds': embedding_matrix,
        'dict': src_word2idx,

        'train': {
            'src': train_src_insts,
            'tgt': train_tgt,
        },
            
        'test': {
            'src': test_src_insts,
        }}
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
