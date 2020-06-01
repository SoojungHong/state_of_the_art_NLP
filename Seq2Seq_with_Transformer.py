# ---------------------------------------------------------------------------
# The model is based on following reference and input & output 
# reference : https://d2l.ai/chapter_attention-mechanisms/transformer.html
# ---------------------------------------------------------------------------

import d2l  # D2L AI library
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()

# ToDo : Why we project and transpose these matrices?
def transpose_qkv(X, num_heads):    # transpose query, key, value matrices
    # Input X shape: (batch_size, seq_len, num_hiddens).
    # Output X shape: (batch_size, seq_len, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)    # -1 for lazy dimension size calculation

    # X shape: (batch_size, num_heads, seq_len, num_hiddens / num_heads)
    X = X.transpose(0, 2, 1, 3)

    # output shape: (batch_size * num_heads, seq_len, num_hiddens / num_heads)
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output


def transpose_output(X, num_heads):
    # A reversed version of transpose_qkv
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def masked_softmax(X, valid_len):   # ToDo : Why masked softmax is necessary? What is valid_len?
    # X: 3-D tensor, valid_len: 1-D or 2-D tensor
    if valid_len is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_len.ndim == 1:
            valid_len = valid_len.repeat(shape[1], axis=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_len, True, axis=1, value=-1e6)
        return npx.softmax(X).reshape(shape)


# getting attention weights (basically the dot product of query and key)
class DotProductAttention(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_len: either (batch_size, ) or (batch_size, xx)  # ToDo : Why valid_len changes?
    def forward(self, query, key, value, valid_len=None):
        d = query.shape[-1]  # dimension
        # Set transpose_b=True to swap the last two dimensions of key
        scores = npx.batch_dot(query, key, transpose_b=True) / math.sqrt(d)  # check the reference (http://www.peterbloem.nl/blog/transformers_ - Why k−−√? Imagine a vector in ℝk with values all c. Its Euclidean length is k−−√c. Therefore, we are dividing out the amount by which the increase in dimension increases the length of the average vectors.
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return npx.batch_dot(attention_weights, value)


class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        from mxnet.gluon import nn  # explicitly import to prevent the confusion with Torch.nn

        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=False, flatten=False)

    def forward(self, query, key, value, valid_len):
        # For self-attention, query, key, and value shape: (batch_size, seq_len, dim), seq_len is the length of input sequence
        # valid_len shape is either (batch_size, ) or (batch_size, seq_len)
        # Project and transpose query, key, and value from
        # (batch_size, seq_len, num_hiddens) to
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads).
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_len is not None:
            # Copy valid_len by num_heads times
            if valid_len.ndim == 1:
                valid_len = np.tile(valid_len, self.num_heads)  #numpy.tile(A, reps) : construct arrau by repeating A with reps times
            else:
                valid_len = np.tile(valid_len, (self.num_heads, 1))

        # For self-attention, output shape:
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        output = self.attention(query, key, value, valid_len)

        # output_concat shape: (batch_size, seq_len, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)  # ToDo : visualize and check the output and output_concat shape


cell = MultiHeadAttention(90, 9, 0.5) # num_hiddens, num_heads, dropout,
cell.initialize()
X = np.ones((2, 4, 5))
print('X : ', X)
valid_len = np.array([2, 3])
print('valid_len : ', valid_len)
print('cell(X, X, X, valid_len).shape : ', cell(X, X, X, valid_len).shape)
print('cell : ', cell)



class PositionWiseFFN(nn.Block):
    def __init__(self, ffn_num_hiddens, pw_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        from mxnet.gluon import nn

        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False, activation='relu')   # Dense implements the operation: output = activation(dot(input, weight) + bias) where activation is the element-wise activation function passed as the activation argument, weight is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
        self.dense2 = nn.Dense(pw_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))



ffn = PositionWiseFFN(4, 8)
ffn.initialize()
print('debug : ', ffn(np.ones((2, 3, 4)))[0])


# ToDo : batch normalization vs. layer normalization
layer = nn.LayerNorm()
layer.initialize()
batch = nn.BatchNorm()
batch.initialize()
X = np.array([[1, 2], [2, 3]])
# Compute mean and variance from X in the training mode
with autograd.record():
    print('layer norm:', layer(X), '\nbatch norm:', batch(X))


class AddNorm(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        from mxnet.gluon import nn # TEST

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


add_norm = AddNorm(0.5)
add_norm.initialize()
print('debug : ', add_norm(np.ones((2, 3, 4)), np.ones((2, 3, 4))).shape)


class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        from mxnet.gluon import nn  #TEST
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)  # :: means
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)


pe = PositionalEncoding(20, 0)
pe.initialize()
Y = pe(np.zeros((1, 100, 20)))
d2l.plot(np.arange(100), Y[0, :, 4:8].T, figsize=(6, 2.5), legend=["dim %d" % p for p in [4, 5, 6, 7]])   # FIXME : visualize


# Saved in the d2l package for later use
class EncoderBlock(nn.Block):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_len):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_len))
        return self.addnorm2(Y, self.ffn(Y))


X = np.ones((2, 100, 24))
encoder_blk = EncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
print('debug : ', encoder_blk(X, valid_len).shape)



# Saved in the d2l package for later use
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        from mxnet.gluon import nn # TEST

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        eb = EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout)
        for _ in range(num_layers):
            self.blks.add(eb)
                #EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout))

    def forward(self, X, valid_len, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X = blk(X, valid_len)
        return X


#encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
#encoder.initialize()
#print('debug :' , encoder(np.ones((2, 100)), valid_len).shape)


class DecoderBlock(nn.Block):
    # i means it is the i-th block in the decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_len = state[0], state[1]
        # state[2][i] contains the past queries for this block
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if autograd.is_training():
            batch_size, seq_len, _ = X.shape
            # Shape: (batch_size, seq_len), the values in the j-th column
            # are j+1
            valid_len = np.tile(np.arange(1, seq_len+1, ctx=X.ctx),
                                   (batch_size, 1))
        else:
            valid_len = None

        X2 = self.attention1(X, key_values, key_values, valid_len)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_len)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state



decoder_blk = DecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_len), valid_len, [None]]
print('debug : ', decoder_blk(X, state)[0].shape)



class TransformerDecoder(d2l.Decoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        from mxnet.gluon import nn  # TEST

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(
                DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                             dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, env_valid_len, *args):
        return [enc_outputs, env_valid_len, [None]*self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X, state = blk(X, state)
        return self.dense(X), state


num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.0, 64, 10
# ORG - lr, num_epochs, ctx = 0.005, 100, d2l.try_gpu()
lr, num_epochs, ctx = 0.005, 100, d2l.try_gpu() # TEST
ffn_num_hiddens, num_heads = 64, 4


# ORG src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps)
# ToDo : Switch the data from query
"""
def ORG_load_data_nmt(batch_size, num_steps, num_examples=1000):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array(source, src_vocab, num_steps, True)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, num_steps, False)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return src_vocab, tgt_vocab, data_iter
"""


import random
import os
import torch
import torch.nn.functional as F

from src.phrase_embedding.phrase_embedding import *


# -----------------------------------
#  Loading input & output sequence
# -----------------------------------
#queries = "C:/Users/shong/PycharmProjects/Sequence2Sequence/data/mvp_input_sequences_with_spec.txt"
#output = "C:/Users/shong/PycharmProjects/Sequence2Sequence/data/mvp_output_sequences_with_spec.txt"

queries = "C:/Users/shong/PycharmProjects/Sequence2Sequence/data/mvp_input_sequences_with_spec_add.txt"
output = "C:/Users/shong/PycharmProjects/Sequence2Sequence/data/mvp_output_sequences_with_spec_add.txt"

# debugging
#queries = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/few_queries_input.txt"
#output = "C:/Users/shong/PycharmProjects/ngls_query/Intent-Slot-Tagging-Model/src/data/few_queries_output.txt"


# -------------------
# global param
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

p_embedding = PhraseEmbedding()


# -------------------------------
#  phrase to index dictionary
# -------------------------------
class PhraseSet:

    def __init__(self, name):
        self.name = name
        self.phrase2index = {}
        self.phrase2count = {}
        #self.phrase2vector = {}
        #self.vector2index = {}
        #self.index2vector = {}
        self.index2phrase = {0: "SOS", 1: "EOS"}
        self.n_phrase = 2  # Count SOS and EOS
        # attribute for Transformer
        self.idx_to_token = {}
        self.token_freqs = {}
        self.token_to_idx = {}

    def add_list_of_phrase(self, list_of_phrase):
        for phrase in list_of_phrase.split(','):
            self.addPhrase(phrase)


    def get_phrase_vector(self, phrase):
        phrase_vector = p_embedding.get_embedding(phrase)
        return phrase_vector


    def addPhrase(self, phrase):
        for p in phrase:
            if p not in self.phrase2index:
                self.phrase2index[p] = self.n_phrase
                self.phrase2count[p] = 1
                self.index2phrase[self.n_phrase] = p
                #self.phrase2vector[p] = self.get_phrase_vector(p)
                #self.vector2index[p] = self.n_phrase
                #self.index2vector[self.n_phrase] = self.get_phrase_vector(p)
                self.n_phrase += 1
                # attribute for Transformer
                self.idx_to_token[self.n_phrase] = p
                self.token_freqs[p] = 1
                self.token_to_idx[p] = self.n_phrase

            else:
                self.phrase2count[p] += 1
                # attribute for Transformer
                self.token_freqs[p] += 1

def readPhrases():
    print("Reading lines...")

    input_lines = []
    output_lines = []
    num_lines = 0
    max_num_phrase = 0

    f = open(queries, "r")
    ff = open(output, "r")

    for line in f:
        line_phrase_list = line.strip().split(',')
        input_lines.append(line_phrase_list)
        num_lines = num_lines + 1

    for line in ff:
        line_phrase_list = line.strip().split(',')
        output_lines.append(line_phrase_list)
        if max_num_phrase < len(line_phrase_list):
            max_num_phrase = len(line_phrase_list)

    pairs_input = []
    pairs_output = []

    # make PhraseSet instances
    for i in range(num_lines):
        pairs_input.append(input_lines[i])
        pairs_output.append(output_lines[i])

    pairs = [pairs_input, pairs_output]
    input_set = PhraseSet('input')
    output_set = PhraseSet('output')

    return input_set, output_set, pairs, num_lines, max_num_phrase


def readPhrases_for_transformer():
    print("Reading lines...")

    input_lines = []
    output_lines = []
    num_lines = 0
    max_num_phrase = 0

    f = open(queries, "r")
    ff = open(output, "r")

    for line in f:
        line_phrase_list = line.strip().split(',')
        input_lines.append(line_phrase_list)
        num_lines = num_lines + 1

    for line in ff:
        line_phrase_list = line.strip().split(',')
        output_lines.append(line_phrase_list)
        if max_num_phrase < len(line_phrase_list):
            max_num_phrase = len(line_phrase_list)

    return input_lines, output_lines



# ------------------------------------
# summary of preparing the data
# ------------------------------------
# Read text file and split into lines, split lines into pairs
# Normalize text, filter by length and content
# Make word lists from sentences in pairs

def prepareData():
    input_set, output_set, pairs, num_lines, max_num_phrase = readPhrases()
    print("Read %s input output pairs" % len(pairs))
    print("Counting phrases...")

    for pair_input in pairs[0]:
        input_set.addPhrase(pair_input)

    for pair_output in pairs[1]:
        output_set.addPhrase(pair_output)

    print("Counted phrases:")
    print(input_set.name, input_set.n_phrase)
    print(output_set.name, output_set.n_phrase)
    #print(input_set.phrase2vector)
    #print(output_set.phrase2vector)

    return input_set, output_set, pairs, num_lines, max_num_phrase


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Trim
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]
    if not is_source:
        lines = [[vocab['<bos>']] + l + [vocab['<eos>']] for l in lines]
    array = np.array([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).sum(axis=1)
    return array, valid_len

from mxnet import autograd, context, gluon, image, init, np, npx

class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_len shape: (batch_size, )
    def forward(self, pred, label, valid_len):
        # weights shape: (batch_size, seq_len, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)


def train_s2s_ch9(model, data_iter, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),'adam', {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',xlim=[1, num_epochs], ylim=[0, 0.25])
    animator.fig.subplots_adjust(hspace=0.3) #FIXME - TEST
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for batch in data_iter:
            X, X_vlen, Y, Y_vlen = [x.as_in_context(ctx) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen-1
            with autograd.record():
                Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
                l = loss(Y_hat, Y_label, Y_vlen)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_vlen.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if epoch % 10 == 0:
            animator.add(epoch, (metric[0]/metric[1],))

    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))


def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps, ctx):
    # ORG src_tokens = src_vocab[src_sentence.lower().split(' ')] # TODO : if vocab doesn't contain, instead of 0, assign other index
    src_tokens = src_vocab[src_sentence.lower().split(',')]
    num_steps = len(src_tokens) # Fix
    enc_valid_len = np.array([len(src_tokens)], ctx=ctx)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = np.array(src_tokens, ctx=ctx)
    # Add the batch_size dimension
    enc_outputs = model.encoder(np.expand_dims(enc_X, axis=0), enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=ctx), axis=0)
    predict_tokens = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # The token with highest score is used as the next timestep input
        dec_X = Y.argmax(axis=2)
        py = dec_X.squeeze(axis=0).astype('int32').item()
        #print("debug : ", py)
        if py == tgt_vocab['<eos>']:
            print('py : ', py)  #ToDo
            break
        elif py == tgt_vocab['<unk>']:
            print('py : ', py)  #ToDo
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))



def my_load_data(batch_size, num_steps, num_examples=10): #num_examples=1000):
    #ORG - text = preprocess_nmt(read_data_nmt())
    #ORG - source, target = tokenize_nmt(text, num_examples)  # source and target are list type

    # TODO :
    #source, target, pairs, num_lines, max_num_phrase = prepareData()
    source, target = readPhrases_for_transformer()

    # ORG - src_vocab = d2l.Vocab(source, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # ORG - src tgt_vocab = d2l.Vocab(target, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_vocab = d2l.Vocab(source, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'])  # ToDo : min_freq influence how?
    tgt_vocab = d2l.Vocab(target, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    src_array, src_valid_len = build_array(source, src_vocab, num_steps, True)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, num_steps, False)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return src_vocab, tgt_vocab, data_iter

src_vocab, tgt_vocab, train_iter = my_load_data(batch_size, num_steps)


encoder = TransformerEncoder(len(src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers,dropout)
model = d2l.EncoderDecoder(encoder, decoder)

#ORG - d2l.train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
# ToDo : save model and load next time without training
for sentence in ["Find a apple store near bistro,apple store,bistro,near a bistro"]:
        #["where can i eat steak in 10 minutes by car around my hotel,steak,10 minutes,10 minutes by car around my hotel,around my hotel"]:
        #["show me nike within a big mall,nike,within a big mall,mall"]:
        #["find me a bus station close to mcdonalds along my route,bus station,close to,mcdonalds,along my route,route"]:
    #["where can i eat hamburger around my hotel,hamburger,my hotel,around my hotel"] :
    #["where can i get something to eat around my hotel at 11 pm ?,something to eat,around my hotel,at 11 pm"]:
   #["Find a food store near bistro,food store,bistro,near a bistro"]:
    print("Input Sequence : ", sentence)
    print("Output Labels : ", predict_s2s_ch9(model, sentence, src_vocab, tgt_vocab, num_steps, ctx))




