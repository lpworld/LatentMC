from __future__ import absolute_import, division, print_function
import argparse, json, os, random, logging, sys, time, math
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import tensorflow as tf
from models import Seq2Seq
from collections import Counter
if tuple(map(int, tf.__version__.split("."))) >= (1, 6, 0):
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell
    _linear = core_rnn_cell._linear
else:
    from tensorflow.python.ops.rnn_cell_impl import _linear

class EmbeddingCompressor(object):
    temperature = 1.0
    batch_size = 64
    model_path = 'output'

    def __init__(self, M, K, model_path):
        self.M = M
        self.K = K
        self._model_path = model_path

    def _gumbel_softmax(self, logits, temperature):
        U = tf.random_uniform(tf.shape(logits),minval=0,maxval=1)
        y = logits + -tf.log(-tf.log(U + eps) + eps)
        y = tf.nn.softmax(y / temperature)
        return y

    def _encode(self, input_matrix, word_ids, embed_size):
        input_embeds = tf.nn.embedding_lookup(input_matrix, word_ids, name="input_embeds")
        with tf.variable_scope("h"):
            h = tf.nn.tanh(_linear(input_embeds, self.M*self.K/2, True))
        with tf.variable_scope("logits"):
            logits = _linear(h, M * K, True)
            logits = tf.log(tf.nn.softplus(logits) + 1e-8)
        logits = tf.reshape(logits, [-1, M, K], name="logits")
        return input_embeds, logits

    def build_export_graph(self, embed_matrix):
        vocab_size = embed_matrix.shape[0]
        embed_size = embed_matrix.shape[1]
        input_matrix = tf.constant(embed_matrix, name="embed_matrix")
        word_ids = tf.placeholder_with_default(np.array([3,4,5], dtype="int32"), shape=[None], name="word_ids")
        codebooks = tf.get_variable("codebook", [self.M*self.K, embed_size])
        input_embeds, logits = self._encode(input_matrix, word_ids, embed_size)
        codes = tf.cast(tf.argmax(logits, axis=2), tf.int32)
        offset = tf.range(self.M, dtype="int32")*self.K
        codes_with_offset = codes + offset[None, :]
        selected_vectors = tf.gather(codebooks, codes_with_offset)
        reconstructed_embed = tf.reduce_sum(selected_vectors, axis=1)
        return word_ids, codes, reconstructed_embed

    def build_training_graph(self, embed_matrix):
        vocab_size = embed_matrix.shape[0]
        embed_size = embed_matrix.shape[1]
        input_matrix = tf.constant(embed_matrix, name="embed_matrix")
        tau = tf.placeholder_with_default(np.array(1.0, dtype='float32'), tuple()) - 0.1
        word_ids = tf.placeholder_with_default(np.array([3,4,5], dtype="int32"), shape=[None], name="word_ids")
        codebooks = tf.get_variable("codebook", [self.M*self.K, embed_size])
        input_embeds, logits = self._encode(input_matrix, word_ids, embed_size)
        D = self._gumbel_softmax(logits, self.temperature, sampling=True)
        gumbel_output = tf.reshape(D, [-1, self.M * self.K])
        maxp = tf.reduce_mean(tf.reduce_max(D, axis=2))
        y_hat = tf.matmul(gumbel_output, codebooks) # Decoding
        loss = 0.5 * tf.reduce_sum((y_hat - input_embeds)**2, axis=1)
        loss = tf.reduce_mean(loss, name="loss")
        max_grad_norm = 0.001
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        global_norm = tf.identity(global_norm, name="global_norm")
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")
        return word_ids, loss, train_op, maxp

    def train(self, embed_matrix, max_epochs=200):
        vocab_size = embed_matrix.shape[0]
        valid_ids = np.random.RandomState(3).randint(0, vocab_size, size=(self.batch_size*10,)).tolist()

        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
                word_ids_var, loss_op, train_op, maxp_op = self.build_training_graph(embed_matrix)
            tf.global_variables_initializer().run()
            best_loss = 100000
            saver = tf.train.Saver()
            vocab_list = list(range(vocab_size))
            for epoch in range(max_epochs):
                start_time = time.time()
                train_loss_list = []
                train_maxp_list = []
                np.random.shuffle(vocab_list)
                for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                    word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                    loss, _, maxp = sess.run([loss_op, train_op, maxp_op],{word_ids_var: word_ids})
                    train_loss_list.append(loss)
                    train_maxp_list.append(maxp)
                valid_loss_list = []
                valid_maxp_list = []
                for start_idx in range(0, len(valid_ids), self._BATCH_SIZE):
                    word_ids = valid_ids[start_idx:start_idx + self._BATCH_SIZE]
                    loss, maxp = sess.run([loss_op, maxp_op],{word_ids_var: word_ids})
                    valid_loss_list.append(loss)
                    valid_maxp_list.append(maxp)
                valid_loss = np.mean(valid_loss_list)
                report_token = ""
                if valid_loss <= best_loss * 0.999:
                    report_token = "*"
                    best_loss = valid_loss
                    saver.save(sess, self.model_path)
                time_elapsed = time.time() - start_time
                bps = len(train_loss_list) / time_elapsed
                print("[epoch{}] trian_loss={:.2f} train_maxp={:.2f} valid_loss={:.2f} valid_maxp={:.2f} bps={:.0f} {}".format(
                    epoch,
                    np.mean(train_loss_list), np.mean(train_maxp_list),
                    np.mean(valid_loss_list), np.mean(valid_maxp_list),
                    len(train_loss_list) / time_elapsed,
                    report_token
                ))
        print("Training Done")

        compressed_embed = []
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph"):
                word_ids_var, codes_op, reconstruct_op = self.build_export_graph(embed_matrix)
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            vocab_list = list(range(embed_matrix.shape[0]))
            for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                codes = sess.run(codes_op, {word_ids_var: word_ids}).tolist()
                for code in codes:
                    compressed_embed.append(code)
        return compressed_embed

class AutoEncoder:
    def __init__(self):
        self.autoencoder = Seq2Seq(emsize=args.emsize,
                                   nhidden=args.nhidden,
                                   ntokens=args.ntokens,
                                   nlayers=args.nlayers,
                                   noise_radius=args.noise_radius,
                                   hidden_init=args.hidden_init,
                                   dropout=args.dropout,
                                   gpu=args.cuda)
        self.optimizer = optim.SGD(self.autoencoder.parameters(), lr=args.lr)
        #self.optimizer = optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if args.cuda:
            self.autoencoder = self.autoencoder.cuda()
            self.criterion = self.criterion.cuda()

    def update(self, batch):
        self.autoencoder.train()
        self.autoencoder.zero_grad()
        source, target, lengths = batch
        train_batch_size, z_dim = source.shape
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))
        output = self.autoencoder(source, lengths)
        output = output.reshape([args.batch_size*lengths[0],args.ntokens])
        loss = self.criterion(output, target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), args.clip)
        self.optimizer.step()
        return loss.data.numpy()

    def anneal(self):
        '''exponentially decaying noise on autoencoder'''
        self.autoencoder.noise_radius = self.autoencoder.noise_radius * args.noise_anneal

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename), 'wb') as f:
            torch.save(self.autoencoder.state_dict(), f)

def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

def batchify(data, bsz, dictionary, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []
    for i in range(nbatch):
        batch = data[i*bsz:(i+1)*bsz]
        source = [[dictionary.sos] + [dictionary[x] for x in line] for line in batch]
        target = [[dictionary[x] for x in line] + [dictionary.eos] for line in batch]
        lengths = [len(x)+1 for x in batch]
        batch, lengths = length_sort(batch, lengths)
        maxlen = max(lengths)
        for x, y in zip(source, target):
            paddings = (maxlen-len(x))*[Dictionary.eos]
            x += paddings
            y += paddings
        source = torch.LongTensor(source)
        target = torch.LongTensor(target)
        if gpu:
            source = source.cuda()
            target = target.cuda()
        batches.append((source, target, lengths))
    return batches

class Dictionary(object):
    pad = 0
    sos = 1
    eos = 2
    oov = 3
    offset = 4

    def __init__(self, words):
        self.word2idx = {}
        self.word2idx['<pad>'] = self.pad
        self.word2idx['<sos>'] = self.sos
        self.word2idx['<eos>'] = self.eos
        self.word2idx['<oov>'] = self.oov
        for idx, word in enumerate(words):
            self.word2idx[word] = idx + self.offset
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, key):
        if type(key) is str:
            if key in self.word2idx:
                return self.word2idx[key]
            else:
                return self.oov
        if type(key) is int:
            print(self.idx2word)
            if key in self.idx2word:
                return self.idx2word[key]
        raise KeyError(key)