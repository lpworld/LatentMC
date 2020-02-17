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
from surprise import KNNBaseline, SVD, BaselineOnly, SlopeOne, CoClustering, Dataset, Reader
from surprise.model_selection import cross_validate
from utils import *
    
parser = argparse.ArgumentParser('Train an autoencoder.')
parser.add_argument('--outf', type=str, default='output', help='output directory name')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--emsize', type=int, default=300, help='size of word embeddings')
parser.add_argument('--emsize_len', type=int, default=5)
parser.add_argument('--nhidden', type=int, default=300, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--lr', type=float, default=1, help='autoencoder learning rate')
parser.add_argument('--noise_radius', type=float, default=0.2, help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995, help='anneal noise_radius exponentially every 100 iterations')
parser.add_argument('--hidden_init', action='store_true', help="initialize decoder hidden state with encoder's")
parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping, max norm')
parser.add_argument('--seed', type=int, default=625, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=1)
args = parser.parse_args()
print(vars(args))

# create output directory
out_dir = './{}'.format(args.outf)
os.makedirs(out_dir, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

# prepare corpus
data = pd.read_csv('test.csv')
text = list(data['text'])
text = [line.strip().lower().split(' ') for line in text]

# train autoencoder
vocab = dict(Counter([word for line in text for word in line]))
vocab = [x for x, y in vocab.items() if y>20]
dictionary = Dictionary(vocab)
args.ntokens = len(vocab)+5
autoencoder = AutoEncoder()
for epoch in range(1, args.epochs + 1):
    batches = batchify(text, args.batch_size, dictionary, shuffle=True)
    global_iters = 0
    start_time = datetime.now()
    for i, batch in enumerate(batches):
        loss = autoencoder.update(batch)
        if i % args.log_interval == 0 and i > 0:
            log.info(('[Epoch {} {}/{} Loss {:.5f} ETA {}]').format(
                epoch, i, len(batches), loss,
                str((datetime.now() - start_time) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        global_iters += 1
        if global_iters % 100 == 0:
            autoencoder.anneal()
autoencoder.save(out_dir, 'autoencoder_model_{}.pt'.format(epoch))

# obtain review embeddings
batches = batchify(text, 1, dictionary, shuffle=False)
embed_matrix = np.zeros((len(text), args.emsize), dtype='float32')
for i in range(len(batches)):
    source, target, lengths = batches[i]
    output = autoencoder.encode(source, lengths)
    embed_matrix[i] = np.array(output.cpu().data.numpy())

# embedding compression
args.M = 6
args.K = 6
compressor = EmbeddingCompressor(args.M, args.K)
compressed_embed = compressor.train(embed_matrix)

# multi-criteria recommendation
data['code'] = compressed_embed
code = []
data = data.groupby("user_id").filter(lambda x: len(x) > 10)
users = list(set(data['user_id']))
data.index = range(len(data))
for i in range(len(data)):
    code.append([float(data.loc[i,'ratings'])] + data.loc[i,'code'])
data['code'] = code
reader = Reader(rating_scale=(1, 5))
subdata = Dataset.load_from_df(data[['user_id', 'business_id', 'code']], reader)
predictions = cross_validate(KNNBaseline(), subdata, measures=['PRECISION_1','RECALL_1','PRECISION_5','RECALL_5'], cv=5, verbose=True)
#predictions = cross_validate(SlopeOne(), subdata, measures=['PRECISION_1','RECALL_1','PRECISION_5','RECALL_5'], cv=5, verbose=True)
#predictions = cross_validate(CoClustering(), subdata, measures=['PRECISION_1','RECALL_1','PRECISION_5','RECALL_5'], cv=5, verbose=True)
#predictions = cross_validate(SVD(), subdata, measures=['PRECISION_1','RECALL_1','PRECISION_5','RECALL_5'], cv=5, verbose=True)