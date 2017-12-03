import tensorflow as tf
import numpy as np
from pre_process2 import process_corp, w2v_embed
import time
import gensim
from vocab_saver import *

start = time.time()

w2v_filepath = 'GoogleNews-vectors-negative300.bin.gz'
training_filepath = 'TREC_training.txt'
testing_filepath = 'TREC_test.txt'

# w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_filepath, binary=True)
print("open training file")
with open(training_filepath) as f:
	training_pairs = [line.split(':', 1) for line in f.readlines()]

# print(training_pairs[0])
training_sentence_count = len(training_pairs)

print("open testing file")
with open(testing_filepath) as f:
	testing_pairs = [line.split(':', 1) for line in f.readlines()]

testing_sentence_count = len(testing_pairs)

print("preprocess training")
training_corpus, training_labels, vocab, label_types, max_sentence_length = process_corp(training_pairs, {}, {}, 0, 0, is_training=True)
print("preprocess testing")
testing_corpus, testing_labels, vocab, label_types, max_sentence_length = process_corp(testing_pairs, vocab, label_types, sorted(vocab.values())[-1], max_sentence_length)
