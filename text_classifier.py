import sys
import tensorflow as tf
import numpy as np
import time
import gensim
from pre_process2 import process_corp, w2v_embed
from vocab_saver import *
start = time.time()

w2v_filepath = 'GoogleNews-vectors-negative300.bin.gz'
training_data_filepath = 'TREC_training.txt'
testing_data_filepath = 'TREC_test.txt'

with open(training_data_filepath) as f:
	pairs_tr = [line.split(':', 1) for line in f.readlines()]

num_trsent = len(pairs_tr)

with open(testing_data_filepath) as f:
	pairs_te = [line.split(':', 1) for line in f.readlines()]

tr_c, tr_l, vocab, lt, sent_len = process_corp(pairs_tr, {}, {}, 0, 0, is_training=True)
te_c, te_l, vocab, lt,  sent_len = process_corp(pairs_te, vocab, lt, sorted(vocab.values())[-1], sent_len)

tr_snum = tr_c.shape[0]
te_snum = te_c.shape[0]

batch_sz = 50
vocab_sz = len(vocab)
embed_sz = 300
chnl_num = 1
num_flts = 100
num_class = 6
drop_prob = .5
num_iter = 10900
num_batches = tr_snum/batch_sz

print("create embed from w2v")
embed = w2v_embed(vocab)
print("save embed to file")
save_vocab('vocab_store.txt', embed)
print("load embed from file")
embed = load_vocab('vocab_store.txt')
print(embed.shape)


sent = tf.placeholder(tf.int32, [batch_sz, sent_len])
ans = tf.placeholder(tf.int32, [batch_sz])
# num_features = tf.placeholder(tf.int32)
p = tf.placeholder(tf.float32)

# E = tf.Variable(tf.truncated_normal(shape=[vocab_sz, embed_sz], stddev=.1))
E = tf.constant(embed, dtype=tf.float32)

flts3 = tf.Variable(tf.truncated_normal(shape=[3, embed_sz, chnl_num, num_flts], stddev=.1))
flts4 = tf.Variable(tf.truncated_normal(shape=[4, embed_sz, chnl_num, num_flts], stddev=.1))
flts5 = tf.Variable(tf.truncated_normal(shape=[5, embed_sz, chnl_num, num_flts], stddev=.1))

conv_bias3 = tf.Variable(tf.truncated_normal(shape=[num_flts],stddev=.1))
conv_bias4 = tf.Variable(tf.truncated_normal(shape=[num_flts],stddev=.1))
conv_bias5 = tf.Variable(tf.truncated_normal(shape=[num_flts],stddev=.1))

W = tf.nn.dropout(tf.Variable(tf.truncated_normal(shape=[3*num_flts,num_class],stddev=.1)), p)
b = tf.Variable(tf.truncated_normal(shape=[num_class],stddev=.1))


embed = tf.nn.embedding_lookup(E, sent)
r_embed = tf.reshape(embed, shape=[batch_sz, sent_len, embed_sz, 1])

conv3_out = tf.squeeze(tf.nn.relu(tf.nn.conv2d(r_embed, flts3, [1,1,1,1], 'VALID') + conv_bias3))
conv4_out = tf.squeeze(tf.nn.relu(tf.nn.conv2d(r_embed, flts4, [1,1,1,1], 'VALID') + conv_bias4))
conv5_out = tf.squeeze(tf.nn.relu(tf.nn.conv2d(r_embed, flts5, [1,1,1,1], 'VALID') + conv_bias5))

max1 = tf.reduce_max(conv3_out, axis=1)
max2 = tf.reduce_max(conv4_out, axis=1)
max3 = tf.reduce_max(conv5_out, axis=1)

# conv_out = tf.concat([conv3_out, conv4_out, conv5_out], 2)
max_act = tf.concat([max1,max2,max3], axis=1)

logits = tf.matmul(max_act, W) + b

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=logits))

tr = tf.train.AdamOptimizer(0.001).minimize(loss)

ses = tf.Session()
ses.run(tf.global_variables_initializer())


# ordering = np.random.permutation(tr_snum)

tr_loss = 0.0
dev_loss = 0.0
dev_prev = 100000
acc = 0
acc_prev = 0
for i in range(num_iter):
	if i%num_batches == 0:
		print 'Epoch', i/num_batches
		print 'Training Loss:', tr_loss
		tr_loss = 0.0
		dev_loss = 0.0
		for j in range(te_snum/batch_sz):
			dev_start = j*batch_sz
			dev_end = j*batch_sz+batch_sz
			dev_batch = te_c[np.arange(dev_start, dev_end), :]
			dev_labels = te_l[dev_start:dev_end]
			dev_loss += ses.run(loss,feed_dict={sent:dev_batch, ans:dev_labels, p:1.0})
		print 'Development Loss:', dev_loss
		print

		ordering = np.random.permutation(tr_snum)

		correct = 0.0
		for k in range(te_snum/batch_sz):
			test_start = k*batch_sz
			test_end = k*batch_sz+batch_sz
			test_batch = te_c[np.arange(test_start, test_end),:]
			test_labels = te_l[test_start:test_end]
			log = ses.run(logits, feed_dict={sent:test_batch, ans:test_labels, p:1.0})

			pred = np.argmax(log, axis=1)
			check = np.equal(pred, test_labels)
			c = np.sum(check)

			correct += c

		prev_acc = acc
		acc = correct/te_snum

	if dev_prev < dev_loss and i/num_batches > 10:
		print 'Accuracy:'
		print prev_acc
		break

	dev_prev  = dev_loss

	index_loc = i%num_batches

	tr_start = index_loc*batch_sz
	tr_end = index_loc*batch_sz + batch_sz
	indicies = ordering[tr_start:tr_end]

	tr_batch = tr_c[indicies,:]
	tr_labels = tr_l[indicies]

	l, _ = ses.run([loss,tr], feed_dict={sent:tr_batch, ans:tr_labels, p:.5})
	tr_loss += l






# print 'Accuracy:'
# print correct/te_snum

print 'Runtime:'
print time.time() - start








# conv3 = tf.layers.conv2d(r_embed, filters=100, kernel_size=[3,embed_sz],
#  strides=(1,0), padding='valid',use_bias=True,
#  bias_initializer=tf.truncated_normal(shape=[num_features],stddev=.1))

# conv4 = tf.layers.conv2d(r_embed, filters=100, kernel_size=[4,embed_sz],
#  strides=(1,0), padding='valid',use_bias=True,
#  bias_initializer=tf.truncated_normal(shape=[num_features],stddev=.1))

# conv5 = conv5 = tf.layers.conv2d(r_embed, filters=100, kernel_size=[5,embed_sz],
#  strides=(1,0), padding='valid',use_bias=True,
#  bias_initializer=tf.truncated_normal(shape=[num_features],stddev=.1))



# conv_bias = tf.Variable(tf.truncated_normal(shape=[tr_slen]))
