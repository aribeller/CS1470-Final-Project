import sys
import tensorflow as tf
import numpy as np
import time
import gensim
from pre_process2 import *
from vocab_saver import *
start = time.time()



w2v_filepath = 'GoogleNews-vectors-negative300.bin.gz'

# Grab training set, testing set, and embeddings. Determine whether new embeddings
# must be made and saved and also what model to train
training_data_filepath = sys.argv[1]
testing_data_filepath = sys.argv[2]
embeddings_filepath = sys.argv[3]
to_make = sys.argv[4] # either MAKE or USE
method = sys.argv[5] # either RAND, STATIC, NONSTATIC or MULTI

print("TRAINING ON: %s" % training_data_filepath)
print("TESTING ON: %s" % testing_data_filepath)

# Take in training and testing data. 
with open(training_data_filepath) as f:
	pairs_tr = [line.split(':', 1) for line in f.readlines()]

num_trsent = len(pairs_tr)

with open(testing_data_filepath) as f:
	pairs_te = [line.split(':', 1) for line in f.readlines()]

# Pre-process corpus
tr_c, tr_l, vocab, lt, sent_len, sent_lens = process_corp(pairs_tr, {}, {}, 0, 0, is_training=True)
te_c, te_l, vocab, lt, sent_len, sent_lens = process_corp(pairs_te, vocab, lt, sorted(vocab.values())[-1], sent_len)

tr_snum = tr_c.shape[0]
te_snum = te_c.shape[0]

# Set hyperparameters
batch_sz = 50
vocab_sz = len(vocab)
embed_sz = 300
chnl_num = 1
num_flts = 100
num_class = len(lt)
drop_prob = .5
num_iter = 100000000
num_batches = tr_snum/batch_sz
patience = 5
hidden_sz = 500

# If necessary, obtain and save word embeddings from word2vec
if to_make == "MAKE":
	print("create embed from w2v")
	embed = w2v_embed(vocab)
	print("save embed to file")
	save_vocab(embeddings_filepath, embed)
print("load embed from file")
embed = load_vocab(embeddings_filepath)
print(embed.shape)

# Take in batch, labels, and dropout percentage
sent = tf.placeholder(tf.int32, [batch_sz, sent_len])
ans = tf.placeholder(tf.int32, [batch_sz])
p = tf.placeholder(tf.float32)

# Determine model being used
if method == "MULTI":
	E1 = tf.reshape(tf.constant(embed, dtype=tf.float32), (vocab_sz,embed_sz,1))
	E2 = tf.reshape(tf.Variable(embed,dtype=tf.float32), (vocab_sz,embed_sz,1))
	E = tf.concat([E1,E2],axis=2)
	chnl_num = 2
elif method == "STATIC":
	E = tf.constant(embed, dtype=tf.float32)
elif method == "NONSTATIC":
	E = tf.Variable(embed, dtype=tf.float32)
else: # method == "RAND"
	E = tf.Variable(tf.truncated_normal(shape=[vocab_sz, embed_sz], stddev=.1))

# 3 groups of filters, one 3 tall, one 4 tall, and one 5 tall
flts3 = tf.Variable(tf.truncated_normal(shape=[3, embed_sz, chnl_num, num_flts], stddev=.1))
flts4 = tf.Variable(tf.truncated_normal(shape=[4, embed_sz, chnl_num, num_flts], stddev=.1))
flts5 = tf.Variable(tf.truncated_normal(shape=[5, embed_sz, chnl_num, num_flts], stddev=.1))

# biases for convolutions
conv_bias3 = tf.Variable(tf.truncated_normal(shape=[num_flts],stddev=.1))
conv_bias4 = tf.Variable(tf.truncated_normal(shape=[num_flts],stddev=.1))
conv_bias5 = tf.Variable(tf.truncated_normal(shape=[num_flts],stddev=.1))

# two linear units for the final classification. Regularize with dropout and clipped norm.
W0 = tf.clip_by_norm(tf.nn.dropout(tf.Variable(tf.truncated_normal(shape=[3*num_flts,hidden_sz],stddev=.1)),p), 3)
b0 = tf.Variable(tf.truncated_normal(shape=[hidden_sz],stddev=.1))
W = tf.clip_by_norm(tf.nn.dropout(tf.Variable(tf.truncated_normal(shape=[hidden_sz,num_class],stddev=.1)), p), 3)
b = tf.Variable(tf.truncated_normal(shape=[num_class],stddev=.1))

# Grab embeddings from E matrix. Shape for convolutions
embed = tf.nn.embedding_lookup(E, sent)
r_embed = tf.reshape(embed, shape=[batch_sz, sent_len, embed_sz, chnl_num])

# Apple convolutions with bias and relu
conv3_out = tf.squeeze(tf.nn.relu(tf.nn.conv2d(r_embed, flts3, [1,1,1,1], 'VALID') + conv_bias3))
conv4_out = tf.squeeze(tf.nn.relu(tf.nn.conv2d(r_embed, flts4, [1,1,1,1], 'VALID') + conv_bias4))
conv5_out = tf.squeeze(tf.nn.relu(tf.nn.conv2d(r_embed, flts5, [1,1,1,1], 'VALID') + conv_bias5))

# Determine max activation
max1 = tf.reduce_max(conv3_out, axis=1)
max2 = tf.reduce_max(conv4_out, axis=1)
max3 = tf.reduce_max(conv5_out, axis=1)

# Put the activation vectors together
max_act = tf.concat([max1,max2,max3], axis=1)

# Feed the results through the linear units.
hidden = tf.nn.relu(tf.matmul(max_act, W0) + b0)
logits = tf.matmul(hidden, W) + b

# Calculate the loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=logits))

# Optimize
tr = tf.train.AdamOptimizer(0.001).minimize(loss)

# Initialize session
ses = tf.Session()
ses.run(tf.global_variables_initializer())



# Initialize loss and accuracy values
tr_loss = 0.0
dev_loss = 0.0
dev_prev = 100000
acc = 0
acc_prev = 0

# Iterate large number of times
for i in range(num_iter):

	# At each epoch
	if i%num_batches == 0:
		# Print epoch number
		print 'Epoch', i/num_batches
		# Print training loss aggregated across the epoch, then reset training loss
		print 'Training Loss:', tr_loss
		tr_loss = 0.0

		# Calculate development loss with current parameters
		dev_loss = 0.0
		for j in range(te_snum/batch_sz):
			# batch and caculate
			dev_start = j*batch_sz
			dev_end = j*batch_sz+batch_sz
			dev_batch = te_c[np.arange(dev_start, dev_end), :]
			dev_labels = te_l[dev_start:dev_end]
			dev_loss += ses.run(loss,feed_dict={sent:dev_batch, ans:dev_labels, p:1.0})
		print 'Development Loss:', dev_loss
		print

		# Permute the data
		ordering = np.random.permutation(tr_snum)

		# Calculate number correct given current parameters
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

		# save the previous accuracy and compute the new one
		prev_acc = acc
		acc = correct/te_snum

	# If the previous dev loss is less than the new dev loss and we have passed
	# patience epoch, print the previous accuracy and break loop.
	if dev_prev < dev_loss and i/num_batches > patience:
		print 'Accuracy:'
		print prev_acc
		break

	# Save the now old dev loss
	dev_prev  = dev_loss

	# index and batch
	index_loc = i%num_batches

	tr_start = index_loc*batch_sz
	tr_end = index_loc*batch_sz + batch_sz
	indicies = ordering[tr_start:tr_end]

	tr_batch = tr_c[indicies,:]
	tr_labels = tr_l[indicies]

	# train the net
	l, _ = ses.run([loss,tr], feed_dict={sent:tr_batch, ans:tr_labels, p:.5})
	tr_loss += l







# Print runtime
print 'Runtime:'
print time.time() - start









