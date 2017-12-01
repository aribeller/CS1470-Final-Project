import numpy as np
import gensim

def process_corp(corp, vocabulary, lt, max_key, max_sent):

	label_types = lt
	lindex = 0
	labels = []

	vocab = vocabulary
	vindex = max_key
	corpus = []

	for label, s in corp:
		if label not in label_types:
			label_types[label] = lindex
			lindex += 1

		labels.append(label_types[label])

		sentence = s.split()[1:]
		ind_sentence = []

		for word in sentence:
			if word not in vocab:
				vocab[word] = vindex
				vindex += 1

			ind_sentence.append(vocab[word])

		corpus.append(ind_sentence)


	max_slength = max(sorted([len(s) for s in corpus])[-1], max_sent)
	if '<PAD>' in vocab:
		pad_num = vocab['<PAD>']
	else:
		vocab['<PAD>'] = vindex
		pad_num = vindex
		vindex += 1

	corpus = [sent + [pad_num]*(max_slength-len(sent)) for sent in corpus]

	labels = np.array(labels)
	corpus = np.array(corpus)

	return corpus, labels, vocab, label_types, max_slength


# def process_corp_w2v(corp, vocabulary, max_sent, w2v, w2v_indicies):

# 	corpus = []
# 	vindex = 0
# 	labels = []
# 	label_types = {}
# 	lindex = 0

# 	# Loop through label sentence pairs in corpus
# 	for label, s in corp:
# 		# If the label is not yet seen, give it a class number add it to the
# 		# dictionary of label values
# 		if label not in label_types:
# 			label_types[label] = lindex
# 			lindex += 1

# 		# Append that label's class value to the sequence of labels
# 		labels.append(label_types[label])

# 		# Split the sentence on whitespace
# 		sent = s.split()
# 		# initialize the numerical representation
# 		index_sentence = []

# 		# For each word in the sentence
# 		for word in sent:
# 			# If the word is not in the vocabulary, assign it an index and add it
# 			if word not in vocabulary and word in w2v:
# 				vocabulary[word] = vindex
# 				vindex += 1

# 				# If the word is in word2vec, add it's w2v index to a list
# 				w2v_index = w2v[word]
# 				if word in w2v and w2v_index not in w2v_indicies:
# 					w2v_indicies.append(w2v_index)
# 				# Otherwise add the index for any word that does not appear in w2v
# 				else:
# 					w2v_indicies.append(w2v['<OUT_OF_VOCAB>'])

# 			# Add the numerical representation of the word to the sentence
# 			index_sentence.append(vocabulary[word])
		
# 		# Append that sentence
# 		corpus.append(index_sentence)

# 	# Determine what the length of the longest sentence is
# 	max_slen = max(max_sent, max([len(sent) for sent in corpus]))

# 	if '<PAD>' in vocabulary:
# 		pad_num = vocabulary['<PAD>']
# 	else:
# 		vocabulary['<PAD>'] = vindex
# 		pad_num = vindex
# 		vindex += 1

# 	corpus = [sent + [pad_num]*(max_slen-len(sent)) for sent in corpus]

# 	corpus = np.array(corpus)
# 	labels = np.array(labels)

# 	return corpus, labels, vocabulary, w2v_indicies, max_slen

def w2v_embed(vocab):

	w2v_filepath = 'GoogleNews-vectors-negative300.bin.gz'

	model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filepath, binary=True)

	embed = np.zeros((len(vocab),300), dtype=np.float32)

	for word,index in vocab.items():
		if word in model:
			embed[index,:] = model[word]
		else:
			embed[index,:] = np.random.normal(0,.1,(300))

	return embed





