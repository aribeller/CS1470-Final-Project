import numpy as np
import gensim

def get_unk_symbol():
	return "*UNK*"

def get_pad_symbol():
	return "<PAD>"

def process_corp(pairs, vocabulary, lt, max_key, previous_max_sentence_length, is_training=False):
	label_types = lt
	lindex = 0
	indexed_labels = []

	vocab = vocabulary
	vindex = max_key
	indexed_corpus = []
	orig_corpus = []

	word_occurrences = {}
	word_count = 0 # just for stats

	for label, s in pairs:
		# only add to labels if it's training.
		# if testing then the labels should already be there
		if is_training and label not in label_types:
			label_types[label] = lindex
			lindex += 1

		indexed_labels.append(label_types[label])

		# extract the proper sentence without preceding word
		sentence = s.split()[1:]

		# save the original words sentences for later
		orig_corpus.append(sentence)

	# perform unking based on whether it's training or testing data
	if is_training:
		# count the word occurences to see which to unk
		for sentence in orig_corpus:
			for word in sentence:
				word_count += 1
				if word not in word_occurrences:
					word_occurrences[word] = 0
				word_occurrences[word] += 1

		# unk words that have 1 occurence
		for sentence in orig_corpus:
			for i in range(len(sentence)):
				word = sentence[i]
				if word_occurrences[word] == 1:
					sentence[i] = get_unk_symbol()
	else: # is testing
		# unk words not in vocab
		for sentence in orig_corpus:
			for i in range(len(sentence)):
				word = sentence[i]
				if word not in vocab:
					sentence[i] = get_unk_symbol()

	# pad sentences, don't override the original sentence lengths, i think we need them later
	print(sorted([len(s) for s in orig_corpus])[-1])
	max_sentence_length = max(sorted([len(s) for s in orig_corpus])[-1], previous_max_sentence_length)
	padded_corpus = [s + [get_pad_symbol()] * (max_sentence_length - len(s)) for s in orig_corpus]

	# perform indexing after unking and padding
	for padded_sentence in padded_corpus:
		indexed_sentence = []
		for word in padded_sentence:
			# only add to vocab if it's for training set
			# word should already be in vocab, or unked, if it's
			if is_training and word not in vocab:
				vocab[word] = vindex
				vindex += 1
			indexed_sentence.append(vocab[word])
		indexed_corpus.append(indexed_sentence)

	indexed_labels = np.array(indexed_labels)
	indexed_corpus = np.array(indexed_corpus)

	# print(indexed_labels[:5])
	# print(indexed_corpus[:5])

	return indexed_corpus, indexed_labels, vocab, label_types, max_sentence_length

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



def process_sst_1(filepaths, vocabulary, lt, max_key, previous_max_sentence_length, is_training=False):
	corpus_filepath, phrases_filepath, sentiments_filepath = filepaths

	with open(training_data_filepath) as f:
		lines = [line.split(':', 1) for line in f.readlines()]

	label_types = lt
	lindex = 0
	indexed_labels = []

	vocab = vocabulary
	vindex = max_key
	indexed_corpus = []
	orig_corpus = []

	word_occurrences = {}
	word_count = 0 # just for stats

	for line in lines:
		sentiment = phrases[line]
		label = get_label(sentiment)

def get_dataset_filepaths(dataset_to_use):
	training_data_filepath = ""
	testing_data_filepath = ""

	TREC_training_data_filepath = 'TREC_training.txt'
	TREC_testing_data_filepath = 'TREC_test.txt'
	SST_1_training_data_filepath = 'sst_1_train.txt'
	SST_1_testing_data_filepath = 'sst_1_dev.txt'
