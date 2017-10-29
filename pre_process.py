import numpy as np

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

