import numpy as np

def save_vocab(savefile, vocab):

	with open(savefile, 'w+') as f:
		for x in vocab:	
			for num in x:
				f.write(str(num) + ' ')
			f.write('\n')


def load_vocab(file):

	with open(file, 'r') as f:
		vecs = f.readlines()

		return np.array([[float(num) for num in v.split()] for v in vecs], dtype=np.float32)
