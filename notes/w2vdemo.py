# from gensim.models import Word2Vec
import gensim as gs

model_file = 'big.w2v'

print 'load'
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = gs.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
print 'loaded'
print 'saving'
model.save(model_file)
print 'saved'
print 'load saved'
model.load(model_file)
print 'loaded saved'

dog = model['dog']
print(dog.shape)
print(dog[:10])

# Deal with an out of dictionary word: YouKnowNothingJohnSnow
if 'YouKnowNothingJohnSnow' in model:
    print(model['YouKnowNothingJohnSnow'].shape)
else:
    print('{0} is an out of dictionary word'.format('YouKnowNothingJohnSnow'))

# Some predefined functions that show content related information for given words
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

print(model.similarity('woman', 'man'))
