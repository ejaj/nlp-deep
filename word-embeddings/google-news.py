from gensim.models import KeyedVectors

filename = '../data/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print("result", result)
