from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]
model = Word2Vec(sentences, min_count=1)
# print(model)
# words = list(model.wv.key_to_index)
# print(words)
# sentence_vector = model.wv['sentence']
# print(sentence_vector)
# model.save('model.bin')
# # load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
# Access the vectors for all words in the vocabulary
X = model.wv.vectors
# print(X)
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
# plt.show()
words = list(model.wv.key_to_index)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
