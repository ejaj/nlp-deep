from gensim.models import word2vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')
sample = open("../data/alice.txt")
s = sample.read()
# Replaces escape character with space
f = s.replace("\n", " ")

data = []
for i in sent_tokenize(f):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)
# Print results
print("Cosine similarity between 'alice' " + "and 'wonderland' - CBOW : ", model1.wv.similarity('alice', 'wonderland'))
print("Cosine similarity between 'alice' " + "and 'machines' - CBOW : ", model1.wv.similarity('alice', 'machines'))

model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)

# Print results
print("Cosine similarity between 'alice' " + "and 'wonderland' - Skip Gram : ",
      model2.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " + "and 'machines' - Skip Gram : ", model2.wv.similarity('alice', 'machines'))
