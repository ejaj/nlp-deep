from nltk import word_tokenize, pos_tag

sent = "NLTK is a leading platform for building Python programs to work with human language data."
tokenize = word_tokenize(sent)
pos_tag = pos_tag(tokenize)
print(pos_tag)