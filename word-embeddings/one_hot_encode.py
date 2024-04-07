def one_hot_encode(text):
    words = text.split()
    vocab = set(words)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    one_hot_encoded = []
    for word in words:
        one_hot_vector = [0] * len(vocab)
        one_hot_vector[word_to_index[word]] = 1
        one_hot_encoded.append(one_hot_vector)
    return one_hot_encoded, word_to_index, vocab


text = 'Natural Language Toolkit, Language'
one_hot_encoded, word_to_index, vocabulary = one_hot_encode(text)

print("Vocabulary:", vocabulary)
print("Word to Index Mapping:", word_to_index)
print("One-Hot Encoded Matrix:")
for word, encoding in zip(text.split(), one_hot_encoded):
    print(f"{word}: {encoding}")
