from nltk.tokenize import sent_tokenize, word_tokenize

# Example text
text = "Hello, world! How are you today? This is an example of sentence tokenization."

# Split text into sentences
sentences = sent_tokenize(text)

# Print each sentence
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")

# Split sentence into words
words = word_tokenize(text)
# Print each word
for word in words:
    print(word)
