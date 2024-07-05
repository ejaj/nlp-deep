import string
import random
from collections import defaultdict, Counter
import nltk
from nltk.corpus import reuters, stopwords
from nltk.util import ngrams
from nltk import FreqDist

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('reuters')

# Input the Reuters sentences
sents = reuters.sents()

# Write the removal characters such as: Stopwords and punctuation
stop_words = set(stopwords.words('english'))
all_punctuation = string.punctuation + '""' + '--' + "''" + '—'
removal_list = list(stop_words) + list(all_punctuation) + ['lt', 'rt']

unigram = []
bigram = []
trigram = []
tokenized_text = []

# Process sentences
for sentence in sents:
    # Lowercase and filter out unwanted tokens
    filtered_sentence = [word.lower() for word in sentence if word not in removal_list and word.isalpha()]
    tokenized_text.append(filtered_sentence)
    unigram.extend(filtered_sentence)
    bigram.extend(
        list(ngrams(filtered_sentence, 2, pad_left=True, pad_right=True, left_pad_symbol=None, right_pad_symbol=None)))
    trigram.extend(
        list(ngrams(filtered_sentence, 3, pad_left=True, pad_right=True, left_pad_symbol=None, right_pad_symbol=None)))


# Define function to remove n-grams with removable words
def remove_stopwords(ngrams_list):
    return [ngram for ngram in ngrams_list if not all(word in removal_list or word is None for word in ngram)]


unigram = remove_stopwords(unigram)
bigram = remove_stopwords(bigram)
trigram = remove_stopwords(trigram)

# Generate frequency of n-grams
freq_bi = FreqDist(bigram)
freq_tri = FreqDist(trigram)

# Create a defaultdict for next word prediction
d = defaultdict(Counter)
for (a, b, c), count in freq_tri.items():
    if a and b and c:  # Check if none of the elements is None
        d[(a, b)][c] += count


# Next Word prediction
def pick_word(counter):
    "Chooses a random element based on its frequency."
    if not counter:
        return None  # Return None or an appropriate default if no entries exist

    total = sum(counter.values())
    r = random.randint(1, total)
    for word, freq in counter.items():
        r -= freq
        if r <= 0:
            return word


# Start with a seed phrase
prefix = ("he", "said")  # Initial bigram to start predictions
print(" ".join(prefix))
s = " ".join(prefix)

for i in range(19):
    suffix = pick_word(d[prefix])
    if not suffix:  # If no suffix could be picked (i.e., no known continuation)
        print("No continuation found for:", prefix)
        break  # Exit the loop or you could choose to restart with a new random prefix
    s += ' ' + suffix
    print(s)
    prefix = (prefix[1], suffix)  # Move to the next bigram
