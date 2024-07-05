from nltk.util import pad_sequence, bigrams, ngrams, everygrams
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer

# text = [
#     ['a', 'b', 'c'],
#     ['a', 'c', 'd', 'c', 'e', 'f']
# ]
# print(list(bigrams(text[0])))
#
# print(list(ngrams(text[1], n=3)))
#
# print(list(pad_sequence(text[0],
#                         pad_left=True, left_pad_symbol="<s>",
#                         pad_right=True, right_pad_symbol="</s>",
#                         n=2)))
# padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>",
#                                 pad_right=True, right_pad_symbol="</s>", n=2))
# print(list(ngrams(padded_sent, n=2)))
# print(list(pad_sequence(text[0],
#                         pad_left=True, left_pad_symbol="<s>",
#                         pad_right=True, right_pad_symbol="</s>",
#                         n=3)))
# padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>",
#                                 pad_right=True, right_pad_symbol="</s>", n=3))
# print(list(ngrams(padded_sent, n=3)))

# print(list(pad_both_ends(text[0], n=2)))
# print(list(bigrams(pad_both_ends(text[0], n=2))))

# padded_bigrams = list(pad_both_ends(text[0], n=2))
# print(padded_bigrams)
# print(list(everygrams(padded_bigrams, max_len=2)))
# print(list(flatten(pad_both_ends(sent, n=2) for sent in text)))
# train, vocab = padded_everygram_pipeline(2, text)
# for ngrams in train:
#     print(list(ngrams))
# training_ngrams, padded_sentences = padded_everygram_pipeline(2, text)
# for ngramlize_sent in training_ngrams:
#     print(list(ngramlize_sent))
#     print()
# print('#############')
# print(list(padded_sentences))

try:  # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize

    # Testing whether it works.
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except Exception as e:  # Catch exceptions explicitly and possibly log them
    import re
    from nltk.tokenize import ToktokTokenizer

    # Explanation and regex source:
    # https://stackoverflow.com/a/25736515/610569
    # This regex splits on spaces that follow a period/question mark/exclamation mark
    # and a non-uppercase letter, before an uppercase letter, which likely indicates
    # the start of a new sentence.
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)

    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = toktok.tokenize  # Fixed the duplicate assignment here

import os
import requests
import io  # codecs

# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)
tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                  for sent in sent_tokenize(text)]
# print(tokenized_text[0])
# print(text[:500])

# Preprocess the tokenized text for 3-grams language modelling
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
model = MLE(n)
print(len(model.vocab))

model.fit(train_data, padded_sents)
print(model.vocab)
print(len(model.vocab))
print(model.vocab.lookup(tokenized_text[0]))
print(model.vocab.lookup('language is never random lah .'.split()))
print(model.counts)
print(model.counts['language'])  # i.e. Count('language')
print(model.counts[['language']]['is'])  # i.e. Count('is'|'language')
print(model.counts[['language', 'is']]['never'])  # i.e. Count('never'|'language is')

print(model.score('language'))  # P('language')
print(model.score('is', 'language'.split()))  # P('is'|'language')
print(model.score('never', 'language is'.split()))  # P('never'|'language is')

print(model.score("<UNK>") == model.score("lah"))
print(model.score("<UNK>") == model.score("leh"))
print(model.score("<UNK>") == model.score("lor"))
print(model.logscore("never", "language is".split()))
print(model.generate(20, random_seed=7))

detokenize = TreebankWordDetokenizer().detokenize


def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


print(generate_sent(model, 20, random_seed=7))
print(model.generate(28, random_seed=0))

print(generate_sent(model, 28, random_seed=0))
print(generate_sent(model, 20, random_seed=1))
print(generate_sent(model, 20, random_seed=30))
print(generate_sent(model, 20, random_seed=42))

# Save the model
import dill as pickle

#
# with open('kazi_ngram_model.pkl', 'wb') as fout:
#     pickle.dump(model, fout)
with open('kazi_ngram_model.pkl', 'rb') as fin:
    model_loaded = pickle.load(fin)
print(generate_sent(model_loaded, 20, random_seed=42))
