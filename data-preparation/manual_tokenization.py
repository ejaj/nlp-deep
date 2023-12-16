import re
import string

filename = '../data/metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()

# split based on words only
# words = re.split(r'\W+', text)
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
stripped = [re_punc.sub('', w) for w in words]
# print(stripped[:100])

re_print = re.compile('[^%s]' % re.escape(string.printable))
result = [re_print.sub('', w) for w in words]
# print(result[:100])

# Normalizing Case

# convert to lower case
words = [word.lower() for word in words]
print(words[:100])
