from nltk.stem import PorterStemmer

"""
Stemming Example
A common stemming algorithm might reduce each of these words to the base "run", because it simply looks for common patterns and suffixes among words to trim them down. The process doesn't consider the context of the word in the sentence, leading to:

"running" → "run"
"runs" → "run"
"ran" → "ran" (Depending on the algorithm, some irregular forms might not be correctly stemmed)
"runner" → "run"
Here, the algorithm might not correctly handle "ran" due to its irregular conjugation, depending on the rules defined in the stemming algorithm.
"""

porter = PorterStemmer()
print(porter.stem("running"))
print(porter.stem("runs"))
print(porter.stem("ran"))
print(porter.stem("Communication"))
