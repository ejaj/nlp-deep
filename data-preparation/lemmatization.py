from nltk.stem import WordNetLemmatizer

"""
Lemmatization Example
Lemmatization involves deeper linguistic understanding. It would convert each form of the word not just to its stem but to the lemma, or base form of the word according to its dictionary definition. This process considers the part of speech and uses a complete vocabulary of the language to ensure the output is a valid word:

"running" (verb) → "run"
"runs" (verb) → "run"
"ran" (verb) → "run"
"runner" (noun) → "runner"
In lemmatization, "ran" is correctly recognized as a past tense of "run" and thus is lemmatized to "run". The word "runner" remains "runner" because it's already in its base noun form. Lemmatization correctly understands different grammatical forms and ensures that the output is a valid word in all cases, reflecting the intended meaning more accurately than stemming.
"""

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("plays", 'v'))
print(lemmatizer.lemmatize("played", 'v'))
print(lemmatizer.lemmatize("play", 'v'))
print(lemmatizer.lemmatize("playing", 'v'))
print(lemmatizer.lemmatize("Communication", 'v'))
