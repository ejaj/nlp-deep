from sklearn.feature_extraction.text import CountVectorizer

documents = ["the cat sat on the mat.",
             "the dog sat on the log",
             "the cat chased the dog"
             ]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
print("Bag-of-Words Matrix:")
print(X.toarray())
print("Vocabulary (Feature Names):", feature_names)
