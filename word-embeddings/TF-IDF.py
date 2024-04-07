from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog"
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
tfidf_values = {}

# # Convert the TF-IDF matrix into an array and display it
# print(tfidf_matrix.toarray())
# # Display the vocabulary (words/features)
# print(vectorizer.get_feature_names_out())
for doc_index, doc in enumerate(documents):
    feature_index = tfidf_matrix[doc_index, :].nonzero()[1]
    tfidf_doc_values = zip(feature_index, [tfidf_matrix[doc_index, x] for x in feature_index])
    tfidf_values[doc_index] = {feature_names[i]: value for i, value in tfidf_doc_values}
# let's print
for doc_index, values in tfidf_values.items():
    print(f"Document {doc_index + 1}:")
    for word, tfidf_value in values.items():
        print(f"{word}: {tfidf_value}")
    print("\n")
