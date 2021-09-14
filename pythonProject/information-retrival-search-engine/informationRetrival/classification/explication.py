from sklearn import feature_extraction
dictionary=["foot","ball"]
dictionary=["life","game"]
if __name__ == '__main__':
    print("Dictionary", dictionary)
    print("shape", len(dictionary))
dictionary = filter(None, list(set(dictionary)))
t=["foot ball but","basket ball", "foot"]
d=["life is an horrible life game","game over","my life"]
vec_1 = feature_extraction.text.CountVectorizer(vocabulary=dictionary)
if __name__ == '__main__':
    X = vec_1.fit_transform(d).toarray()
    print("la matrice x", X)
