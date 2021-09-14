from pymongo import MongoClient
from gensim.utils import tokenize
from gensim.models import word2vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import gensim

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
api_key="e98d289c6ad0c43d488bca69ab2466ff"
base_url="https://api.themoviedb.org/3/"

doc2 = []
host = '127.0.0.1'  # or localhost
port = 27017
client = MongoClient(host, port)

db = client['oumar']
collection = db["movie"]

for i in (collection.find({"name": "oumar"})):
    doc2.append(i)
doc2 = doc2[0]

i=0
sentance=[]
while i <len(doc2["content"]):
    s=doc2["content"][i]["overview"]
    print(s)
    sentance.append(list(tokenize(s, deacc=True, lower=True)))
    i=i+1
print("sentances", sentance)

#construction de mon model
model = word2vec.Word2Vec(sentance, size=300, window=20,
                              min_count=2, workers=1, iter=100)
# Load Google's pre-trained Word2Vec model.
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/do/Téléchargements/GoogleNews-vectors-negative300.bin', binary=True)
def trouve_mot_similaire(phrase):
    sim=model.predict_output_word(phrase,topn=3)
    res_sim=[sim[0][0],sim[1][0],sim[2][0]]
    return res_sim
if __name__ == '__main__':
    print("overvieww: ",doc2["content"][0]["overview"])
    i=0
    sentance=[]
    while i <len(doc2["content"]):
        s=doc2["content"][i]["overview"]
        print(s)
        sentance.append(list(tokenize(s, deacc=True, lower=True)))
        i=i+1
    print("sentances", sentance)

    #construction de mon model
    model = word2vec.Word2Vec(sentance, size=300, window=20,
                              min_count=2, workers=1, iter=100)

    print(model['woman'])
    #voir le pourcentage qui dit un mot ressemble à un autre
    print("similarity between Man and Brother",model.similarity("man","brother"))
    print(model.most_similar("man"))
    print('predict_out_word : he sets out to raise *** daughter on *** own.'  )
    p=model.predict_output_word(["he","sets", "out","to","raise","daughter","on"," own"])
    print(p)
    print(trouve_mot_similaire('like '))

    tsne_plot(model)




