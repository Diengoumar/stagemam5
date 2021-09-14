#coding=utf-8
import imp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import string
import numpy as np
import sys
#sys.path.append(['/home/ubuntu/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/classification'])
#import lemmatization
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#from sklearn.externals import joblib
import joblib
#sys.modules['sklearn.externals.joblib'] = joblib
from sklearn import feature_extraction
from pymongo import MongoClient
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import sys
import  pandas as pd
import requests
from PIL import Image
from io import BytesIO
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
imp.reload(sys)

class lemmatization(object):
    def __init__(self):

        self.lmtzr = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def removeStopWords(self, words):
        """ Removes the stopwords from the given list of words.
        :param words: List of all the words
        :return: the unique words in a list of words after removing stop words
        """
        line = []
        for w in words:
            if w not in self.stop_words:
                line.append(w)
        return line

    def getBiwords(self, words):
        """ Removes all the biwords from the given sentence.
        :param words: List of all the words
        :return: the bigrams from the sentence provided
        """
        bigrams_val = nltk.bigrams(words)
        biwords = []
        for word in bigrams_val:
            biwords.append(word)
        return biwords

    def lemmatizeWord(self, lst):
        """ Lemmatize the list of words.
        :param words: List of all the words
        :return: the lemmatized version of the words
        """
        lemmatized_list = []
        for item in lst:
            lemmatized_list.append(self.lmtzr.lemmatize(item))
        return lemmatized_list

class Classification(object):
    path=''
    host = '127.0.0.1'  # or localhost
    port = 27017
    client = MongoClient(host, port)
    # 创建数据库dialog
    db = client['allMovies']
    # 创建集合scene
    collection = db["Movie"]
    def __init__(self,path):
        self.path=path

    def Train(self):
        """
        Function to train data set
        """
        
        lem = lemmatization()
        #Get Mongo client
        client = MongoClient()
        db = client['IR']
        collection = db['Movies']
        print("collection: ",collection)
        host = '127.0.0.1'  # or localhost
        port = 27017
        client = MongoClient(host, port)
        # # 创建数据库dialog
        db = client['allMovies']
        # # 创建集合scene
        #collection = db["Movie"]
        print(collection.__sizeof__())
        print(collection.find_one({"content.genres.name":"Drama"}))

        #Path to folder to store trained data set
        path=self.path
        #Queries to get 500 horror, romance and crime movies
        qr1= self.collection.find({"content.genres.name":"Horror"})#.limit(200)
        qr2= self.collection.find({"content.genres.name":"Romance"})#.limit(200)
        qr3= self.collection.find({"content.genres.name":"Crime"})#.limit(200)
        qr4 = self.collection.find({"content.genres.name": "Comedy"})#.limit(200)
        print (qr1)
        #Combine queries
        query_results=[]
        for rec in qr1:
            query_results.append(rec)
        for rec in qr2:
            query_results.append(rec)
        for rec in qr3:
            query_results.append(rec)
        for rec in qr4:
            query_results.append(rec)
        print("queryyy",query_results)
        #Dictionary to store the terms appearing in the genres
        dictionary = []

        #List to store category of each record
        categories = []
        
        training_data = []
        #Document ids of records to be trained
        doc_ids = []
        a=0
        for movie in query_results:
            training_data.append(movie['content']['overview'])
            doc_ids.append(movie['_id'])
            for genre in movie['content']['genres']:
                print("genre ",genre['name'])
                a = a + 1
                if ((genre['name']=='Horror') or (genre['name']=='Romance') or (genre['name']=='Crime') or genre['name']=='Comedy') and a<=160:
                   categories.append(genre['name'])



                
            #Convert to lower case and remove stop words from overview
            dict_rec = movie['content']['overview'].lower()
            #table = maketrans(string.punctuation, " ")
            for s in string.punctuation:
                dict_rec = dict_rec.replace(s, "")
            #dict_rec = str(dict_rec).translate(string.punctuation)
            dict_rec = lem.removeStopWords(dict_rec.split(" "))

            #Add to dictionary
            if dict_rec not in dictionary:
                dictionary.extend(dict_rec)
        print("Dictionary", dictionary)
        print("shape", len(dictionary))
        dictionary = filter(None, list(set(dictionary)))

        #Store dictionary in a file
        joblib.dump(dictionary, path + "_Genre_Dictionary")
        
        #Store doc ids of trained data in a file
        myfile = open(r'doc_ids.pkl', 'wb')
        pickle.dump(doc_ids,myfile)
        myfile.close()

        #Initialize training models
        mod_1 = SVC(kernel='linear', C=1, gamma=1)
        mod_2 = LogisticRegression()
        mod_3 = GaussianNB()
        mod_4 = MultinomialNB()
        mod_5 = BernoulliNB()

        #Ensemble classifiers
        mod_6 = RandomForestClassifier(n_estimators=50)
        mod_7 = BaggingClassifier(mod_2, n_estimators=50)
        mod_8 = GradientBoostingClassifier(loss='deviance', n_estimators=100)

        mod_9 = VotingClassifier(
            estimators=[("SVM", mod_1), ("LR", mod_2), ("Gauss", mod_3), ("Multinom", mod_4), ("Bernoulli", mod_5),
                        ("RandomForest", mod_6), ("Bagging", mod_7), ("GB", mod_8)], voting='hard')
        mod_10 = VotingClassifier(estimators=[("SVM", mod_1), ("LR", mod_2), ("Multinom", mod_4),("Bernoulli", mod_5),("Bagging",mod_7)], voting='hard',weights=[1, 2, 3, 2,1])

        #Vectorizers for feature extraction
        vec_1 = feature_extraction.text.CountVectorizer(vocabulary=dictionary)
        vec_2 = feature_extraction.text.TfidfVectorizer(vocabulary=dictionary)
        
        vec_list = [vec_1, vec_2]
        vec_list=[vec_1]
        #List of training models
        model_list = [mod_1, mod_2, mod_3, mod_4, mod_5, mod_6, mod_7, mod_8, mod_9, mod_10]
        
        models_used = ["SVM", "LOGISTIC REGRESSION", "GAUSSIAN NB",
                      "MULTINOMIAL NB", "BERNOULLI NB", "RANDOM FOREST", "BAGGING", "GRADIENT",
                      "Voting", "Voting With Weights"]

        vec_used = ["COUNT VECTORIZER", "TFIDF VECTORIZER"]

        print("Starting training. This might take a while...")
        b=1
        #Start training
        for model in range(0, len(model_list)):
            a=1
            for vec in range(0, len(vec_list)):
                mod = model_list[model]
                vector = vec_list[vec]
                print("tour",a,b)
                print("taille training : ", (np.shape(training_data)))
                print(training_data)
                print(vector)
                #print("fit_tarnsform", vector.fit_transform(training_data))
                X = vector.fit_transform(training_data).toarray()
                #print("x",X)
                print("shape X",np.shape(X))
                print (np.shape(categories))
                #categories.reshape((80, 2))
                # l=[]
                # l.append([categories[0:79],categories[79:,159]])
                # print(l)
                print("categories",categories)

                print(np.unique(categories))
                print(np.unique(X))
                mod.fit(X, categories)
                print("fiit",mod.fit(X, categories))

                #Store in a file
                joblib.dump(mod, path + models_used[model] + "_" + vec_used[vec] + ".pkl")
                
                print(models_used[model] + " " + vec_used[vec]+" finished!")
                a=a+1
            b=b+1
            break
        print("All Done!!")
        
    def Classify_Data(self):
        """
        Function to classify data from the database.
        Prints results of classification
        """

        lem = lemmatization()

        #Get Mongo Client
        client = MongoClient()
        db = client['allMovies']
        collection = db['Movies']


        #Path to folder containing the training model files
        path=self.path

        #Get the list of doc ids trained
        trained_docs=[]

        #Mongo queries to retrieve Horror, Romance and Crime movies
        qr1 = self.collection.find({"content.genres.name":"Horror"})
        qr2 = self.collection.find({"content.genres.name":"Romance"})
        qr3 = self.collection.find({"content.genres.name":"Crime"})
        qr4 = self.collection.find({"content.genres.name": "Comedy"})
        print ("111")
        print (qr3)

        myfile = open('doc_ids.pkl', 'rb')
        trained_docs = pickle.load(myfile)
        #Get 100 Horror, Romance and Crime movies each, which are not in the trained data set
        
        horr=[]
        i=0
        for rec in qr1:
            if rec['_id'] not in trained_docs:
               i=i+1
               horr.append(rec)
               
            if i>=333:
                break
        rom=[]
        i=0
        for rec in qr2:
            if rec['_id'] not in trained_docs:
               i=i+1
               rom.append(rec)
              
            if i>=333:
                break

        crime=[]
        i=0
        for rec in qr3:
            if rec['_id'] not in trained_docs:
               i=i+1
               crime.append(rec)
               
            if i>=334:
                break
        comedy = []
        i = 0
        for rec in qr4:
            if rec['_id'] not in trained_docs:
                i = i + 1
                comedy.append(rec)

            if i >= 334:
                break

        #Combine the query results
        query_results=[]
        for rec in horr:
            query_results.append(rec)
        for rec in rom:
            query_results.append(rec)
        for rec in crime:
            query_results.append(rec)
        print (query_results)
        #Data to be classified
        test_data = []

        #Genres of records to be classified 
        categories = []
        a=0
        for movie in query_results:
            test_data.append(movie['content']['overview'])
            for genre in movie['content']['genres']:
                a=a+1
                if ((genre['name']=='Horror') or (genre['name']=='Romance') or (genre['name']=='Crime')or (genre['name']=='Comedy') and a<=80):
                   categories.append(genre['name'])



        #Lists of training models and vectorizers
        models = ["SVM", "LOGISTIC REGRESSION", "GAUSSIAN NB",
                      "MULTINOMIAL NB", "BERNOULLI NB", "RANDOM FOREST", "BAGGING", "GRADIENT",
                      "Voting","Voting With Weights"]

        vectorizers = ["COUNT VECTORIZER", "TFIDF VECTORIZER"]

        #Load dictionary containing terms appearing in genres
        dictionary = joblib.load(path + "_Genre_Dictionary")
        
        vec_1 = feature_extraction.text.CountVectorizer(vocabulary=dictionary)
        vec_2 = feature_extraction.text.TfidfVectorizer(vocabulary=dictionary)
        vec_list = [vec_1, vec_2]

        #List to store the classification stats for each model
        stats = []
        #Generate results
        for i in range(0, len(models)):
            for j in range(0, len(vectorizers)):
                time0=time.process_time()
                model = joblib.load(path + models[i] + "_" + vectorizers[j].replace('-', '') + ".pkl")
                vec = vec_list[j]
                Y = vec.fit_transform(test_data).toarray()
                print("y",Y)
                predicted_genres = model.predict(Y)
                k = 0
                horror = 0
                romance = 0
                crime = 0

                #Keeps track of correct predictions
                y_correct = []

                #Keeps track of incorrect predictions
                y_predicted = []
                for pred in predicted_genres:
                    if (categories[k] == "Horror"):
                        if (pred == "Horror"):
                            horror += 1
                            y_predicted.append(0)
                        elif (pred == "Romance"):
                            y_predicted.append(1)
                        else:
                            y_predicted.append(2)
                        y_correct.append(0)
                    elif (categories[k] == "Romance"):
                        if (pred == "Romance"):
                            romance += 1
                            y_predicted.append(1)
                        elif (pred == "Horror"):
                            y_predicted.append(0)
                        else:
                            y_predicted.append(2)
                        y_correct.append(1)
                    elif (categories[k] == "Crime"):
                        if (pred == "Crime"):
                            crime += 1
                            y_predicted.append(2)
                        elif (pred == "Horror"):
                            y_predicted.append(0)
                        else:
                            y_predicted.append(1)
                        y_correct.append(2)
                    k = k + 1

                #Print results
                score = precision_recall_fscore_support(y_correct, y_predicted, average='weighted')
                #print("Number of records classified per second = %d" % (round((1000/(time.process_time()-time0)),3)))
                print("________SCORES__________")
                print("MODEL      :  " + models[i])
                print("VECTORIZER :  " + vectorizers[j])
                print("Horror     :  %d/333" % (horror))
                print("Romance    :  %d/333" % (romance))
                print("Crime      :  %d/334" % (crime))
                print("Precision  :  %.5f" % (score[0]))
                print("Recall     :  %.5f" % (score[1]))
                print("F(1) Score :  %.5f" % ((score[1] * score[0] / (score[1] + score[0])) * 2))
                print("F(W) Score :  %.5f" % (score[2]))
                print("Accuracy   :  %.5f" % accuracy_score(y_correct, y_predicted))
                #print(confusion_matrix(y_correct, y_predicted))
                
                dic={}
                dic['model']=models[i].title()
                dic['vectorizer']=vectorizers[j][:-11]
                dic['horror']=str(horror)+'/'+'333'
                dic['romance']=str(romance)+'/'+'333'
                dic['crime']=str(crime)+'/'+'334'
                dic['precision']=round(score[0], 3)
                dic['Recall']=round(score[1], 3)
                dic['F(1) Score']=round(((score[1] * score[0] / (score[1] + score[0])) * 2), 3)
                dic['F(W) Score']=round(score[2], 3)
                dic['accuracy']=round(accuracy_score(y_correct, y_predicted), 3)
                stats.append(dic)
        #Store stats in file        
        joblib.dump(stats, path + "classification_results.txt")

        print("Done")
        return stats
    

    def Classify_Text(self,overview):
        """
        Function takes in the overview of a movie as input from the user and classifies the text
        """

        #convert text to lower case 
        overview = overview.lower()
        
        path=self.path        

        #start time
        time0 = time.process_time()

        #Use ensemble classifier - voting with weights

        #model = joblib.load(path + "MULTINOMIAL NB_TFIDF VECTORIZER" + ".pkl")
        model=joblib.load("/home/do/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/frontend/static/frontend/text/SVM_COUNT VECTORIZER.pkl")
        dictionary = joblib.load(path + "_Genre_Dictionary")
        vec = feature_extraction.text.CountVectorizer(vocabulary=dictionary)

        print(vec)
        #overview="An undercover cop and a mole in the police"
        Y = vec.fit_transform([overview]).toarray()
        print(vec.get_feature_names())
        print (Counter(Y[0]))
        #print(Counter(Y[1]))
        print(model)
        predicted_genre = model.predict(Y)
        print (predicted_genre)

        
        #Return predicted genre and time taken for classification
        return predicted_genre, str(round(time.process_time() - time0, 3)) + " seconds"


    def get_classification_results(self):
        """
        This functions returns a data structure containing the results of classification
        """
        try:
            path=self.path
            print(path + "classification_results.txt")
            results = joblib.load(path + "classification_results.txt")
            print(results)
            return results
        
        #Call Classify_Data() if results are not found
        except EOFError as eoferror:
            print ("Classification results not found. Generating results...")
            return self.Classify_Data()
        except IOError as ioerror:
            print ("Classification results not found. Generating results...")
            return self.Classify_Data()
# class Net(Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.cnn_layers = Sequential(
#             # Defining a 2D convolution layer
#             Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.linear_layers = Sequential(
#             Linear(12060, 21)
#         )
#
#     # Defining the forward pass
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x
#
dd = pickle.load(open('/home/do/Téléchargements/newdata/label_newdata.pkl', 'rb'))

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(0.4)
    self.batchnorm1 = nn.BatchNorm2d(16)
    self.batchnorm2 = nn.BatchNorm2d(32)
    self.batchnorm3 = nn.BatchNorm2d(64)
    self.fc1 = nn.Linear(64 * 5 * 5, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 25)

  def forward(self, x):
    x = self.batchnorm1(F.relu(self.conv1(x)))
    x = self.batchnorm2(F.relu(self.conv2(x)))
    x = self.dropout(self.batchnorm2(self.pool(x)))
    x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
    x = self.dropout(self.conv4(x))
    x = x.view(-1, 64 * 5 * 5)  # Flatten layer
    x = self.dropout(self.fc1(x))
    x = self.dropout(self.fc2(x))
    x = F.log_softmax(self.fc3(x), dim=1)
    return x


def model():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 26))
        #nn.Linear(128, len(np.unique(dd))))
    return model
def label(i):

    d = np.unique(dd)
    return d[i]
def vgg():
    model = models.vgg16(pretrained=True)

    # num_classes = 3

    # Freezing the weights
    for param in model.parameters():
        param.required_grad = False

    # Replacing the final layer
    model.fc = nn.Sequential(nn.Linear(512, 256),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(256, 25),
                             nn.LogSoftmax(dim=1))
    return model
from torch.utils.data import Dataset
class Dataset(Dataset):
    def __init__(self, img_name, transform=None):
        self.img_name = img_name
        self.transform = transform


    def __len__(self):
        return 1

    def __getitem__(self, index):

        image = Image.open(self.img_name)
        image = image.convert('RGB')
        image = image.resize((300, 300))
        label = 10
        if self.transform is not None:
            image = self.transform(image)
        return image, label
if __name__ == '__main__':
    path='/home/do/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/frontend/static/frontend/text/'
    c = Classification(path)
    c.Train()
    #c.Classify_Data()
    #c.Classify_Text("An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.")
    #print(c.get_classification_results())
