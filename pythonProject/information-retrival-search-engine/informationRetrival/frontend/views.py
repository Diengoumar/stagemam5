from django.shortcuts import render
from .forms import SearchForm, ClassifyForm, UploadFileForm
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh import index as i
from whoosh import scoring
import whoosh.query as QRY
import time
import pandas as pd
from typing import Dict, List, Sequence

import torchvision.transforms as transforms
from PIL import Image
# import Image
import requests
from io import BytesIO
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import MultifieldParser
from whoosh.filedb.filestore import RamStorage
from whoosh.analysis import StemmingAnalyzer

from datetime import datetime
from indexing.crawl import crawl_and_update
from classification.classify import Classification
from classification.classify import Net
from classification.classify import label

from numpy import unicode
from .vgg16_p import compare
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from  classification.bert import todo
from melanger.melanger import todo_all
from melanger.vgg16_p.newvgg import res_comparer
from classification.searchengine import SearchEngine
from pymongo import MongoClient

import torch


model_classify = Net()

model_classify.load_state_dict(torch.load('/home/do/Téléchargements/newdata/model_torchnet',map_location='cpu'))
#model_classify = keras.models.load_model('/home/ubuntu/Desktop/mymodel')
INDEX_FILE = '/home/do/PycharmProjects/pythonProject/information-retrival-search-engine/index'

WRITE_FILE="/home/do/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/frontend/static/frontend/json/"
CLASSIFICATION_PATH = '/home/do/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/frontend/static/frontend/text/'
doc2 = []
host = '127.0.0.1'  # or localhost
port = 27017
client = MongoClient(host, port)

db = client['oumar']
collection = db["movie"]

for i in (collection.find({"name": "oumar"})):
    doc2.append(i)
doc2 = doc2[0]
def show(request):
    if request.method == 'POST':
        overview = request.POST.get('overview')
        title = request.POST.get('title')
        poster_path = request.POST.get('poster_path')
        id = request.POST.get('imdb_id')
        print (id)
        ix = i.open_dir(INDEX_FILE)
        searcher = ix.searcher()
        docnum = searcher.document_number(imdb_id=id)
        recoms = searcher.more_like(docnum,'overview')
        return render(request, 'frontend/show.html', {'overview': overview, 'title': title, 'poster_path': poster_path, 'recommendations': recoms})

def index(request):
    if request.method == 'POST':

        query = request.POST.get("search_text")
        search_list = request.POST.getlist("search")
        trii=[]
        for i in search_list:
            if i=='0':
                trii.append("title")
            if i=='1':
                trii.append("overview")
        print("trii",trii)
        file_obj = request.FILES.get('uploadPicture')
        if file_obj:
            start_time=time.time()
            new_img = Image.open(file_obj)
            new_img = new_img.crop()
            extract_label=res_comparer(new_img)
            schema = Schema(
                id=ID(stored=True),
                title=TEXT(stored=True),
                description=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                overview=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                tags=KEYWORD(stored=True)
            )

            engine = SearchEngine(schema)
            engine.index_documents(doc2['content'])
            query=[]
            for i in range(len(extract_label)):
                query.append(extract_label[i][1])
            print('query',query)

            for q in query:
                res = engine.query(q, trii, highlight=True)

            # res = todo_all(query,search_list)
                print("res", res)
                if res:
                    IMAGE_BASE_PATH = "https://image.tmdb.org/t/p/w500"
                    p = res[0]["poster_path"]
                    img_path = IMAGE_BASE_PATH + p
                    html = requests.get(img_path, verify=False)
                    print("ok")
                    poster = Image.open(BytesIO(html.content))
                    poster_img = poster.crop()
                    poster_img.show()
                    elapsed_time = time.time() - start_time
                    return render(request, 'frontend/index.html',
                          {'search': search_list, 'error': False, 'hits': [res[0]], 'search_text': q,
                           'elapsed': elapsed_time, 'number': 1, 'year': '1990', 'rating': '0.1', 'results': res})
            print("je suis là")
            return render(request, 'frontend/index.html', {'search_text': ""})

        print(query)
        res = []
        start_time = time.time()
        print(search_list)

        print(doc2['content'][0]['title'])
        schema = Schema(
            id=ID(stored=True),
            title=TEXT(stored=True),
            description=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            overview=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            tags=KEYWORD(stored=True)
        )

        engine = SearchEngine(schema)
        engine.index_documents(doc2['content'])
        res=engine.query(query,trii,highlight=True)
        #res = todo_all(query,search_list)
        print("res",res)
        print("res",len(res))

        if res:
            IMAGE_BASE_PATH = "https://image.tmdb.org/t/p/w500"
            p = res[0]["poster_path"]
            img_path = IMAGE_BASE_PATH + p
            html = requests.get(img_path, verify=False)
            print("ok")
            poster = Image.open(BytesIO(html.content))
            poster_img = poster.crop()
            #poster_img.show()
            year = "1900,2020"
            rating = "0,10"

            results=[]
            results.append(res[0])
            for i in range(0,len(res)-1):
                if res[i]!=res[i+1]:
                    results.append(res[i+1])
            elapsed_time = time.time() - start_time

            return render(request, 'frontend/index.html',
                          {'search': search_list, 'error': False, 'hits': results, 'search_text': query,
                           'elapsed': elapsed_time, 'number': 1, 'year': year, 'rating': rating, 'results':results})

        return render(request, 'frontend/oumarsaysno.html')
    else:
        print("je suis là")
        return render(request, 'frontend/index.html', {'search_text': ""})


def filter(request):
    res = request.GET.getlist("result")
    print(res)
    rating = request.GET.get("rating")
    year = request.GET.get("year")
    query = request.GET.get("search_text")
    genre_list = request.GET.getlist('multi_genre')
    date_q = QRY.DateRange("release_date", datetime.strptime(year.split(",")[0], "%Y"),datetime.strptime(year.split(",")[1], "%Y"))
    rating_q = QRY.NumericRange("vote_average",int(rating.split(",")[0]), int(rating.split(",")[1]))
    filter_q = QRY.And([date_q, rating_q])
    if len(genre_list) > 0:
        genres_q=QRY.Or([QRY.Term(u"genres",unicode(x.lower())) for x in genre_list])
        filter_q = QRY.And([filter_q, genres_q])
    ix = i.open_dir(INDEX_FILE)
    searcher = ix.searcher(weighting=scoring.TF_IDF())
    hitsList=[]
    for x in res:
        res_q = QRY.Term(u"id", unicode(x))
        filter = QRY.And([filter_q, res_q])
        temp_hit = searcher.search(filter, filter=None, limit=None)
        if temp_hit:
            for y in temp_hit:
                hitsList.append(y)





    print(filter_q)

    return render(request, 'frontend/index.html',
                  { 'error': False, 'hits': hitsList, 'search_text': query,
                    'number': len(hitsList), 'year': year, 'rating': rating,'results':res})


    #         rating = request.GET.get("rating")
    #         year = request.GET.get("year")
    #         genre_list = request.GET.getlist('multi_genre')
    #         filter_q = None
    #         # TODO: Change Directory here
    #         ix = i.open_dir(INDEX_FILE)
    #         start_time = time.time()from whoosh.index import create_in
    #         if query is not None and query != u"":
    #             parser = MultifieldParser(search_field, schema=ix.schema)
    #             if year is not None:
    #                 date_q = QRY.DateRange("release_date", datetime.strptime(year.split(",")[0], "%Y"), datetime.strptime(year.split(",")[1], "%Y"))
    #                 rating_q = QRY.NumericRange("vote_average",int(rating.split(",")[0]), int(rating.split(",")[1]))
    #
    #                 if len(genre_list)>0:
    #                     genres_q=QRY.Or([QRY.Term(u"genres",unicode(x.lower())) for x in genre_list])
    #                     combi_q = QRY.And([rating_q, genres_q])
    #                     filter_q = QRY.Require(date_q, combi_q)
    #                 else:
    #                     filter_q = QRY.Require(date_q, rating_q)
    #
    #
    #             else:
    #                 year = "1900,2020"
    #                 rating = "0,10"
    #
    #             try:
    #                 qry = parser.parse(query)
    #
    #             except:
    #                 qry = None
    #                 return render(request, 'frontend/index.html', {'error': True, 'message':"Query is null!"})
    #             if qry is not None:
    #                 searcher = ix.searcher(weighting=scoring.TF_IDF())
    #                 corrected = searcher.correct_query(qry, query)
    #                 if corrected.query != qry:
    #                     return render(request, 'frontend/index.html', {'search_field': search_field, 'correction': True, 'suggested': corrected.string, 'search_text':query})
    #                 print(qry,filter_q)
    #                 hits = searcher.search(qry, filter=filter_q, limit=None)
    #                 print(hits)
    #                 elapsed_time = time.time() - start_time
    #                 elapsed_time = "{0:.3f}".format(elapsed_time)
    #                 print(query,search_list)
    #                 return render(request, 'frontend/index.html', {'search': search_list,'error': False, 'hits': hits, 'search_text': query, 'elapsed': elapsed_time,
    #                                                                'number': len(hits), 'year': year, 'rating': rating})
    #             else:
    #                 return render(request, 'frontend/index.html', {'error': True, 'message':"Sorry couldn't parse", 'search_text':query})
    #         else:
    #             return render(request, 'frontend/index.html', {'error': True, 'message':'oops', 'search_text':query})
    #     else:
    #         return render(request, 'frontend/index.html', {'search_text':""})
    # else:
    #     return render(request, 'frontend/index.html', {'search_text': ""})
    #

def classification(request):
    results_dict = Classification(CLASSIFICATION_PATH).get_classification_results()
    results = pd.DataFrame(results_dict)
    for column in ['romance','crime','horror']:
        results[column] = results[column].apply(lambda x: str((int(x.split('/')[0]) * 100)/int(x.split('/')[1]))+" %")

    results.columns = ['F(1) Score', 'F(W) Score', 'Recall', 'Accuracy', 'Crime', 'Horror', 'Model', 'Precision', 'Romance','Vectorizer']
    results = results[['Model','Vectorizer', 'Crime', 'Horror', 'Romance', 'F(1) Score', 'F(W) Score', 'Recall', 'Accuracy', 'Precision']]
    results = results.to_html

    if request.method == "POST":
        form = ClassifyForm(request.POST)
        file_obj = request.FILES.get('uploadPicture2')
        if file_obj:
            print('img class ',file_obj)
            #start_time = time.time()

            img=Image.open(file_obj).resize((300,300))
            img.show()
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            timg = transform(img).unsqueeze(0)
            output_ = model_classify(timg)
            output_ = output_.argmax()
            predicted=output_
            print("resultat classification :",predicted)
            return render(request, 'frontend/classify.html', {'results': results, 'form': form, 'genre': label(predicted.item()),'time': 0.5,'upload ': 0.6})

        if form.is_valid():
            plot = form.cleaned_data['classify_plot']
            genre, time = Classification(CLASSIFICATION_PATH).Classify_Text(plot)
            return render(request, 'frontend/classify.html', {'results': results, 'form': form, 'genre': genre[0], 'time': time})
        else:
            return render(request, 'frontend/classify.html', {'results': results, 'form': form})
    else:
        form = ClassifyForm()
        return render(request, 'frontend/classify.html', {'results': results, 'form': form})


def crawl(request):
    if request.method == "GET":
        form = SearchForm(request.GET)
        date_now = datetime.now()
        search_field = request.GET.get('search_field')
        query = request.GET.get('search_text')
        ix = i.open_dir(INDEX_FILE)
        parser = QueryParser("release_date", schema=ix.schema)
        qry = parser.parse(date_now.strftime("%Y-%m-%d"))
        searcher = ix.searcher()
        hits = searcher.search(qry, limit=1)
        print("hitsc",hits)
        if (len(hits)==0):
        # send new records directory to the indexing function to add them to the index
            total_records = crawl_and_update(date_now, WRITE_FILE, INDEX_FILE)
        else:
            total_records = "Already up-to-date"

        return render(request, 'frontend/crawl.html', {'total_records': total_records, 'form': form})





def handleImg(request):
    year = "1900,2020"
    rating = "0,10"

    start_time = time.process_time()
    form = UploadFileForm(request.POST, request.FILES)
    print(request.FILES)

    file_obj = request.FILES.get('upload_picture')
    with open('frontend/static/frontend/images/temp.jpg', 'wb+') as destination:
        destination.write(file_obj.read())
    res = compare()

    ix = i.open_dir(INDEX_FILE)
    searcher = ix.searcher(weighting=scoring.TF_IDF())

    res_q = QRY.Or([QRY.Term(u"movie_id", unicode(x.lower())) for x in res])

    # parser = MultifieldParser(search_field, schema=ix.schema)


    hits = searcher.search(res_q, filter=None, limit=None)
    elapsed_time = time.process_time() - start_time
    return


