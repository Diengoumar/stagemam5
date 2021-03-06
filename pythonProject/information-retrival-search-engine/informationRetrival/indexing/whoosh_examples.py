# coding=utf-8
import importlib
import os
import ast
from MovieData import MovieData
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, DATETIME, NUMERIC, BOOLEAN
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
from whoosh import index, query
from numpy import unicode
from whoosh import scoring
import json
import sys
importlib.reload(sys)
def unicode_convert(input):
    if isinstance(input, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [unicode_convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input
BASE_PATH="/home/ubuntu/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/frontend/static/frontend/json/" # json文件地址
#BASE_PATH="../../../trial/movies/movies" # json文件地址
FILEPATH="../../../Index_tmp_test" # whoosh index索引地址
#Schema for Movie data
#Stored= True corresponds to all items that require to be returned with the search result
schema = Schema(overview=TEXT(analyzer=StemmingAnalyzer(), spelling=True, stored=True),
                tagline=TEXT(analyzer=StemmingAnalyzer(), spelling=True, stored=True),
                title=TEXT(analyzer=StemmingAnalyzer(), spelling=True, stored=True),
                production_companies=TEXT(analyzer=StemmingAnalyzer(), spelling=True, stored=True),
                genres=TEXT(analyzer=StemmingAnalyzer(), spelling=True, stored=True),
                runtime=STORED,
                poster_path=STORED,
                actorUrl=STORED,
                imageUrl=STORED,
                imdb_id=ID(stored=True),
                popularity=NUMERIC(float,bits=64, stored=True),
                revenue=NUMERIC(float,bits=64, stored=True),
                vote_average=NUMERIC(float,bits=64, stored=True),
                adult=BOOLEAN(stored=True),
                release_date=DATETIME(stored=True)
                )

# instantiate index
if not os.path.exists(FILEPATH):
    os.mkdir(FILEPATH)
try:
    ix = index.open_dir(FILEPATH)
except Exception as err:
    ix = index.create_in(FILEPATH, schema)
#open index writer

writer = ix.writer()

all_file = os.listdir(BASE_PATH)
num = 0
for i in all_file:
    if num > 500:
        break;
    num += 1
    if len(i.split(".")) > 1:
        if i.split(".")[1].strip("\n") != "json":
            continue
    else :
        continue
    path = BASE_PATH + "/" + i
    #print(path)
    with open(path) as f:
        a = json.load(f)
        sep = json.dumps(a, ensure_ascii=False)
        #c = unicode_convert(json.loads(sep))
        c = json.loads(sep)
    #print (i)
    prod = []
    # print (f2.get("production_companies"))
    for x in c["production_companies"]:
        prod.append(x['name'])
    genres=[]
    for x in c["genres"]:
        genres.append(x['name'])

    prodstring = ''
    for x in prod:
        prodstring = prodstring+x+' '
    #print (prodstring)
    genrestring = ''
    for x in genres:
        genrestring = genrestring+x+' '
    #print (genrestring)
    rdate = c["release_date"]
    #print (rdate)
    if rdate == '':
        rdate = u'2100-10-10'
    f2=dict()
    f2['id'] = c['id']
    f2['overview'] = c['overview']
    f2['tagline'] = c['tagline']
    f2['title'] = c['title']
    f2['runtime'] = c['runtime']
    f2['poster_path'] = c['poster_path']
    f2['actorUrl'] = c['actorUrl']
    f2['imageUrl'] = c['imageUrl']
    f2['imdb_id'] = c['imdb_id']
    f2['popularity'] = c['popularity']
    f2['revenue'] = c['revenue']
    f2['vote_average'] = c['vote_average']
    f2['adult'] = c['adult']
    print(genrestring)



    #print ("##################################################################################")
    writer.add_document(id = unicode(f2['id']),overview=unicode(f2['overview']), tagline=unicode(f2['tagline']),title=unicode(f2['title']), production_companies=unicode(prodstring),
                    genres=unicode(genrestring),runtime=unicode(f2['runtime']), poster_path=unicode(f2['poster_path']), actorUrl=unicode(f2['actorUrl']), imageUrl=unicode(f2['imageUrl']),imdb_id=unicode(f2['imdb_id']),popularity=unicode(f2['popularity']),
                    revenue=unicode(f2['revenue']), vote_average=unicode(f2['vote_average']), adult=unicode(f2['adult']), release_date=unicode(rdate))

#commit writer
writer.commit()
print("done ##################################################################################")



