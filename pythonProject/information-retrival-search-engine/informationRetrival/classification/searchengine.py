from typing import Dict, List, Sequence

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import MultifieldParser
from whoosh.filedb.filestore import RamStorage
from whoosh.analysis import StemmingAnalyzer
from pymongo import MongoClient
import json
from PIL import Image
# import Image
import requests
from io import BytesIO

#
# Simple example indexing to an in-memory index and performing a search
# across multiple fields and returning an array of highlighted results.
#
# One lacking feature of Whoosh is the no-analyze option. In this example
# the SearchEngine modifies the given schema and adds a RAW field. When doc
# are added to the index only stored fields in the schema are passed to Whoosh
# along with json encoded version of the whole doc stashed in the RAW field.
#
# On query the <Hit> in the result is ignored and instead the RAW field is
# decoded containing any extra fields present in the original document.
#


class SearchEngine:

    def __init__(self, schema):
        self.schema = schema
        schema.add('raw', TEXT(stored=True))
        self.ix = RamStorage().create_index(self.schema)

    def index_documents(self, docs: Sequence):
        writer = self.ix.writer()
        for doc in docs:
            d = {k: v for k,v in doc.items() if k in self.schema.stored_names()}
            d['raw'] = json.dumps(doc) # raw version of all of doc
            writer.add_document(**d)
        writer.commit(optimize=True)

    def get_index_size(self) -> int:
        return self.ix.doc_count_all()

    def query(self, q: str, fields: Sequence, highlight: bool=True) -> List[Dict]:
        search_results = []
        with self.ix.searcher() as searcher:
            results = searcher.search(MultifieldParser(fields, schema=self.schema).parse(q))
            for r in results:
                d = json.loads(r['raw'])
                if highlight:
                    for f in fields:
                        if r[f] and isinstance(r[f], str):
                            d[f] = r.highlights(f) or r[f]

                search_results.append(d)

        return search_results

if __name__ == '__main__':

    docs = [
        {"id": "2",
            "title": "F",
            "description": "oyh",
            "tags": ['idskhg'],
            "extra": "oyh"},
        {
            "id": "1",
            "title": "First document banana",
            "description": "This is the first document we've added in San Francisco!",
            "tags": ['foo', 'bar'],
            "extra": "kittens and cats"
        },
        {
            "id": "2",
            "title": "Second document hatstand",
            "description": "The second one is even more interesting!",
            "tags": ['alice'],
            "extra": "foals and horses"
        },
        {
            "id": "3",
            "title": "Third document slug",
            "description": "The third one is less interesting!",
            "tags": ['bob'],
            "extra": "bunny and rabbit"
        },
    ]
    doc2=[]
    host = '127.0.0.1'  # or localhost
    port = 27017
    client = MongoClient(host, port)

    db = client['allMovies']
    collection = db["Movie"]

    for i in (collection.find({"name": "183.txt"})):
        doc2.append(i)
    doc2=doc2[0]
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

    print("indexed ",engine.get_index_size()," documents")

    fields_to_search = ["title","overview"]
    y=[]
    for q in ["Fatherhood", "Fitzell", "first", "second", "alice", "bob", "san francisco","oyh"]:
        print("Query:: ",q)
        print("\t", engine.query(q, fields_to_search, highlight=True))
        x=engine.query(q, fields_to_search, highlight=True)
        y.append(x)
        print("-"*70)
    IMAGE_BASE_PATH = "https://image.tmdb.org/t/p/w500"
    p=y[0][0]["poster_path"]
    img_path = IMAGE_BASE_PATH + p
    html = requests.get(img_path, verify=False)
    print("ok")
    poster = Image.open(BytesIO(html.content))
    poster_img = poster.crop()


    poster_img.show()