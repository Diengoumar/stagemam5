# coding=utf-8
from pymongo import MongoClient
import json
import os
import ast



host = '127.0.0.1' # or localhost
port = 27017
    # 创建mongodb客户端
client = MongoClient(host, port)
    # 创建数据库dialog
db = client.oumar
    # 创建集合scene
collection = db.movie



def write_database():
    file='oumar'
    datalines=[]
    for i in range(1,221):
        g = open(
            '/home/do/PycharmProjects/pythonProject/information-retrival-search-engine/informationRetrival/frontend/static/frontend/json/'+str(i)+'.txt',
            'r')
        line=ast.literal_eval(g.read())
        datalines.append(line)
    data = {
        "name": file,
        "content": datalines
    }
    try:
        myquery = {"name": file}  # 查询条件
        collection.update(myquery, data, upsert=True)  # upsert=True不存在则插入，存在则更新
        collection.insert(data)
        print('Insert successfully')
    except Exception as e:
        print(e)


write_database()