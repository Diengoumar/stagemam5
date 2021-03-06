# Information-retrival-search-engine  

[![Stories in Ready](https://badge.waffle.io/BhavyaLight/information-retrival-search-engine.png?label=ready&title=Ready)](https://waffle.io/BhavyaLight/information-retrival-search-engine) [![Stories in Backlog](https://badge.waffle.io/BhavyaLight/information-retrival-search-engine.png?label=backlog&title=Backlog)](https://waffle.io/BhavyaLight/information-retrival-search-engine)

# Etape 2
## Option 1
## Project Development Set-Up

### Requirements (doesnt work, please see how to run locally)
- Install docker
- Git clone this repository into your local disk

Run the following commands:  
```bash
docker build -t information_retrival:version1 .
```
```bash
docker run --publish=8001:8000 information_retrival:version1
```
Note: The django website will now be available on port 8001 instead of 8000

## To run locally without Docker 

### Requirments
- Run pip install -r requirements.txt
- Download sklearn, nltk, pymongo
- Download stopwords from nltk corpus
- Connect to MongoDb through mongodb compass @ 155.69.160.73
- Standard command to connect via python looks like below:

```
from pymongo import MongoClient

client=MongoClient("mongodb://10.27.136.138")

db=client['IR']
movies=db['Movies']
```

### Point to correct path
- Under views.py please change the following (as shown in the example) :
```
INDEX_FILE = '/Users/noopurjain/Desktop/Index'
WRITE_FILE = '/Users/noopurjain/Desktop/Trial_2'
CLASSIFICATION_PATH = '/mnt/d/model_files_new_with_voting_with_weights/'
```
Note the backslash after CLASSIFICATION_PATH. They should point to the index folder (called 'Index' on dropbox), raw crawled files (inside crawledData.zip on dropbox) and model files (inside dropbox model files)

## Option 2

download my virtual environment "O" https://drive.google.com/drive/folders/19YoMc7CGpPtam95ENyvWgcqu-RJC-Va4 which contains all the packages installed from python 3.5 and in addition you have access to the database for the classification genre_image
 after downloading the virtual environment, you have to put it inside Anaconda/venv, and it will be installed directly.

I prefer this option because I had a lot of problems with package installations 

# Etape 3
## Run the code finally
- Inside information-retrival-search-engine/informationRetrival directory do  
```bash
$ python manage.py runserver
```
- Open http://localhost:8000 knock yourself out searching and classifying (only, for now :( ) movies


END
