#
# Libraries
#

import pymongo
from flask import Flask, request, render_template
from flask_paginate import Pagination, get_page_args


# create app instance
app = Flask(__name__)

#
# Routes
#

if __name__ == '__main__':
    app.secret_key='secret123'
    app.run(debug=True, threaded=True)
