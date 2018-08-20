from pandas import DataFrame as DF
import pandas as pd
import numpy as np 
import urllib.error
import urllib.parse as urlparse
from urllib.parse import urlencode
import urllib.request 
import urllib.response
import json 
import time 
import os
from lxml import html
from datetime import datetime

from baseCrawler import BaseCrawler

class coinsuperCrawler ( BaseCrawler):
    @property
    def endpoint(self):
        return self.__endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        #TODO error handling 
        if endpoint:    
            self.__endpoint = endpoint

    def __init__(self, symbol= None , startTime = None, limit = None ):
        params = {"symbol": "BTCUSDT", "startTime": 1483228800000, "interval": "1d", "limit": 500 }
        endpoint = "https://api.coinsuper.com"
        # Update the parameters to accommodate different currency pairs 
        # Defaults are in __params
        endpoint = super().update_url( endpoint, params ) 
        if symbol:
            endpoint = super().update_url( endpoint, {'symbol': symbol})
        if startTime:
            endpoint = super().update_url( endpoint, {'startTime': startTime})
        if limit:
            endpoint = super().update_url( endpoint, {'limit': limit})

        self.__endpoint = endpoint
        # Init data to an empty DataFrame        
        self.__data = DF() 