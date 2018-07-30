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
from time import sleep

from baseCrawler import BaseCrawler

class BitfinexCrawler ( BaseCrawler):
    @property
    def endpoint(self):
        return self.__endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        #TODO error handling 
        if endpoint:    
            self.__endpoint = endpoint

    @property
    def symbols_endpoint(self):
        return self.__symbols_endpoint

    @symbols_endpoint.setter
    def symbols_endpoint(self, symbols_endpoint):
        #TODO error handling 
        if symbols_endpoint:    
            self.__symbols_endpoint = symbols_endpoint

    @property
    def symbols(self):
        return self.__symbols

    @symbols.setter
    def symbols(self, symbols):
        #TODO error handling 
        if symbols:    
            self.__symbols = symbols

    def __init__(self, symbol= "tBTCUSD" , end = None, limit = None ):
        params = { "limit": 100 , "end": 1531180800000 }
        endpoint = "https://api.bitfinex.com/v2/candles/trade:1D:"+ symbol+ "/hist"
        # Update the parameters to accommodate different currency pairs 
        # Defaults are in __params
        endpoint = super().update_url( endpoint, params ) 
        if symbol:
            endpoint = super().update_url( endpoint, {'symbol': symbol})
        if end:
            endpoint = super().update_url( endpoint, {'end': end*1000}) #To milliseconds
        if limit:
            endpoint = super().update_url( endpoint, {'limit': limit})

        self.__endpoint = endpoint

        self.__symbols_endpoint = "https://api.bitfinex.com/v1/symbols"

        self.__symbols = super().get_Symbols( self.symbols_endpoint)

    def run(self):
        data = {}
        #get symbols 
        #crawl them all
        then = time.time()
        for count, symbol in enumerate( self.symbols):
            #fix endpoint
            self.endpoint = super().update_url( self.endpoint, {'symbol': symbol})
            #Get the symbol's data
            data[symbol] = super().get_API_Json( self.endpoint )
            #delay to comply with rate limit
            seconds = time.time() - then
            print( "(seconds=%.2f, Count=%d)" % (seconds,count) )
            if  seconds <60 and count>0 and count%44==0: 
                #Ratelimit 45 req/min
                delay = 60-seconds +0.1
                print("Sleeping for %d seconds" % delay)
                sleep( delay )
                then = time.time()
        return data

if __name__ == "__main__":
    #t for trading, f for funding
    b = BitfinexCrawler( limit= 1)
    delta = b.get_days_between( "July 18, 2018", "January 1, 2017")
    data = b.run()
    print( "Symbols: \n", b.symbols)
    print( "Data: \n", data)
# // response with Section = "hist"
# [
#   [ MTS, OPEN, CLOSE, HIGH, LOW, VOLUME ], 
#   ...
# ]


