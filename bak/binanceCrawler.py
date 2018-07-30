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

from basecrawler import BaseCrawler

class BinanceCrawler ( BaseCrawler):
    """
    Get daily volume data from binance
    """
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
        endpoint = "https://api.binance.com/api/v1/klines"
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
    
    def run(self):
        pass

if __name__ == "__main__":

    b = BinanceCrawler( limit=10)
    ar = b.get_API( b.endpoint )
    print( type(ar))



"""
[
  [
    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "17928899.62484339" // Ignore
  ]
]
    """
