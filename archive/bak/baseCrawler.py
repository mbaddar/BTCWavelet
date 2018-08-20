import json
import os
import time
import urllib.error
import urllib.parse as urlparse
import urllib.request
import urllib.response
from datetime import datetime
from urllib.parse import urlencode
import requests

import numpy as np
import pandas as pd
from lxml import html
from pandas import DataFrame as DF
from time import sleep

class BaseCrawler (object):

    def update_url (self, endpoint, params ):
        """
        Updates the parameters of a URL from the params dict
        Add new parameters and modifies the existing ones
        """
        url_parts = list( urlparse.urlparse( endpoint ))
        query = dict(urlparse.parse_qsl(url_parts[4]))
        query.update(params)
        url_parts[4] = urlencode(query)
        return urlparse.urlunparse(url_parts)

    # def full_url( self, params ):
    #     url_values = urlparse.urlencode( params)
    #     full_url = self.__endpoint + '&' + url_values
    #     return full_url

    def append_data(self, data, file_path):
        # TODO error handling 
        with open(file_path, "a+") as f:
            f.write(data)

    def get_API_Json(self, url):
        resp = requests.get( url)
        resp_json = None
        if resp.ok:
            resp_json = resp.json()
        return resp_json

    def get_epoch( self, date):
        #Ex. March 8, 2017
        utc_time = datetime.strptime( date, "%B %d, %Y")
        epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
        return int( np.round( epoch_time))

    def get_days_between (self, date1, date2):
        date_format = "%B %d, %Y"
        delta = datetime.strptime( date1, date_format) - datetime.strptime( date2, date_format)
        return delta.days

    def today_midnight( self):
        pass 

    def get_API (self, url):
        formatted_json = "error"
        resp_json = self.get_API_Json( url)
        formatted_json = json.dumps( resp_json , indent=4, sort_keys=True)
        return formatted_json
        # Old code. To be purged
        # res = urllib.request.urlopen( full_url)
        # b = res.read() #read resonse bytes
        # json_string = b.decode('utf8').replace("'", '"')
        # json_data = json.loads(json_string)

    def post_API( self, full_url, params):
        # params = {
        #     "common":{
        #         "accesskey" : "1900000109",            # Personal Accesskey
        #         "sign":"sdfsdfa1231231sdfsdfsd"        # MD5 Encrypted Secret Key
        #         "timestamp":1500000000000,        # UTC Time (Milliseconds)
        #     },
        #     "data":{
        #                                          # Data Specific to Request
        #     }
        # }        
        #     formatted_json = "error"
        #     res = requests.post( full_url, json= params)
        pass

    def filter_list (self, symbols, symbol= 'btc'):
        symbols = [x for x in symbols if x.lower().find( symbol) >= 0 ]
        return symbols

    def get_Symbols( self, symbols_endpoint ):
        symbols = self.get_API_Json( symbols_endpoint)
        #Only keep BTC pair
        symbols = self.filter_list( symbols, 'btc')
        return symbols


    def json2List(self, filename, key= 'close'):
        output = []
        with open(filename , "r") as f:
            json_data = json.load(f)
            data_list = json_data['Data']
            for item in data_list:
                output.append( item[key])
        #Converted to numpy.array for compatibility with wavelet module
        return output

    def json2ndarray(self, filename, key = 'close'):
        output = self.json2List( filename, key)
        #Converted to numpy.array for compatibility with wavelet module
        return np.array( output )
    
    def get_file_names (self, path):
        """
        return an ordered list of files starting with "hourly" in a given file path
        """ 
        files = os.listdir(path)
        files.sort()
        return files
