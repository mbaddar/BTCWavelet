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

class Crawler:
    """
    Crawl Bitcoin hourly data from cryptocompare 
    json data are in ascending order
    """
    #__endpoint = "https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD"
    __e = '' #Exchanges
    __aggregate = 1 #if skips are needed
    __delay = 1000 #in ms
    __year_hours = 8760 # No of hours in a year 
    __limit = 2000 # API Hard limit 
    __hour_epoch = 3600 # no of seconds of an hour
    #__toTs = 1283299200 # 1/9/2010
    #__toTs = 1280620800 # 1/8/2010
    __toTs = 1530608400 #3/7/2018 9:00 UTC
    #The api only defines a to TS and backtracks
    __data_period = 5 # in years

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        # TODO Handle error more gracefully
        if  not data.empty:
            self.__data = data
        else:
             self.__data = DF() #Empty dataframe
    
    @property
    def endpoint(self):
        return self.__endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        #TODO error handling 
        if endpoint:    
            self.__endpoint = endpoint

    def __init__(self, currency_from = None, currency_to = None):
        endpoint = "https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD"
        # Update the parameters to accommodate different currency pairs 
        # Defaults are BTC and USD
        if currency_from:
            endpoint = self.update_url( endpoint, {'fsym': currency_from})

        if currency_to:
            endpoint = self.update_url( endpoint, {'tsym': currency_to})
        self.__endpoint = endpoint

        # Init data to an empty DataFrame
        self.__data = DF() 


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

    def full_url( self, params ):
        url_values = urlparse.urlencode( params)
        full_url = self.__endpoint + '&' + url_values
        return full_url

    def append_data(self, data, file_path):
        # TODO error handling 
        with open(file_path, "a+") as f:
            f.write(data)

    def get_data(self, toTs = 0, limit = 0):
        if toTs==0:
            toTs = self.__toTs
        if limit==0:
            limit = self.__limit
        params = {}
        params['limit'] = limit #defaults to 1 year (almost)
        params['toTs'] = toTs + self.__hour_epoch*limit #Advance the To TS to deal with backtracking
        # url_values = parse.urlencode(param)
        # full_url = self.__endpoint + '&' + url_values
        full_url = self.update_url( self.endpoint, params)
        try: 
            res = urllib.request.urlopen( full_url)
            b = res.read() #read resonse bytes
            json_string = b.decode('utf8').replace("'", '"')
            json_data = json.loads(json_string)
            formatted_json = json.dumps( json_data , indent=4, sort_keys=True)
        except urllib.error.URLError as e:
            print(e.reason)
        return formatted_json
    
    def run(self, path= "2018"):
        # loop for 5 years
        # To work around the API hard limit of 2000 items
        # Still a dirty way of getting many years of data

        #data collected in chunks of 2000 points
        hops = (  self.__data_period *self.__year_hours) //self.__limit
        #remainder = self.__year_hours%self.__limit
        limit = self.__limit
        for index in range( 0, hops + 1):
            toTs = self.__toTs - index* self.__limit* self.__hour_epoch
            print( toTs, time.gmtime( toTs))
            # API call
            data = self.get_data( 
                toTs = toTs,
                limit= limit )
            #Write to file
            self.append_data(data, path + "/hourly_" + str(toTs) + ".json")

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

    def get_complete_list( self, path, key = 'close'):
        file_list = self.get_file_names( path)
        output = []
        #Get the list out of each file and merge it 
        for file_name in file_list:
            current = self.json2List( path+ "\\" +file_name, key)
            output += current
        return output

    def get_complete_ndarray(self, path, key = 'close'):
        output = self.get_complete_list( path, key)
        return np.array( output)

    def json_generator(self, filename, key = 'close'):
        """
        Generates a tuple(time, key)
        """
        with open(filename , "r") as f:
            json_data = json.load(f)
            data_list = json_data['Data']
            #Generates a tuple 
            for item in data_list:
                yield( (item['time'], item[key])) 

    def get_complete_df( self, path, 
                         keys = ['close', 'high', 'low', 'open', 'volumefrom', 'volumeto']):
        frames = [ self.get_df( path, key) for key in keys]
        #return frames
        df = pd.concat(frames, axis=1)
        self.__data = df 
        # TODO remove return
        return df

    def add_df(self, df):
        """
        Add another DataFrame to the stored data
        """
        if not df.empty:
            self.data = pd.concat([self.data, df], axis=1)

    
    def get_df( self, path, key='close'):
        file_list = self.get_file_names( path)
        # Empty df
        df1 = DF()
        for file_name in file_list:
            df1 = df1.append( DF( self.json_generator( path + "\\"+ file_name, key)).set_index( 0) )
        # Need to set column names
        df1.index.names = ['Time']
        df1.columns = [ key]
        #df1.rename(index={0:'Date'}, columns={1: key}, inplace=True)
        return df1 
    

    def scrape_history(self):
        """
        Scrapes an html file originally saved from
        https://99bitcoins.com/price-chart-history/
        """
        def get_epoch( date):
            utc_time = datetime.strptime(date, "%B %d, %Y")
            epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
            return int(np.round(epoch_time))

        with open('history.html', encoding="utf8") as f:
            html_string = f.read()
        tree = html.fromstring( html_string )
        events = tree.xpath('//div[@class="bitcoin_history"]/h3/text()')
        prices = tree.xpath('//span[@class="label label-info"]/text()')
        later_prices = tree.xpath('//span[contains(@style,"color")]/text()')
        event_list = { "Event":  [ events[i][:events[i].rfind('-')-1] for i in range( len(prices))], 
        "Date": [ events[i][events[i].rfind('-')+2:] for i in range( len(prices))], 
        "Epoch": [ get_epoch( events[i][events[i].rfind('-')+2:]) for i in range( len(prices))],
        "Price1": [ prices[i] for i in range( len(prices))], 
        "Price2": [ later_prices[i] for i in range( len(prices))]
        } 

        df = DF( event_list).set_index("Epoch")
        return df


if __name__ == "__main__":

    c = Crawler( "BTC",  "USDT")
    #c.run( path="USDT")
    filename = "2018/hourly_1530608400.json"
    print( c.scrape_history())
    # df = DF(c.json_generator( filename)).set_index(0)
    # c.data = df
    # print( c.data)
    # c.add_df ( c.scrape_history())
    # print( c.data)
    # filename = "2018/hourly_1530608400.json"
    # filename2 = "2018/hourly_1480208400.json"
    # df1 = DF()
    # df2 = DF(c.json_generator( filename2)).set_index(0)
    # print("df1 size=" + str(df1.size) + " df2 size=" + str(df2.size) , " total size=" 
    # + str(df1.size + df2.size))
    # print(df1)
    # df1 = df1.append(df2)
    # print(df1)
    # for element in c.json_generator( filename):
    #     print(element)
    # output = c.json2ndarray( filename, 'close')

    # path = "2018"
    #file_list = c.get_file_names( os.getcwd() + "\\2018")
    # file_list = c.get_file_names( path)
    #df1 = c.get_df( path)
    # df2 = c.get_complete_df ( path, ['close', 'high'] )
    # print(df2['close'])

    # output = c.get_complete_list( path)
    # print(len(output))
    # print("[%s]" % ", ".join(["%.2f" % x for x in output]))
