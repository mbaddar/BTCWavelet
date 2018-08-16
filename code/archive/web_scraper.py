from lxml import html
import requests
from datetime import datetime
import numpy
import json
import pandas as pd
from pandas import DataFrame as DF

with open('history.html', encoding="utf8") as f:
    html_string = f.read()
tree = html.fromstring( html_string )
events = tree.xpath('//div[@class="bitcoin_history"]/h3/text()')
prices = tree.xpath('//span[@class="label label-info"]/text()')
later_prices = tree.xpath('//span[contains(@style,"color")]/text()')

def get_epoch(date):
    utc_time = datetime.strptime(date, "%B %d, %Y")
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
    return int(numpy.round(epoch_time))

event_list = { "Event":  [ events[i][:events[i].rfind('-')-1] for i in range( len(prices))], 
   "Date": [ events[i][events[i].rfind('-')+2:] for i in range( len(prices))], 
   "Epoch": [ get_epoch( events[i][events[i].rfind('-')+2:]) for i in range( len(prices))],
   "Price1": [ prices[i] for i in range( len(prices))], 
   "Price2": [ later_prices[i] for i in range( len(prices))]
  } 

df = DF( event_list)
