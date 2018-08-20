import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
import random
import pandas as pd
from pandas_datareader import data as pdr
from pandas import Series, DataFrame
import datetime
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
import time
from time import sleep
from sklearn import linear_model
from datetime import datetime,  timedelta
import calendar 

from crawlers.crawler import Crawler
from decomposition import Wavelet_Wrapper 

date_format = "%Y-%m-%d %H:%M:%S"

def get_days_between ( date1, date2):
    delta = datetime.strptime( date1, date_format) - datetime.strptime( date2, date_format)
    return delta.days

def add_days ( from_date, days=1 ):
    new_date = datetime.strptime( from_date, date_format) + timedelta(days = days)
    return new_date

def str_from_date( t , date_format = "%Y-%m-%d" ):
    return t.strftime( date_format )

def get_date_from_epoch( epoch):
    t1 = time.gmtime( epoch)
    t = datetime(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec)
    str_time = t.strftime( date_format )
    return str_time

def get_epoch( date_str): #change to use the sinceEpoch function
    utc_time = datetime.strptime( date_str, date_format)
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
    return int( np.round( epoch_time))

def get_epoch2( date_str ): #change to use the sinceEpoch function
    utc_time = datetime.strptime( date_str, "%b %d, %Y")
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
    return int( np.round( epoch_time))

def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())

def toYearFraction(epoch):
    t1 = time.gmtime( epoch)
    year = t1.tm_year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = epoch - sinceEpoch(startOfThisYear)
    yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return year + fraction

def to_year_from_fraction(fraction):
    year = int(np.floor(fraction))
    remainder = fraction - year 
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
    year_epoch = remainder*yearDuration + sinceEpoch(startOfThisYear)
    t1 = time.gmtime( year_epoch)
    t = datetime(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec)
    return t

class Data_Wrapper:
    """
    from: 17/09/2013 9 am
    to: 06/07/2018 4 pm
    """
    # @property
    # def lppl_data(self):
    #     return self.__lppl_data

    # @lppl_data.setter
    # def lppl_data(self, lppl_data):
    #     self.__lppl_data = lppl_data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    # @property 
    # def data_size(self):
    #     return self.__data_size
    # @data_size.setter
    # def data_size(self, data_size):
    #     self.__data_size = data_size

    def __init__ (self, hourly = True, data_source = 'BTC'):
        """
        Support multiple data sources
        data_source: one of: 'BTC', 'SP500'
        """
        # self.lppl_data = None 
        self.data = None
        if data_source == 'BTC':
            if hourly:
                self.get_hourly_data()
            else:
                self.get_lppl_data()
        # Turn into dictoinary
        elif data_source == 'SSE':
            self.get_SSE_data()
        elif data_source == 'SP500': #bad branching. To compact
            self.get_Historical_data( path = 'sp5001987.csv')
        elif data_source == 'DIJA':
            self.get_Historical_data( path = 'dija1929.csv')
        elif data_source == 'OMXS30':
            self.get_omx_data( path = 'OMXS30.csv', sep=';')
        else:
            print("Invalid data source") #TODO raise error 

    def get_SSE_data ( self, path = 'Data\\SSE.csv'):
        df = None
        try:
            names=[ "Date", "Price", "Open", "High", "Low", "Vol", "Change" ]
            df = pd.read_csv( path , sep=',', parse_dates=['Date'], index_col='Date', 
                                    names = names,
                                    header= 0).reset_index()  
            for col in names[1:4]:
                df[ col] = df[ col].apply(lambda x: float( x.replace(',','') ) )
            df['LogClose'] = df['Price'].apply( lambda x: np.log(x))
            df['StrDate'] = df['Date']
            df['Date'] = df['Date'].apply( lambda date: int(np.round((date - datetime(1970, 1, 1)).total_seconds()) ))
        except BaseException as e:
            print( e )
        self.data = df
        self.data_size = df['LogClose'].size
        return df 
    #Need to refactor and generalize
    def get_omx_data(self, path = 'sp5001987.csv', sep=','):
        path = 'Data\\EWS-QR-LPPL-data-master\\' + path
        df = None
        try:
            #names=[ "Date", "Price", "Open", "High", "Low", "Vol", "Change" ]
            df = pd.read_csv( path , sep= sep, 
                names=['Date', 'High', 'Low', 'Close', 'Average', 'Total', 'Turnover', 'Trades'], 
                parse_dates=['Date'], index_col='Date', usecols = ['Date', 'Close'],
                header= 0).reset_index()  
            df.columns = ['Date', 'Close']
            df['Close'] = df['Close'].apply(lambda x: float( x.replace(',','') ) )
            # df['LogClose'] = df['Close']
            df['LogClose'] = df['Close'].apply( lambda x: np.log(x))
            # df['StrDate'] = df['Date']
            df['Date'] = df['Date'].apply( lambda date: int(np.round((date - datetime(1970, 1, 1)).total_seconds()) ))
            df.index = reversed(df.index)
            df = df.sort_index()
        except BaseException as e:
            print( e )
        self.data = df
        self.data_size = df['LogClose'].size
        return df 

    def get_Historical_data(self, path = 'sp5001987.csv', sep=','):
        path = 'Data\\EWS-QR-LPPL-data-master\\' + path
        df = None
        try:
            #names=[ "Date", "Price", "Open", "High", "Low", "Vol", "Change" ]
            df = pd.read_csv( path , sep= sep, names=['Date', 'Close'], parse_dates=['Date'], index_col='Date', 
                                    header= 0).reset_index()  
            df.columns = ['Date', 'Close']
            df['LogClose'] = df['Close'].apply( lambda x: np.log(x))
            df['StrDate'] = df['Date']
            df['Date'] = df['Date'].apply( lambda date: int(np.round((date - datetime(1970, 1, 1)).total_seconds()) ))
            #df['Date'] = df['Date'].apply( lambda x: )
        except BaseException as e:
            print( e )
        self.data = df
        self.data_size = df['LogClose'].size
        return df 

    def get_lppl_data(self, date_from = '2015-09-01 00:00:00', date_to = '2015-10-24 00:00:00', force_read = False ):
        # BTC
        # daily_data = self.lppl_data
        #read once 
        # if not self.lppl_data or force_read:
        daily_data = pd.read_csv( "Data/cmc/daily.csv", sep='\t',
                                names=[ 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap'],
                                header=0)
        #data preprocessing. remove the , mark 
        daily_data['Open'] = daily_data['Open'].apply(lambda x: float( x.replace(',','') ) )
        daily_data['High'] = daily_data['High'].apply(lambda x: float( x.replace(',','') ) )
        daily_data['Low'] = daily_data['Low'].apply(lambda x: float( x.replace(',','') ) )
        daily_data['Close'] = daily_data['Close'].\
                apply(lambda x: float( x.replace(',','') ) )
        # Lppl works on log prices
        daily_data['LogClose'] = daily_data['Close'].apply( lambda x: np.log(x) )
        daily_data['Date'] = daily_data['Date'].apply( lambda x: get_epoch2( x) )
        # self.lppl_data = daily_data 
        daily_data['StrDate'] = daily_data['Date'].apply( lambda epoch: get_date_from_epoch( epoch) )
        #Filter
        # daily_data = daily_data.loc[daily_data.index >= date_from] #Min
        # daily_data = daily_data.loc[daily_data.index <= date_to] #Max
        # reverse index
        daily_data.index = reversed(daily_data.index)
        daily_data = daily_data.sort_index()
        self.data = daily_data
        self.data_size = daily_data.LogClose.size
        return daily_data
    def get_data_series( self, data = None, direction = 1, col = 'LogClose', fraction = 1):
        """
        Direction: +/- 1
        fraction is a flag. if set to 0 (default): dataSeries[0] is time points indexed from 0
        if set to 1: return fractional year. Example: 2018.604060 for 9/8/2018
        """
        if data is None or data.empty:
            data = self.data
        if direction not in [-1,1]:
            direction = 1 #Should raise some error 
        # Remove na first 
        values = data[ col][ data[ col].notna()]
        #data = np.array( data[index: to] if to>-1 else data[index:] ) 
        data_size = values.size 
        print("data size" , str(data_size))
        #time = np.linspace( 0, data_size-1, data_size) #just a sequence 
        time = None
        if fraction: #apply a filter then convert to numpy array
            time = data['Date'].apply( lambda epoch: toYearFraction( epoch) ).values[-data_size:]
        else:
            time = np.arange( data_size )
        values = (values.values if direction==1 else np.flip(values, axis=0).values )
        dataSeries = [time, values]
        # Reset data size
        #self.data_size = data_size        
        return dataSeries
    def get_hourly_data( self, path = "2018"):
        
        c = Crawler()
        # Create a DataFrame from the crawled files
        # reset index pushes the date index into a column
        hourly_data = c.get_complete_df ( path, ['close'] ).reset_index()
        hourly_data['LogClose'] = hourly_data['close'].apply( lambda x: np.log(x) )
        hourly_data.columns = ['Date', 'Close', 'LogClose']
        hourly_data['StrDate'] = hourly_data['Date'].apply( lambda epoch: get_date_from_epoch( epoch) )
        self.data = hourly_data
        self.data_size = hourly_data.LogClose.size 
        return hourly_data
    def trim_by_date(self, date_from, date_to):
        """
        Example date: 2017-09-17 11:00:00
        """
        df = self.data
        df = df.loc[ df['Date'] >= get_epoch( date_from) ]
        df = df.loc[ df['Date'] <= get_epoch( date_to) ]
        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        self.data = df
        self.data_size = df['Date'].size
    def trim_by_date_and_count(self, date, count):
        """
        Example date: 2017-09-17 11:00:00
        Extract count number of data points from/to date
        +ve count from left (earlier) end. -ve from right (later) end
        """
        df = self.data
        if count>0: #Trip from the start date
            df = df.loc[ df['Date'] >= get_epoch( date) ]
            df = df.iloc[ :count ]
        else: #negative count. trim from the end date
            df = df.loc[ df['Date'] <= get_epoch( date) ]
            df = df.iloc[ count: ]

        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        self.data = df
        self.data_size = df['Date'].size
    def filter_by_date(self, date_from, date_to):
        """
        Example date: 2017-09-17 11:00:00
        Does not update the underlying data
        """
        df = self.data
        df = df.loc[ df['Date'] >= get_epoch( date_from) ]
        df = df.loc[ df['Date'] <= get_epoch( date_to) ]
        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        return df
    def filter_by_loc(self, f=0, t=-1):
        return self.data.iloc[f:t]
    def get_test_data(self):
        #Test data
        a = np.linspace(1,4,10).tolist() + np.linspace(4,3,5).tolist() + \
            np.linspace(3,7,10).tolist() + np.linspace(7,4,10).tolist()

        a += np.linspace(4,12,10).tolist() + np.linspace(12,10,5).tolist() \
            + np.linspace(10,15,10).tolist()
        df = DataFrame(a)
        df.columns = ['LogClose']
        #df.index = reversed(df.index)
        return df
    def get_data(self, path= "Data/cmc/daily.csv"):
        daily_data = pd.read_csv( path, sep='\t', parse_dates=['Date'], index_col= 'Date',
                                    names=[ 'Date', 'Open', 'High', 'Low', 'PirceClose', 'Volume', 'MarketCap'],
                                    header=0)

        #filter some dates
        #daily_data = daily_data.loc[daily_data.index >= '2015-01-01 00:00:00']
        daily_data = daily_data.loc[daily_data.index <= '2018-1-1 00:00:00']

        daily_data['Open'] = daily_data['Open'].apply(lambda x: float( x.replace(',','') ) )
        daily_data['High'] = daily_data['High'].apply(lambda x: float( x.replace(',','') ) )
        daily_data['Low'] = daily_data['Low'].apply(lambda x: float( x.replace(',','') ) )
        daily_data['PirceClose'] = daily_data['PirceClose'].apply(lambda x: float( x.replace(',','') ) )
        # Lppl works on log prices
        daily_data['LogClose'] = daily_data['PirceClose'].apply( lambda x: np.log(x) )
        #reverse index
        daily_data = daily_data.reset_index()
        daily_data.index = reversed(daily_data.index)
        #Index reset to a seq
        daily_data = daily_data.sort_index()
        print( daily_data.head() )
        self.data = daily_data
        return daily_data
    def save_to_file ( self, data, path, mode = "w+"):
        try:
            with open( path , mode) as f:
                f.write( data )
        except BaseException as e:
            print(e)
    def save_df ( self, df, path ):
        try:
            df.to_csv( path, sep='\t', encoding='utf-8' )
        except BaseException as e:
            print( e )

if __name__ == "__main__":
    # d = Data_Wrapper( data_source= 'SP500')
    # d.filter_by_date( "1984-1-3 00:00:00", "1984-1-10 00:00:00")
    d = Data_Wrapper( data_source= 'DIJA')
    d.filter_by_date( "1926-1-3 00:00:00", "1926-1-10 12:59:59")

class Epsilon_Drawdown:
    """
    Epsilon Drawdown Method developed by Johansen and Sornette (1998, 2001)
    and further used in (Johansen and Sornette, 2010; Filimonov and Sornette, 2015).
    """
    #__threshold = 0.1
    #__threshold = 0.6
    __DST = 0.65 #Short term threshold
    __DLT = 0.65 #Long term thresold
    #Originals Sornette Thresholds
    # __DST = 0.65 #Short term threshold
    # __DLT = 0.95 #Long term thresold
    #Only getters
    @property
    def short_threshold(self):
        return self.__DST
    
    @property 
    def long_threshold( self):
        return self.__DLT

    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, data):
        self.__data = data

    @property
    def data_size(self):
        return self.__data_size
    @data_size.setter
    def data_size(self, data_size):
        self.__data_size = data_size

    @property
    def col(self):
        return self.__col
    @col.setter
    def col(self, col):
        self.__col = col

    def e0_search_space(self):
        """
        Returns the epsilon E0 threshod search space. Currently [0.1:5]
        This is to incorporate the dynamics of realized return volatility
        in calculating the stopping tolerance for the drawups/downs
        """

        #Round to 1 decimal place
        return np.around( [i for i in np.arange( 5 , 6 , .3)], 1).tolist()
        # return np.around( [i for i in np.arange( 1 , 6 , .3)], 1).tolist()
        #return np.around( [i for i in np.arange( 1.1 , 1.3 , 0.1)], 1).tolist()
        # return np.around( [i for i in np.arange( 0.1 , 2.1, 0.1)], 1).tolist()
        #return np.around( [i for i in np.arange( 0.1 , 5.1, 0.1)], 1).tolist()

    def window_search_space(self):
        """
        The time window search space is used to calculate the sliding volatility 
        """
        return range( 72 ,361, 24) # 
        # return range( 12 ,241, 12) #20 different volatility windows 
        #return range( 10 ,61, 5) #Daily 

    def __init__ (self , data , col = 'LogClose'):
        """
        Will use data series instead of a column Dataframe
        """
        data = data[ data[ col].notna()] 
        #dates = data['Date'][:values.size]

        self.data_size =data[col].size# values.size 
        self.col = col
        self.data = data #pd.concat( [dates, values], axis=1)


    def volatility( self, i, window = 5):
        """
        calculate the sd of the data of the past window time intervals 
        starting from index i. If index smaller than window-1 only take the first 
        i+1 elements. 
        The standard deviation is not robust to outliers. TODO Implement a more robust way 
        such as: 
        https://quant.stackexchange.com/questions/30173/what-volatility-estimator-for-continuous-data-and-small-time-window
        """
        # Data are either taken from (i-window, i) if i>=window 
        # or (o, window) otherwise
        window_data = self.data[self.col].values[ (i- window if i> window-1 else 0): (i if i> window-1 else window)]
        vol = window_data.std()
        return 0.01 if np.isnan(vol) else vol
    def epsilon (self, e0, i, w):
        """
        Calculate the stop tolerance Epsilon E(e0,w)=e0*volatility(i, w)
        """
        return e0 * self.volatility(i, w)
    def get_peaks(self):
        epsilon_list = []
        #Grid search for different e0 and time windows
        for e0,w in [ (x,y) for x in self.e0_search_space() for y in self.window_search_space() ]:
            epsilon = (e0,w, self.epsilon( e0, w, w))
            epsilon_list.append( epsilon  )
        return epsilon_list
    def log_return(self, i):
        """
        r(i) = ln P[ti] - ln P[ti-1]; i = 1,2,... 
        """
        r = 0
        if i>0:
            r = self.data[self.col].values[i]-self.data[self.col].values[i-1]
        return r
    def __p(self, i0, i):
        """
        cum_log_return
        """
        #i,i0 = i-1, i0-1
        try:
            r = 0
            if i0>=0 and i>i0:
                r = self.data[self.col].values[i]-self.data[self.col].values[i0]
            return r
        except:
            print("stopped at i=", i)
            return 0
    def __argm (self, elements, func):
        l = np.array(elements)
        argm = func(l) 
        return argm
    def __argmax(self, elements):
        """
        Finding the index of the first occurence of the max item in a list
        Using numpy.argmax 
        """
        return self.__argm( elements, np.argmax)
    def __argmin(self, elements):
        return self.__argm( elements, np.argmin)
    def plot_delta(self, i0, i):
        deltas = [self.delta(i0, k) for k in range(i0+1,i+1)]
        plt.plot(deltas)
        plt.show()
        return deltas 
    def plot_logreturns(self, io, i):
        deltas = [self.log_return( k) for k in range(1,i)]
        plt.plot(deltas)
        plt.show()
        return deltas
    def delta( self, i0, i, drawup=True):
        """
        Drawup: max(Pi0,k)-Pi0,i for i0<=k<=i
        Drawdown: Pi0,i-min(Pi0,k) for i0<=k<=i
        """
        #i,i0 = i-1, i0-1
        d = 0
        #[x- self.data[i0] for x in self.__p_list [i0:i+1] ]
        # self.__p_list is pre-calculated. i0 is indexed at 0 to save space 
        # Extracting only the i-i0 elements 
        #local_p_list = self.__p_list[i0][:i-i0+1] 
        local_p_list = np.subtract( np.array( self.data[self.col][ i0:i+1]), \
                self.data[self.col].values[i0]).tolist()
        #[self.__p(i0,k) for k in range(i0+1, i+1)]
        if drawup:
            d = np.max(local_p_list )- self.__p(i0,i)
        else: #drawdown
            d = self.__p(i0,i)- np.min( local_p_list)
        return d
    def i1(self, i0, epsilon, drawup=True ):
        """
        Stop when delta exceeds threshold
        Then return argmax() in case of drawup or argmin otherwise
        """
        delta = 0
        i=0
        for i in range( i0+1, self.__data_size ):
            delta = self.delta(i0, i, drawup)
            if delta >= epsilon[i]: #Epsilon at each point depending on sliding window volatility 
                # print("breaking at ", i)
                break
        #local_p_list = ( self.__p_list[i0][:i-i0+1] if i0<self.data_size-1 else [0] )
        local_p_list = ( np.subtract( np.array( self.data[self.col][ i0:i+1]), \
        self.data[self.col].values[i0]).tolist() if i0<self.data_size-1 else [0] )

        #[x- self.data[i0] for x in self.__p_list [i0:i+1] ]
        i1 = (self.__argmax( local_p_list ) if drawup else self.__argmin( local_p_list ) )
        # Adding i0 as local_p_list starts from 0
        return np.asscalar( i0+ i1), i
    def peaks(self, epsilon, plot=False ):
        i=1
        drawup = first_drawup = True
        if self.log_return(i)<=0: #Drawup
            drawup = first_drawup = False
        #Alternate between drawup and drawdown
        # draws,breaks = [],[]
        draws = []
        # i1, br = self.i1(0, epsilon, drawup= drawup) #find the peak drawup
        i1, _ = self.i1(0, epsilon, drawup= drawup) #find the peak drawup
        while i < self.data_size-1: #find if a drawup or drawdown
            # print("Found ", ("drawup " if drawup else "drawdown "), "at ", str(i1))
            draws.append(i1)
            #breaks.append(br)
            # Increment and flip drawup
            i=i1+1
            if i==self.data_size:
                break
            #The algorithm alternates between drawup and drawdown
            drawup = not drawup 
            i1,_ = self.i1(i, epsilon, drawup= drawup) #find the peak drawup
        
        peaks = [draws[d] for d in range( (0 if first_drawup else 1) ,len(draws),2) ]
        
        if plot:
            #plt.plot(self.data.LogClose)
            #draw_points = [(d, self.data.LogClose[d]) for d in draws]
            #break_points = [(d, l.data.LogClose[d]) for d in breaks]
            # z = zip(*draw_points)
            # x= zip(*break_points)
            # plt.scatter(*z, color='blue')
            # plt.scatter(*x, color='red')
            # Peaks start from 0 for a drawup rally and from 1 for a drawdown rally
            #draw_points = [(d, self.data.LogClose[d]) for d in peaks]
            #z = zip(*draw_points)
            #colors = cm.rainbow(np.linspace(0, 1, 100)
            #colors = itertools.cycle(["r", "b", "g"])
            #plt.scatter(*z) #plot with random color
            #plt.scatter(*z, c = np.random.rand(3,1)) #plot with random color
            #show later
            #plt.show()
            pass
        return peaks
    def tpeaks(self, plot = False  ):
        """
        For each epsilon window pair find a list of peaks.
        Loop over threshold spectrum for each e0 from the e0_search_space() 
        Returns a 2-d list of peaks for each e0 and each window
        tp is estimated over a moving window of the past w time points
        """
        e0_space = self.e0_search_space()
        tpeaks =[]
        window_space = self.window_search_space()
        for e0 in e0_space: 
            ts = self.threshold_spectrum( e0)
            then = time.time()
            for window in range( len(window_space)) : 
                peaks = self.peaks( epsilon = ts[window], plot = plot ) #Supports sliding window
                tpeaks.append(peaks)
                print("Window %d run: %.3f sec" % (window, time.time() - then) )
                then = time.time()
            print("e0 run: ", e0)
        if plot:
            plt.show()
        return tpeaks
    def unique( self, tpeaks):
        u = set()
        for item in tpeaks:
            u = u.union( set(item) )
        return u
    def total_search_space(self):
        #Total no. of WindowxEpsilon search combinations
        return len( self.window_search_space() ) * len( self.e0_search_space() ) 
    def Ntpk (self, tpeaks):
        """
        Count the no. of times each element in the unique set of peaks appeared in tpeaks
        """
        u = self.unique( tpeaks)
        #counts is a list of occurences of the elements in u across all tpeaks
        counts = [0]*len(u)
        for counter, value in enumerate(u): 
            #enumerates iterates over the index(counter) and the value of a list
            count = 0
            for lst in tpeaks:
                if value in lst:
                    count +=1
            counts[counter] = count
        fractions = np.array(counts) / self.total_search_space()
        fractions = fractions.tolist()
        
        ntpk_tuples = [ (peak,fraction) for  peak, fraction  in  zip( u, fractions) ]
        return ntpk_tuples
    def volatility_spectrum(self, plot = False):
        """
        Calculates the volatility of each data point on a sliding window and using different 
        window sizes
        Returns a 2-d list: each row represents a different window size and contains volatility
        of all data points
        """
        window_space = self.window_search_space()
        v = [ ]
        for w,i in [(window, point) for window in  window_space for point in range( 0,self.data_size)]:
            #Find volatility for each window for each data point
            v.append( self.volatility( i,w) )
        print( "Max. volatility %.5f" % np.amax(v) )
        print( "min. volatility %.5f" % np.amin(v) )
        print( "Volatility sd %.5f" % np.std(v) )

        v = np.reshape( v, ( len( window_space) , self.data_size) )

        if plot:
            for i in range(len(v)): 
                plt.plot(v[i])
            plt.show()
        return v
    def threshold_spectrum( self, e0 ):
        """
        Epsilon(e0, w, i) = e0 * volatility(w, i)
        Or just multiplying e0 by the volatility spectrum
        """
        v = np.array( self.volatility_spectrum() )
        spectrum = np.multiply( v, e0 ).tolist()
        return spectrum
    def potential_bubble_points(self, ntpk, threshold ):
        """
        Potential long term bubble: Ntp,k>=DLT for k=1,...,Ntp
        Potential short term bubble: Ntp,k>=DST and k<DLT for k=1,...,Ntp.  Excluding potential long term 
        """
        lst = []
        #Sort by the first item of the ntpk tuple
        if threshold == self.__DST: # short term bubble
            #ntpk is a tuple list of peaks and their fractions
            # TODO avoid constructing a list and write a efficient code 
            lst = [ x[0] for x in ntpk if x[1] >= self.__DST and x[1] < self.__DLT ]
        elif threshold == self.__DLT:
            lst = [ x[0] for x in ntpk if x[1] >= self.__DLT ]
        else: #All
            lst = [ x[0] for x in ntpk if x[1] >= self.__DST]

        return lst
    def get_bubbles (self, threshold ):
        """
        Return: List of tuples
        """
        tp = self.tpeaks( plot = False )
        ntpk = self.Ntpk( tp)
        ntpk = sorted(ntpk, key=lambda x: x[0] )
        print("Ntpk:\n", ntpk)
        potential_bubbles = self.potential_bubble_points( ntpk, threshold )
        print("Potential bubbles:\n", potential_bubbles)
        points = [(d, self.data[self.col].values[d]) for d in potential_bubbles]
        dated_points = [(toYearFraction(self.data.Date.values[d]), self.data[self.col].values[d]) for d in potential_bubbles]
        return points, dated_points

# if __name__ == "__main__":
#     pass
    # potential_bubbles = l.potential_bubble( ntpk, l.short_threshold )
    # print(potential_bubbles)
    # draw_points2 = [(d, l.data.LogClose[d]) for d in potential_bubbles]
    # plt.scatter(*zip(*draw_points2) )

    # l.plot_delta(1, 10)
    #deltas = l.plot_delta(1, 250)long_threshold
    #i1 = l.i1drawup(0)
    # i1 = l.i1(0, drawup=True)
    # print("Argmax found:" + str(i1))
    # print( l.data.PriceClose[i1])
    # /df = DataFrame(deltas)
    # l.plot_logreturns( 1, df.LogClose.size)
    #l.peaks()
    #l.peaks()

    # l = Lagrange_regularizer()
    # l.getSSE_and_SSEN_as_a_func_of_dt( normed= True, plot= True)
    # sse, ssen, x,y = l.getSSE_and_SSEN_as_a_func_of_dt()
    # slope = l.LagrangeMethod( sse)
    # SSEL = l.obtainLagrangeRegularizedNormedCost(x, y, slope)
    # plt.plot(x, SSEL)
    # plt.show()

