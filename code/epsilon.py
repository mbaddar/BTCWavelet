import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
import random
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
#yf.pdr_override() # <== that's all it takes :-)
from pandas import Series, DataFrame
import datetime
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
from crawlers.crawler import Crawler
import time
from time import sleep
from sklearn import linear_model

class Epsilon_Drawdown:
    """
    Epsilon Drawdown Method developed by Johansen and Sornette (1998, 2001)
    and further used in (Johansen and Sornette, 2010; Filimonov and Sornette, 2015).
    """
    __threshold = 0.1
    __threshold_Hourly = 0.6
    __DST = 0.65 #Short term threshold
    __DLT = 0.95 #Long term thresold

    def e0_search_space(self):
        """
        Returns the epsilon E0 threshod search space. Currently [0.1:5]
        This is to incorporate the dynamics of realized return volatility
        in calculating the stopping tolerance for the drawups/downs
        """
        return np.around( [i for i in np.arange( 0.1 , 5.1, 0.1)], 1).tolist()
        #return np.around( [i for i in np.arange( 0.1 , 5.1, 0.1)], 1).tolist()
    def window_search_space(self):
        """
        The time window search space is used to calculate the sliding volatility 
        """
        #return range( 24 ,241, 24)
        return range( 10 ,61, 5)
    
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

    def __init__ (self, path= "Data/cmc/daily.csv"):
        #self.__data = self.get_hourly_data( )
        self.__data = self.get_data( path)
        #self.__data = self.get_test_data()
        # A primitive way ot caching the log return p list
        self.__data_size = self.__data.LogClose.size
        #Optimize the return p_list starting from every item on the TS
        #self.__p_list = [self.__p(0,k) for k in range( 0, self.__data_size )]
        #retiring __2dplist
        # __2dplist = []
        # if p_list: 
        #     for i in range(0, self.__data_size):
        #         #p = [self.__p(i,k) for k in range( i, self.__data_size )]
        #         # p = [ self.data.LogClose[k]-self.data.LogClose[i] \
        #         #     for k in range( i, self.__data_size )]
        #         p = np.subtract( np.array( self.data.LogClose[i:self.__data_size]), \
        #         self.data.LogClose[i]).tolist()
        #         print( "Plist:%d of %d"% (i, self.__data_size) )
        #         __2dplist.append(p)
        #     self.__p_list = __2dplist
            # for l in __2dplist:
            #     print(len(l), " elements: [" , ",".join( [str("{0:.2f}".format(i)) for i in l ]), "]"  )
            #print("Done plist")

    def get_hourly_data( self):
        path = "2018"
        c = Crawler()
        # Create a DataFrame from the crawled files
        hourly_data = c.get_complete_df ( path, ['close'] ).reset_index()
        hourly_data['LogClose'] = hourly_data['close'].apply( lambda x: np.log(x) )
        self.data = hourly_data
        return hourly_data

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

    def volatility( self, i, window = 5):
        """
        calculate the sd of the data of the past window time intervals 
        starting from index i. If index smaller than window-1 only take the first 
        i+1 elements. 
        The standard deviation is not robust to outliers. TODO Implement a more robust way 
        such as: 
        https://quant.stackexchange.com/questions/30173/what-volatility-estimator-for-continuous-data-and-small-time-window
        """
        window_data = self.data.LogClose[ i-window if i> window-1 else 0:i]
        vol = window_data.std()
        return 0.01 if np.isnan(vol) else vol


    def epsilon (self, e0, i, w):
        """
        Calculate the stop tolerance Epsilon E(e0,w)=e0*volatility(i, w)
        """
        return e0 * self.volatility(i, w)

    def get_peaks(self):
        epsilon_list = []
        for e0,w in [ (x,y) for x in self.e0_search_space() for y in self.window_search_space() ]:
            epsilon = (e0,w, self.epsilon( e0, w, w))
            epsilon_list.append( epsilon  )
        return epsilon_list

    def log_return(self, i):
        """
        r(i) = ln P[ti] - ln P[ti-1]; i = 1,2,... 
        """
        #i = i-1
        r = 0
        if i>0:
            r = self.data.LogClose[i]-self.data.LogClose[i-1]
        return r
    
    def __p(self, i0, i):
        """
        cum_log_return
        """
        #i,i0 = i-1, i0-1
        try:
            r = 0
            if i0>=0 and i>i0:
                r = self.data.LogClose[i]-self.data.LogClose[i0]
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
        #[x- self.data.LogClose[i0] for x in self.__p_list [i0:i+1] ]
        # self.__p_list is pre-calculated. i0 is indexed at 0 to save space 
        # Extracting only the i-i0 elements 
        #local_p_list = self.__p_list[i0][:i-i0+1] 
        local_p_list = np.subtract( np.array( self.data.LogClose[ i0:i+1]), \
                self.data.LogClose[i0]).tolist()
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
        local_p_list = ( np.subtract( np.array( self.data.LogClose[ i0:i+1]), \
        self.data.LogClose[i0]).tolist() if i0<self.data_size-1 else [0] )

        #[x- self.data.LogClose[i0] for x in self.__p_list [i0:i+1] ]
        i1 = (self.__argmax( local_p_list ) if drawup else self.__argmin( local_p_list ) )
        # Adding i0 as local_p_list starts from 0
        return np.asscalar( i0+ i1), i

    def peaks(self, epsilon, plot=False ):
        i=1
        drawup = first_drawup = True
        if self.log_return(i)<=0: #Drawup
            drawup = first_drawup = False
        #Alternate between drawup and drawdown
        draws,breaks = [],[]
        i1, br = self.i1(0, epsilon, drawup= drawup) #find the peak drawup
        while i < self.data_size-1: #find if a drawup or drawdown
            # print("Found ", ("drawup " if drawup else "drawdown "), "at ", str(i1))
            draws.append(i1)
            breaks.append(br)
            # Increment and flip drawup
            i=i1+1
            if i==self.data_size:
                break
            #The algorithm alternates between drawup and drawdown
            drawup = not drawup 
            i1,br = self.i1(i, epsilon, drawup= drawup) #find the peak drawup
        
        peaks = [draws[d] for d in range( (0 if first_drawup else 1) ,len(draws),2) ]
        
        if plot:
            plt.plot(self.data.LogClose)
            draw_points = [(d, self.data.LogClose[d]) for d in draws]
            #break_points = [(d, l.data.LogClose[d]) for d in breaks]
            # z = zip(*draw_points)
            # x= zip(*break_points)
            # plt.scatter(*z, color='blue')
            # plt.scatter(*x, color='red')
            # Peaks start from 0 for a drawup rally and from 1 for a drawdown rally
            draw_points = [(d, self.data.LogClose[d]) for d in peaks]
            z = zip(*draw_points)
            #colors = cm.rainbow(np.linspace(0, 1, 100)
            #colors = itertools.cycle(["r", "b", "g"])
            plt.scatter(*z) #plot with random color
            #plt.scatter(*z, c = np.random.rand(3,1)) #plot with random color
            #show later
            #plt.show()
        return peaks

    def tpeaks(self, plot = False  ):
        """
        For each epsilon window pair find a list of peaks.
        Loop over threshold spectrum for each e0 from the e0_search_space() 
        Returns a 2-d list of peaks for each e0 and each window
        """
        e0_space = self.e0_search_space()
        tpeaks =[]
        window_space = self.window_search_space()
        for e0 in e0_space: #50 runs
            ts = self.threshold_spectrum( e0)
            then = time.time()
            for window in range( len(window_space)) : #11 runs
                peaks = self.peaks( epsilon = ts[window], plot = plot ) #Supports sliding window
                tpeaks.append(peaks)
                print("Window run: %.3f sec" % (time.time() - then) )
                then = time.time()
            print("e0 run")
        if plot:
            plt.show()
        return tpeaks

    def unique( self, tpeaks):
        u = set()
        for item in tpeaks:
            u = u.union( set(item) )
        return u

    def total_search_space(self):
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
        v = np.reshape( v, ( len( window_space) , self.data_size) )
        if plot:
            for i in range(len(v)): 
                # print(v[i])
                plt.plot(v[i])
            plt.show()
        return v
    
    def threshold_spectrum( self, e0 ):
        """
        Epsilon(e0, w, i) = e0 * volatility(w, i)
        Or just multiplying e0 by the volatility spectrum
        """
        v = np.array( self.volatility_spectrum() )
        return np.multiply( v, e0 ).tolist()

    def potential_bubble(self, ntpk, threshold ):
        """
        Potential long term bubble: Ntp,k>=DLT for k=1,...,Ntp
        Potential short term bubble: Ntp,k>=DST and k<DLT for k=1,...,Ntp.  Excluding potential long term 
        """
        lst = []
        if threshold == self.__DST: # short term bubble
            #ntpk is a tuple list of peaks and their fractions
            # TODO avoid constructing a list and write a efficient code 
            lst = [ x[0] for x in ntpk if x[1] >= self.__DST and x[1] < self.__DLT ]
        else:
            lst = [ x[0] for x in ntpk if x[1] >= self.__DLT ]

        return lst

class Lagrange_regularizer:
    """
    Copyright: G.Demos @ ETH-Zurich - Jan.2017
    Swiss Finance Institute
    SSRN-id3007070
    Guilherme DEMOS
    Didier SORNETTE
    """
    __B_max = 0 #Power law amplitude
    __m_range = np.arange(0,1,20).tolist()  #PL exponent [0,1]
    __omeaga_range = np.arange(4, 25, 2).tolist() #LP frequency [4,25]
    __D_min = 0.5 #Damping [0.5, inf]
    __O_min = 2.5 #No of oscillations [2.5, inf]

    def simulateOLS( self):
        """ Generate synthetic OLS as presented in the paper """
        nobs = 200
        X = np.arange(0,nobs,1)
        e = np.random.normal(0, 10, nobs)
        beta = 0.5
        Y = [beta*X[i] + e[i] for i in range(len(X))]
        Y = np.array(Y)
        X = np.array(X)
        Y[:100] = Y[:100] + 4*e[:100]
        Y[100:200] = Y[100:200]*8
        return X, Y

    def fitDataViaOlsGetBetaAndLine( self, X, Y):
        """ Fit synthetic OLS """
        beta_hat = np.dot(X.T,X)**-1. * np.dot(X.T,Y) # get beta
        Y = [beta_hat*X[i] for i in range(len(X))]
        # generate fit
        return Y

    def getSSE( self, Y, Yhat, p=1, normed=False):
        """
        Obtain SSE (chi^2)
        p -> No. of parameters
        Y -> Data
        Yhat -> Model
        """
        error = (Y-Yhat)**2. #SSE
        obj = np.sum(error) 
        if normed == False:
            obj = np.sum(error) 
        else:
            obj = 1/np.float(len(Y) - p) * np.sum(error)
        return obj

    def getSSE_and_SSEN_as_a_func_of_dt( self, normed=False, plot=False):
        """ Obtain SSE and SSE/N for a given shrinking fitting window w """
        # Simulate Initial Data
        X, Y = self.simulateOLS()
        # Get a piece of it: Shrinking Window
        _sse = []
        _ssen = []
        for i in range(len(X)-10): # loop t1 until: t1 = (t2 - 10):
            xBatch = X[i:-1]
            yBatch = Y[i:-1]
            YhatBatch = self.fitDataViaOlsGetBetaAndLine( xBatch, yBatch)
            sse = self.getSSE(yBatch, YhatBatch, normed=False)
            sseN = self.getSSE(yBatch, YhatBatch, normed=True)
            _sse.append(sse)
            _ssen.append(sseN)
        if plot == False:
            pass
        else:
            f, ax = plt.subplots( 1,1,figsize=(6,3) )
            ax.plot( _sse, color= 'k')
            a = ax.twinx()
            a.plot( _ssen, color='b')
            plt.tight_layout()
        if normed == False: 
            return _sse, _ssen, X, Y # returns results + data
        else:
            return _sse/max(_sse), _ssen/max(_ssen), X, Y # returns results + data
    ########################    

    def LagrangeMethod( self, sse):
        """ Obtain the Lagrange regulariser for a given SSE/N """
        # Fit the decreasing trend of the cost function
        slope = self.calculate_slope_of_normed_cost(sse)
        return slope[0]        

    def calculate_slope_of_normed_cost( self, sse):
        #Create linear regression object using statsmodels package
        regr = linear_model.LinearRegression( fit_intercept=False)
        # create x range for the sse_ds
        x_sse = np.arange(len(sse))
        x_sse = x_sse.reshape(len(sse),1)
        # Train the model using the training sets
        res = regr.fit(x_sse, sse)
        return res.coef_
    ########################

    def obtainLagrangeRegularizedNormedCost( self, X, Y, slope):
        """ Obtain the Lagrange regulariser for a given SSE/N Pt. III"""
        Yhat = self.fitDataViaOlsGetBetaAndLine(X,Y) # Get Model fit
        ssrn_reg = self.getSSE(Y, Yhat, normed=True) # Classical SSE
        ssrn_lgrn = ssrn_reg - slope*len(Y) # SSE lagrange
        return ssrn_lgrn
    
    def GetSSEREGvectorForLagrangeMethod( self, X, Y, slope):
        """
        X and Y used for calculating the original SSEN
        slope is the beta of fitting OLS to the SSEN
        """
        # Estimate the cost function pondered by lambda using a Shrinking Window.
        _ssenReg = []
        for i in range(len(X)-10):
            xBatch = X[i:-1]
            yBatch = Y[i:-1]
            regLag = self.obtainLagrangeRegularizedNormedCost(xBatch, yBatch, slope)
            _ssenReg.append(regLag)
        return _ssenReg


if __name__ == "__main__":
    l = Epsilon_Drawdown( )
    l.data.LogClose.plot()
    tp = l.tpeaks( plot = False )
    ntpk = l.Ntpk( tp)
    potential_bubbles = l.potential_bubble( ntpk, l.long_threshold )
    draw_points = [(d, l.data.LogClose[d]) for d in potential_bubbles]

    #plt.scatter(*zip(*draw_points) )

    potential_bubbles = l.potential_bubble( ntpk, l.short_threshold )
    print(potential_bubbles)
    draw_points2 = [(d, l.data.LogClose[d]) for d in potential_bubbles]
    plt.scatter(*zip(*draw_points2) )

    plt.show()
    # l.plot_delta(1, 10)
    #deltas = l.plot_delta(1, 250)long_threshold
    #i1 = l.i1drawup(0)
    # i1 = l.i1(0, drawup=True)
    # print("Argmax found:" + str(i1))
    # print( l.data.PriceClose[i1])
    # /df = DataFrame(deltas)
    # l.plot_logreturns( 1, df.LogClose.size)
    # plt.plot(l.data.LogClose)
    # plt.show()
    #l.peaks()
    #l.peaks()

    # l = Lagrange_regularizer()
    # l.getSSE_and_SSEN_as_a_func_of_dt( normed= True, plot= True)
    # sse, ssen, x,y = l.getSSE_and_SSEN_as_a_func_of_dt()
    # slope = l.LagrangeMethod( sse)
    # SSEL = l.obtainLagrangeRegularizedNormedCost(x, y, slope)
    # plt.plot(x, SSEL)
    # plt.show()