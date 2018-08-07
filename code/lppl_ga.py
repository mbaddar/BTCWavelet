import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc, least_squares, minimize
import random
import pandas as pd
from pandas_datareader import data as pdr
from pandas import Series, DataFrame
import datetime
import itertools
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.cluster import KMeans

from epsilon import Data_Wrapper, Epsilon_Drawdown
from decomposition import Wavelet_Wrapper  
import time

def lppl (t,x): #return fitting result using LPPL parameters
    a = x[0]
    b = x[1]
    tc = x[2]
    m = x[3]
    c = x[4]
    w = x[5]
    phi = x[6]
    try:
        # return a + ( b*np.power( np.abs(tc-t), m) ) *(1 + ( c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
        return a + ( np.power( tc-t, m)) *\
               ( b + ( c * np.cos((w*np.log(  tc-t  ))-phi)) )
    except (BaseException, RuntimeWarning) :
        print( "(tc=%d,t=%d)"% (tc,t) )


class Pipeline:
    @property
    def data_wrapper(self):
        return self.__data_wrapper
    @data_wrapper.setter
    def data_wrapper( self, data_wrapper):
        self.__data_wrapper = data_wrapper

    def __init__ (self, data_Wrapper):
        self.data_wrapper = data_Wrapper

    def do_pass ( self, level =1 ):
        #Will modify wavelet to accept a series 
        if level > 0:
            wavelet = Wavelet_Wrapper( self.data_wrapper.data['LogClose'].tolist() , padding = False)
        #for level in range(1, wavelet.coeffs_size):
        # plt.close('all')
            recon = pd.DataFrame( wavelet.reconstruct( level ) , columns = ['LogClose'] )
        else:
            recon = self.data_wrapper.data
        # TODO bug this class depends on LogClose
        l = Epsilon_Drawdown( recon )
        # l.data.LogClose.plot()
        potential_bubble_points = l.get_bubbles ( l.long_threshold )
        # plt.scatter(*zip(*draw_points) )
        # plt.savefig("Bubblepoints-level-%2d.png" % level )
        #plt.show()
        return recon, potential_bubble_points


def cluster (points, n_clusters =2 ):
    kmeans = KMeans(init='k-means++', n_clusters= n_clusters, n_init=10).fit(points)
    return kmeans.cluster_centers_, kmeans.labels_


class Lppl_Wrapper:

    @property
    def sse(self):
        return self.__sse

    @sse.setter
    def sse(self, sse):
        self.__sse = sse

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __init__ ( self, model, sse):
        self.__model = model       
        self.__sse = sse 
        self.__tc = model[2]

    @property
    def tc(self):
        return self.__tc

    def generator (self, ts): #return fitting result using LPPL parameters
        """
        ts is a numpy array of time points starting from 0
        """
        #TODO check for errors. E.g. empty model
        #TODO equation needs review. phi or -phi? B and C are different from the paper 
        #TODO isert b inside the cos upper bracket 
        x = self.model

        values = lppl(ts, x)
        return values 

    def plot(self, ts, offset=0):
        # Offset is to plot it on a full TS 
        #plt.plot(ts + offset , list( self.generator(ts)[0] )  )
        ts = ts + offset
        print("plotting ts from point: ", ts[0])
        plt.plot(ts , list( self.generator(ts) )  )

def lpplc1 (t,x): #return fitting result using LPPL parameters
    a = x[0] 
    b = x[1]
    tc = x[2]
    m = x[3]
    c1 = x[4]
    w = x[5]
    c2 = x[6]
    try:
        # return a + ( b*np.power( np.abs(tc-t), m) ) *(1 + ( c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
        pwr = np.power( tc-t, m)
        fun = a + pwr * (b  + c1 * np.cos( ( w*np.log( tc-t ))) + c2 * np.sin( ( w*np.log( tc-t ))) )
        return fun
    except BaseException:
        print( "(tc=%d,t=%d)"% (tc,t) )

class Grid_Fit:
    @property
    def d(self):
        return self.__d
    @d.setter
    def d(self, d):
        self.__d = d

    @property
    def data_series(self):
        return self.__data_series
    @data_series.setter
    def data_series(self, data_series):
        self.__data_series = data_series
    
    def __init__(self, date1 = "2017-12-1 00:00:00", date2= "2017-12-5 00:00:00", data_source = 'BTC'):
        d = Data_Wrapper( hourly = True, data_source = data_source)
        d.filter_by_date( date1, date2)
        self.d = d
        self.data_series = d.get_data_series( direction = 1)


    def fi(self, tc, t, m):
        """
        Used for the matrix analytic solution of the linear parameters
        """
        return np.power( tc-t, m)

    def sum_fi(self, tc, m):
        return np.sum( np.power( tc-self.data_series[0], m) )  
    def sum_gi (self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ) )
    def sum_hi( self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply( np.power( tc-t, m) , np.sin( ( w*np.log( tc-t ))) ) )

    def sum_fi_square(self, tc, m):
        t = self.data_series[0]
        return np.sum( np.power( tc-t, 2*m) )  
    def sum_gi_square (self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.power( np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ), 2 ) )
    def sum_hi_square ( self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.power( np.multiply( np.power( tc-t, m) , np.sin( ( w*np.log( tc-t ))) ),2 ) )

    def sum_yi(self):
        return np.sum( self.data_series[1] )
    def sum_yi_fi(self, tc, m):
        t = self.data_series[0]
        return np.sum( np.multiply( self.data_series[1], np.power( tc-t, m) ) )  
    def sum_yi_gi(self, tc, m, w):
        t = self.data_series[0]
        return np.sum( np.multiply( self.data_series[1], np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ) ) )  
    def sum_yi_hi(self, tc, m, w):
        t = self.data_series[0]
        return np.sum( np.multiply( self.data_series[1], np.multiply( np.power( tc-t, m) , np.sin( ( w*np.log( tc-t ))) ) ) )  

    def sum_fi_gi(self, tc, m, w):
        t = self.data_series[0]
        return np.sum( np.multiply( np.power( tc-self.data_series[0], 2*m),  np.cos( ( w*np.log( tc-t ))) ) ) 
    def sum_gi_hi (self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply( np.multiply( np.power( tc-t, 2*m), np.cos( ( w*np.log( tc-t ))) ) ,np.sin( ( w*np.log( tc-t ))) ) ) 
    def sum_fi_hi( self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply ( np.power( tc-self.data_series[0], 2*m) , np.sin( ( w*np.log( tc-t )) ) ) ) 
    

    def linear_constraint (self, x):
        """
        Fits A, B, C1, C2 according to an analytic solution given tc, m, w
        Returns a vector of A, B, C1, C2
        """
        tc = x[2]
        m = x[3]
        w = x[5]
        from numpy.linalg import inv
        b = np.array( [self.sum_yi(), self.sum_yi_fi(tc, m), self.sum_yi_gi (tc, m, w), self.sum_yi_gi (tc, m, w) ] ).T
        # Symmetric 
        a = inv( np.array( [[ self.d.data_size, self.sum_fi(tc, m), self.sum_gi( tc, m, w), self.sum_hi( tc, m, w)],
                           [self.sum_fi( tc, m), self.sum_fi_square(tc, m), self.sum_fi_gi(tc, m, w), self.sum_fi_hi(tc, m, w)],
                           [self.sum_gi(tc, m, w), self.sum_fi_gi(tc, m, w), self.sum_gi_square( tc, m, w), self.sum_gi_hi( tc, m, w)],
                           [self.sum_hi( tc, m, w) , self.sum_fi_hi( tc, m, w) , self.sum_gi_hi ( tc, m, w) , self.sum_hi_square (tc, m, w)] ]
                    ) )
        sol = np.dot( a, b) # A, B, C1, C2
        return sol

    def conA (self, x):
        sol = self.linear_constraint(x)
        #limits = ( a, b, tc, m, c1, w, c2)
        obj = sol[0]-x[0] #for the minimize function to work. Objective must be 0. Or so I understand!
        return obj
    def conB (self, x):
        sol = self.linear_constraint(x)
        #limits = ( a, b, tc, m, c1, w, c2)
        obj = sol[1]-x[1] #for the minimize function to work. Objective must be 0. Or so I understand!
        return obj
    def conC1 (self, x):
        sol = self.linear_constraint(x)
        #limits = ( a, b, tc, m, c1, w, c2)
        obj = sol[2]-x[4] #for the minimize function to work. Objective must be 0. Or so I understand!
        return obj
    def conC2 (self, x):
        sol = self.linear_constraint(x)
        #limits = ( a, b, tc, m, c1, w, c2)
        obj = sol[3]-x[6] #for the minimize function to work. Objective must be 0. Or so I understand!
        return obj

    def sse(self, y, yest):
        """
        Both are array-like of the same shape (n,)
        Returns: Sum-squared error 
        """
        return  np.sum( np.power( y-yest, 2) )

    def objective(self, x):
        t = self.data_series[0]
        y = self.data_series[1]
        yest = lpplc1(t, x)
        # Reduce the SSE to help the minimization algorithm reach a local minma.
        # Still unsure about the global minima 
        obj = self.sse(y, yest)
        return obj

    def solve (self, con = True):
        """
        Set con = True. to search with the below constraints
        """
        a = (-np.inf, np.inf)
        b = (-np.inf, -0.001)
        tc = (self.d.data_size, 2*self.d.data_size)
        m = (0.1, 0.9)
        c1 = (-1, 1)
        w = (4,25)
        c2 = (-1, 1)
        limits = ( a, b, tc, m, c1, w, c2)
        x0 = np.array( [0, -1, self.d.data_size, 0.2, -.5, 13, .5])
        then = time.time()
        if con:
            conA = { 'type': 'eq', 'fun': self.conA }
            conB = { 'type': 'eq', 'fun': self.conB }
            conC1 = { 'type': 'eq', 'fun': self.conC1 }
            conC2 = { 'type': 'eq', 'fun': self.conC2 }
            options = { 'maxiter': 150000 ,'ftol': 1e-2}
            print("Minimizing...")
            # methods: SLSQP, 
            # Nelder-Mead: Simplex does not work in the below form
            # Bounded: L-BFGS-B, TNC, SLSQP
            solution = minimize( self.objective ,x0 ,method='SLSQP',bounds=limits,\
                                constraints=[conA, conB, conC1, conC2],\
                                options= options )
        else:
            solution = minimize( self.objective ,x0 ,method='SLSQP',bounds=limits,\
                                options= options )
            
        print( "Minimization completed in %.2f seconds" % (time.time()-then) )
        crash = (solution.x[2]- self.d.data_size)/24. 
        print( solution)
        return solution, crash 

    def plot_solution (self, con = True, col = 'LogClose'):
        plt.plot(self.d.data[ col ])
        self.data_series = self.d.get_data_series( direction = 1, col = col)
        print("New data size =", self.d.data_size )  
        solution, crash = self.solve ( con )
        model_data = lpplc1( self.data_series[0], solution.x )
        plt.plot( self.data_series[0], model_data )
        plt.xlabel("t - crash in %.2f days" % crash)
        plt.ylabel("Log P(t)")
        # solution = self.minimize ( data_series[0], data_series[1])
        #A , #B, Tc, m, c, omega, phi

def run( Test = False, date1 = "2017-11-1 00:00:00", date2= "2017-12-16 00:00:00" ):
    l = Grid_Fit("1999-11-1 00:00:00", "2017-12-16 00:00:00" )
    #l.plot_solution( con = True)
    d = Data_Wrapper( hourly = True)
    d.filter_by_date( date1, date2 )
    pl = Pipeline( d )

    #TODO Get bubble points. Uncomment
    # data, bubble_points = pl.do_pass (level = 0) 
    # #Add reconstructed data
    # d.data = data
    # print( bubble_points )
    plt.plot( d.data['LogClose'])
    # plt.scatter( *zip(*bubble_points) )
    # print("Bubble points size: ", len(bubble_points) )

    #for i in range( 1, len(bubble_points ) ):
    for i in range( 1, 2 ):
        _to = 200
        _from = 1
        # _to = bubble_points[i][0]
        # _from = bubble_points[i-1][0]
        print("Fitting data from peak point %d to point %d " % (_from, _to) )
        data_size = _to - _from + 1
        points = []
        # for step in range(0, data_size-240,12 ): #skip 12 time points and recalculate
        #TODO test range. Use the above 
        if data_size > 48:
            print("Data size:", data_size )
            #for step in range( 0, data_size - 24, 12 ): #skip 12 time points and recalculate
            for step in [0]: 
            #for step in range(0, data_size - 240, 12 ): #skip 12 time points and recalculate
                # respective minimum and maximum values ​​of the seven parameters fitting process
                data_series = d.get_data_series( index = _from+ step, to=_to,  direction = 1) 
                dt = data_series[0].size
                lppl = pl.model_lppl( data_series ) # returns 3 fits
                print("Step: %d, dt: %d" % (step, dt))
                #x = lppl.Population(limits, 20, 0.3, 1.5, .05, 4)
                points.append( (lppl[0].tc-dt, dt ) )
                for i in range(3):
                    lppl[i].plot( data_series[0], offset = step + _from )
        # plt.scatter( *zip( *points) )
        # plt.gca().set_xlabel('tc-t2')
        # plt.gca().set_ylabel('dt = t2-t1')
        # clusters, labels = cluster(points, 4)
        # plt.scatter( *zip( *clusters) )
        # plt.show(block=True)
        # print( metrics.silhouette_score(points, labels) )
        # print ( "Clusters: ", clusters )

if __name__ == "__main__":
    # start date 17/9/2013
    # plt.show(block=True)
    random.seed()
    l = Grid_Fit("2017-1-15 00:00:00", "2017-12-10 00:00:00" )
    #l = Grid_Fit("2017-06-15 00:00:00", "2017-12-10 00:00:00" )
    #l = Grid_Fit("2013-11-10 00:00:00", "2013-11-18 00:00:00" )
    wavelet = Wavelet_Wrapper( l.d.data['LogClose'].tolist() , padding = False)
    recon = pd.DataFrame( wavelet.reconstruct( level = 1 ) , columns = ['Recon'] )
    l.d.data['Recon'] = recon['Recon']
    # l.data_series = l.d.get_data_series( direction = 1)
    # print(" Dataseries shape =", l.data_series[0].shape) 
    l.plot_solution( con = True, col = 'LogClose') #Recon data size issue
    #pl = Pipeline( l.d )
    #lpplfit = pl.model_lppl( l.data_series ) # returns 3 fits
    plt.show()

