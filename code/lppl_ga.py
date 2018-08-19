import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm
from scipy.optimize import fmin_tnc, least_squares, minimize, basinhopping, differential_evolution
import random
import pandas as pd
from pandas_datareader import data as pdr
from pandas import Series, DataFrame
import datetime
import itertools
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.cluster import KMeans
import time
import sys, os
import struct
from numpy.linalg import inv

from epsilon import Data_Wrapper, Epsilon_Drawdown, to_year_from_fraction, str_from_date
from decomposition import Wavelet_Wrapper  

def format_func(x, pos):
    t= to_year_from_fraction(x)
    return t.strftime( "%Y-%m-%d" )

day_fraction = 0.0027
old_stdout = sys.stdout

def shift_to_log(file = "message.log" ):
    log_file = open( file,"w")
    sys.stdout = log_file
    return log_file


class Pipeline:
    @property
    def data_wrapper(self):
        return self.__data_wrapper
    @data_wrapper.setter
    def data_wrapper( self, data_wrapper):
        self.__data_wrapper = data_wrapper

    @property 
    def data_series(self):
        return self.__data_series
    @data_series.setter
    def data_series(self, data_series):
        self.__data_series = data_series 

    @property 
    def wavelet_data_series(self):
        return self.__wavelet_data_series
    @wavelet_data_series.setter
    def wavelet_data_series(self, wavelet_data_series):
        self.__wavelet_data_series = wavelet_data_series 

    @property 
    def nonlinear_fit(self):
        return self.__nonlinear_fit
    @nonlinear_fit.setter
    def nonlinear_fit(self, nonlinear_fit):
        self.__nonlinear_fit = nonlinear_fit 

    def __init__ (self, date1=None, date2=None, data_source = 'BTC', count =0 , hourly=True):
        self.data_wrapper = Data_Wrapper( hourly = hourly , data_source = data_source )
        if count !=0: #Count from both ends of the time series. The other end idetified by date
            # +ve count from left (earlier) end. -ve from right (later) end
            #Extract count no. of data points from/to date1
            self.data_wrapper.trim_by_date_and_count( date1 if count>0 else date2, count)
        elif date1 and date2:
            self.data_wrapper.trim_by_date( date1, date2)
        self.set_dataseries( )
        self.wavelet_recon( level= 1, fraction= 1)

    def set_dataseries(self, date1= None, date2=None, fraction = 1):
        d = self.data_wrapper
        data = d.data
        if date1 and date2: #Only if dates are entered 
            data = d.filter_by_date( date1, date2) #Return a df. Does not update the underlying data 
        data_series = d.get_data_series( data, direction = 1, fraction = fraction)
        self.data_series = data_series

    def set_dataseries_by_loc ( self, f=0, t=-1, fraction = 1):
        d = self.data_wrapper
        data_series = d.get_data_series( d.data.iloc[f:t] , direction = 1, fraction = fraction)
        self.data_series = data_series

    def wavelet_recon ( self, level =1, fraction=1):
        data_series = self.data_series
        wavelet = Wavelet_Wrapper( data_series[1].tolist() , padding = False)
        recon = wavelet.reconstruct( level = level )
        ap = data_series[1].size-recon.size 
        recon = pd.DataFrame( np.append( [np.nan]*ap , recon ), columns = ['Recon'] )
        self.data_wrapper.data['Recon']= recon
        return self.data_wrapper.data

    def wavelet_recon_by_loc ( self, f=0, t=-1, level =1, fraction=1):
        data_series = [ self.data_series[0][f:t], self.data_series[1][f:t] ]
        wavelet = Wavelet_Wrapper( data_series[1].tolist() , padding = False)
        recon = pd.DataFrame( wavelet.reconstruct( level = level ) , columns = ['Recon'] )
        data = self.data_wrapper.data.iloc[f:t]
        data['Recon'] = recon['Recon']
        self.wavelet_data_series =\
            self.data_wrapper.get_data_series( data, direction = 1, col='Recon', fraction =1)
        return self.wavelet_data_series

    def do_pass ( self, level =1 , plot=True ):
        #Will modify wavelet to accept a series 
        if level>1: #Level 1 precalculated
            self.wavelet_recon(level= level )
        recon = self.data_wrapper.data
        #     recon = self.data_wrapper.data
        # TODO bug this class depends on LogClose
        # recon = self.data_wrapper.data
        l = Epsilon_Drawdown( recon, col='LogClose' if level==0 else 'Recon')
        # l.data.LogClose.plot()
        #potential_bubble_points, plot_points = l.get_bubbles ( l.long_threshold )
        potential_bubble_points, plot_points = l.get_bubbles ( -1 ) #all
        plot_dates = [ ( to_year_from_fraction(point[0]).strftime( "%Y-%m-%d" ), point[1] )  for point in plot_points]
        print( plot_dates )

        plt.scatter(*zip(*plot_points), color='red' )
        if level>0:
            data_series = self.data_wrapper.get_data_series( recon, col='Recon')
        else:
            data_series = self.data_series

        plt.plot( data_series[0], data_series[1] )
        plt.title("BTC Peak Points")
        plt.xlabel("t")
        plt.ylabel("Ln P")
        # plt.savefig("Bubblepoints-level-%2d.png" % level )
        #plt.show()
        return potential_bubble_points
    
    def run(self, level = 0, Test = False ):
        # self.set_dataseries(date1, date2)
        # l = Nonlinear_Fit( self.data_series)
        #l.plot_solution( con = True)
        # d = Data_Wrapper( hourly = True)
        # d.filter_by_date( date1, date2 )
        # pl = Pipeline( d )

        #TODO Get bubble points. Uncomment
        #bubble_points = self.do_pass (level = 0) 
        # #Add reconstructed data
        # d.data = data
        # print( bubble_points )
        # plt.plot( self.data_series[0], self.data_series[1])
        self.do_pass(level=level)
        # plt.scatter( *zip(*bubble_points) )
        # print("Bubble points size: ", len(bubble_points) )

        #for i in range( 1, len(bubble_points ) ):
        # for i in range( 1, 2 ):
        #     _to = 200
        #     _from = 1
        #     # _to = bubble_points[i][0]
        #     # _from = bubble_points[i-1][0]
        #     print("Fitting data from peak point %d to point %d " % (_from, _to) )
        #     data_size = _to - _from + 1
        #     points = []
        #     # for step in range(0, data_size-240,12 ): #skip 12 time points and recalculate
        #     #TODO test range. Use the above 
        #     if data_size > 48:
        #         print("Data size:", data_size )
        #         #for step in range( 0, data_size - 24, 12 ): #skip 12 time points and recalculate
        #         for step in [0]: 
        #         #for step in range(0, data_size - 240, 12 ): #skip 12 time points and recalculate
        #             # respective minimum and maximum values ​​of the seven parameters fitting process
        #             data_series = d.get_data_series( index = _from+ step, to=_to,  direction = 1) 
        #             dt = data_series[0].size
        #             #lppl = pl.model_lppl( data_series ) # returns 3 fits
        #             print("Step: %d, dt: %d" % (step, dt))

                    #x = lppl.Population(limits, 20, 0.3, 1.5, .05, 4)
                    # points.append( (lppl[0].tc-dt, dt ) )
                    # for i in range(3):
                    #     lppl[i].plot( data_series[0], offset = step + _from )
            # plt.scatter( *zip( *points) )
            # plt.gca().set_xlabel('tc-t2')
            # plt.gca().set_ylabel('dt = t2-t1')
            # clusters, labels = cluster(points, 4)
            # plt.scatter( *zip( *clusters) )
            # plt.show(block=True)
            # print( metrics.silhouette_score(points, labels) )
            # print ( "Clusters: ", clusters )

def cluster (points, n_clusters =2 ):
    kmeans = KMeans(init='k-means++', n_clusters= n_clusters, n_init=10).fit(points)
    return kmeans.cluster_centers_, kmeans.labels_

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
        pwr =  np.power( tc-t, m)
        lg = np.log(  tc-t  )
        val = a + pwr * ( b +  c * np.cos( (w*lg)-phi) )
        return val
    except (BaseException, RuntimeWarning) :
        print( "(tc=%d,t=%d)"% (tc,t) )

def lpplc1tc (t,x): #return fitting result using LPPL parameters. C1 and C2 remove phi
    a = x[0] 
    b = x[1]
    tc = x[2] #tc days as a fraction of a year+ timehead. 
    m = x[3]
    c1 = x[4]
    w = x[5]
    c2 = x[6]
    try:
        # return a + ( b*np.power( np.abs(tc-t), m) ) *(1 + ( c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
        pwr = np.float_power( tc-t, m)
        lg = w*np.log( tc-t )
        values = a + pwr * ( b + c1* np.cos( lg)+ c2* np.sin( lg) )
        return values
    except BaseException:
        print( "(tc=%d,t=%d)"% (tc,t) )

def lpplc1 (t,x): #return fitting result using LPPL parameters. C1 and C2 remove phi
    a = x[0] 
    b = x[1]
    tc = x[2]/365+x[7] #tc days as a fraction of a year+ timehead. 
    m = x[3]
    c1 = x[4]
    w = x[5]
    c2 = x[6]
    try:
        # return a + ( b*np.power( np.abs(tc-t), m) ) *(1 + ( c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
        pwr = np.float_power( tc-t, m)
        lg = w*np.log( tc-t )
        values = a + pwr * ( b + c1* np.cos( lg)+ c2* np.sin( lg) )
        return values
    except BaseException:
        print( "(tc=%d,t=%d)"% (tc,t) )

class Nonlinear_Fit:
    def __init__(self, data_series = None ):
        self.data_series = data_series

    @property
    def data_series(self):
        return self.__data_series
    @data_series.setter
    def data_series(self, data_series):
        self.__data_series = data_series
    
    def fi(self, tc, t, m):
        """
        Used for the matrix analytic solution of the linear parameters
        """
        return np.power( tc-t, m)
    def sum_fi(self, tc, m): #(tc-t)^m
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
        return np.sum( np.multiply( np.power( tc-self.data_series[0], 2.*m),  np.cos( ( w*np.log( tc-t ))) ) ) 
    def sum_gi_hi (self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply( np.multiply( np.power( tc-t, 2.*m), np.cos( ( w*np.log( tc-t ))) ) ,np.sin( ( w*np.log( tc-t ))) ) ) 
    def sum_fi_hi( self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply ( np.power( tc-self.data_series[0], 2.*m) , np.sin( ( w*np.log( tc-t )) ) ) ) 
    
    def linear_constraint (self, x):
        """
        Fits A, B, C1, C2 according to an analytic solution given tc, m, w
        Returns a vector of A, B, C1, C2
        """
        tc = x[2]/365+x[7] #tc days as a fraction of a year+ timehead.
        m = x[3]
        w = x[5]
        from numpy.linalg import inv
        b = np.array( [self.sum_yi(), self.sum_yi_fi(tc, m), self.sum_yi_gi (tc, m, w), self.sum_yi_hi (tc, m, w) ] ).T
        sol = None
        mat = np.array( 
            [[ self.data_series[0].size, self.sum_fi(tc, m), self.sum_gi( tc, m, w), self.sum_hi( tc, m, w)],
            [self.sum_fi( tc, m), self.sum_fi_square(tc, m), self.sum_fi_gi(tc, m, w), self.sum_fi_hi(tc, m, w)],
            [self.sum_gi(tc, m, w), self.sum_fi_gi(tc, m, w), self.sum_gi_square( tc, m, w), self.sum_gi_hi( tc, m, w)],
            [self.sum_hi( tc, m, w) , self.sum_fi_hi( tc, m, w) , self.sum_gi_hi ( tc, m, w) , self.sum_hi_square (tc, m, w)] ]
        )
        try:
            # Symmetric 
            a = inv( mat )
            sol = np.dot( a, b) # A, B, C1, C2
        except BaseException as e:
            print( e ," Could not invert:" )
            print(mat)
        return sol
    
    def reduced_lppl( self, t, x): #x=[tc, m, w, timehead]
        linear_par = self.linear_constraint2( x ) 
        a, b, c1, c2 = linear_par[0], linear_par[1], linear_par[2], linear_par[3]
        tc = x[0]/365+ x[3] #tc days as a fraction of a year+ timehead. 
        m = x[1]
        w = x[2]
        try:
            # return a + ( b*np.power( np.abs(tc-t), m) ) *(1 + ( c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
            pwr = np.float_power( tc-t, m)
            lg = w*np.log( tc-t )
            values = a + pwr * ( b + c1* np.cos( lg)+ c2* np.sin( lg) )
            return values
        except BaseException:
            print( "(tc=%d,t=%d)"% (tc,t) )

    def linear_constraint2 (self, x ):
        """
        Fits A, B, C1, C2 according to an analytic solution given tc, m, w
        Returns a vector of A, B, C1, C2
        """
        tc = x[0]/365+x[3] #tc days as a fraction of a year+ timehead.
        m = x[1]
        w = x[2]
        b = np.array( [self.sum_yi(), self.sum_yi_fi(tc, m), self.sum_yi_gi (tc, m, w), self.sum_yi_hi (tc, m, w) ] ).T
        sol = None
        mat = np.array( 
            [[ self.data_series[0].size, self.sum_fi(tc, m), self.sum_gi( tc, m, w), self.sum_hi( tc, m, w)],
            [self.sum_fi( tc, m), self.sum_fi_square(tc, m), self.sum_fi_gi(tc, m, w), self.sum_fi_hi(tc, m, w)],
            [self.sum_gi(tc, m, w), self.sum_fi_gi(tc, m, w), self.sum_gi_square( tc, m, w), self.sum_gi_hi( tc, m, w)],
            [self.sum_hi( tc, m, w) , self.sum_fi_hi( tc, m, w) , self.sum_gi_hi ( tc, m, w) , self.sum_hi_square (tc, m, w)] ]
        )
        try:
            # Symmetric 
            a = inv( mat )
            sol = np.matmul( a, b) # A, B, C1, C2
        except BaseException as e:
            print( e ," Could not invert:" )
            print(mat)
        return sol

    def conA (self, x):
        sol = self.linear_constraint(x)
        #limits = ( a, b, tc, m, c1, w, c2)
        obj = sol[0]-x[0] #for the minimize function to work. Objective must be 0. 
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
    def mse(self, y, yest):
        """
        Both are array-like of the same shape (n,)
        Returns: Mean-squared error 
        """
        return  np.sum( np.power( y-yest, 2) )/y.size
    def sse(self, y, yest):
        """
        Both are array-like of the same shape (n,)
        Returns: Mean-squared error 
        """
        return  np.sum( np.power( y-yest, 2) )
    
    def rms(self, y, yest):
        """
        Both are array-like of the same shape (n,)
        Returns: Sum-squared error 
        """
        return  np.sqrt( np.sum( np.power( y-yest, 2) ) /y.size )
    def objective(self, x):
        t = self.data_series[0]
        y = self.data_series[1]
        yest = lpplc1(t, x)
        # Reduce the SSE to help the minimization algorithm reach a local minma.
        # Still unsure about the global minima 
        #obj = self.sse(y, yest)
        # Experimenting: constraints should be 0
        # obj = self.rms(y, yest) 
        obj = self.sse(y, yest) 
        return obj

    def objective2(self, x): #x=[tc, m, w, timehead]
        t = self.data_series[0]
        y = self.data_series[1]
        yest = self.reduced_lppl(t, x)
        # Reduce the SSE to help the minimization algorithm reach a local minma.
        # Still unsure about the global minima 
        #obj = self.sse(y, yest)
        # Experimenting: constraints should be 0
        # obj = self.rms(y, yest) 
        obj = self.sse(y, yest) 
        return obj

    def solve2 (self, method = 'basinhopping', niter = 10 ):
        """
        Returns: Solution, crash date, tc in no. of days
        """
        rnd = struct.unpack("<I", os.urandom( 4 ))[0]
        np.random.seed( rnd )

        def print_fun(x, f, accepted):
            print("New minima %.8f %s. Solution: Crash=%.2f days from %s, m=%.2f ,w=%.2f ." 
                %(f, "accepted" if accepted else "not accepted" ,
                 x[0] , to_year_from_fraction(x[3]), x[1], x[2] ) )
        # Now tc represet a fractional year instead of a time index
        data_size = self.data_series[0].size
        time_head = self.data_series[0][data_size-1]+day_fraction 
        tc = (0, 100 ) #look 3 months in advance
        m = (0.1, 0.9)
        w = (3,25)
        
        class MyTakeStep(object):
            def __init__(self, stepsize=0.5): #0.5 for hourly
                self.stepsize = stepsize
            def __call__(self, x):
                #s = self.stepsize
                x[0] += np.random.uniform( -10, 10) #tc
                x[1] += np.random.uniform( -.8, .8) #m
                x[2] += np.random.uniform( -22, 22 ) #w
                return x    

        limits = ( tc, m, w, (time_head,time_head) )
        bounds = list(limits) 
        x0 = np.array( 
            [ 
             np.random.uniform(0,100 ), #tc
             np.random.uniform( m[0], m[1] ), #m
             np.random.uniform( w[0], w[1]) , #w 
             time_head] ) 
        options = { 'maxiter': 10000 ,'ftol': 1e-8 }
        minimizer_kwargs = { "method": "SLSQP",
                             "options": options, "bounds": limits
                           } 
        mytakestep = MyTakeStep()
        if method == 'basinhopping':
            # print("x0: [" , ", ".join( [str("{0:.3f}".format(i)) for i in x0 ]), "]"  )
            solution = basinhopping( self.objective2, x0, minimizer_kwargs=minimizer_kwargs,
                        T= 1 , niter=niter, take_step = mytakestep)# ,callback= print_fun)   
        else: 
            solution = \
                differential_evolution( self.objective2, bounds= bounds, 
                    tol=1e-6, maxiter=1000, popsize = 50, seed = struct.unpack("<I", os.urandom( 4 ))[0]
                    )        
        # Now crash is a real point in time
        x = solution.x #tc,m,w,timehead
        crash = x[0]/365 + x[3] 
        lp = self.linear_constraint2(x) #[a,b,c1,c2]
        #print("x0: [" , ", ".join( [str("{0:.3f}".format(i)) for i in x0 ]), "]"  )
        print( "Solution: A= %.3f, B=%.3f, ,c1=%.2f, c2=%.2f ,Crash=%.2f days/%s, m=%.2f  ,w=%.2f. Cost=%.5f" 
                %(lp[0], lp[1], lp[2], lp[3], x[0], to_year_from_fraction(crash) , x[1],x[2], solution.fun))
        return solution, crash, x[0] 

    def solve (self, method = 'basinhopping', niter = 10 ):
        rnd = struct.unpack("<I", os.urandom( 4 ))[0]
        np.random.seed( rnd )

        def print_fun(x, f, accepted):
            print("New minima %.8f %s. Solution: A= %.3f, B=%.3f, Crash=%.2f days from %s, m=%.2f ,c1=%.2f ,w=%.2f ,c2=%.2f." 
                %(f, "accepted" if accepted else "not accepted" ,
                 x[0], x[1], x[2] , to_year_from_fraction(x[7]), x[3], x[4], x[5], x[6]) )
        a = (.5, 1000)
        b = (-1000, -0.001)
        # Now tc represet a fractional year instead of a time index
        data_size = self.data_series[0].size
        time_head = self.data_series[0][data_size-1]+ day_fraction 
        tc = (0, 100 ) #look 3 months in advance
        m = (0.1, 0.9)
        c1 = c2 =  (-10, 10)
        w = (3,25)
        
        class MyTakeStep(object):
            def __init__(self, stepsize=0.5): #0.5 for hourly
                self.stepsize = stepsize
            def __call__(self, x):
                #s = self.stepsize
                x[0] += np.random.uniform( -100, 100) #a
                x[1] += np.random.uniform( -100, 100) #b
                x[2] += np.random.uniform( -10, 10 ) #moves 1 day +/-. Tc now represents days
                x[3] += np.random.uniform( -.8, .8) #m
                x[4] += np.random.uniform( -2, 2) #c1
                x[5] += np.random.uniform( -22, 22) #w
                x[6] += np.random.uniform( -2, 2 ) #c2
                #print("Step: [" , ", ".join( [str("{0:.3f}".format(i)) for i in x ]), "]"  )
                return x    

        limits = (a, b, tc, m, c1, w, c2, (time_head,time_head))
        bounds = list(limits) 
        # bounds = [(.5, 10000), (-10000, -0.001), (time_head, time_head +.4 )
        # , (0.1, 0.9), (-100, 100), (3,25), (-100, 100) ]
        #initial guess inline with what parameters represent
        x0 = np.array( 
            [  np.random.uniform( a[0], np.average(self.data_series[1]) ) , #a
             - np.random.uniform( b[1], (np.amax(self.data_series[1])-np.amin(self.data_series[1]) ) ), #b
             0, #tc
             np.random.uniform( m[0], m[1] ), #m
             np.random.uniform( c1[0], c1[1] ), #c1
             np.random.uniform( w[0], w[1]) , #w 
             np.random.uniform( c2[0], c2[1] ), #c2
             time_head]) #adding new parameter to fix the initial tc
        constraints = [ { 'type': 'eq', 'fun': self.conA },
                        { 'type': 'eq', 'fun': self.conB },
                        { 'type': 'eq', 'fun': self.conC1 },
                        { 'type': 'eq', 'fun': self.conC2 },
                    ]
        options = { 'maxiter': 500 ,'ftol': 1e-4 }
        print(method, " Minimizing..." )
        minimizer_kwargs = { "method": "SLSQP", "constraints": constraints ,
                             "options": options, "bounds": limits
                           } 
        mytakestep = MyTakeStep()
        # mybounds = MyBounds( xmin, xmax )
        if method == 'basinhopping':
            print("x0: [" , ", ".join( [str("{0:.3f}".format(i)) for i in x0 ]), "]"  )
            solution = basinhopping( self.objective, x0, minimizer_kwargs=minimizer_kwargs,
                        T= 1 , niter=niter, take_step = mytakestep, callback= print_fun )   
        elif method == 'SLSQP':     
            solution = minimize( self.objective ,x0 ,method='SLSQP',bounds=limits,\
                                constraints=constraints,\
                                options= options )
        else: 
            solution = \
                differential_evolution( self.objective, bounds= bounds, 
                    tol=1e-6, maxiter=1000, popsize = 50, seed = struct.unpack("<I", os.urandom( 4 ))[0]
                    )        
        # Now crash is a real point in time
        x = solution.x
        crash = x[2]/365 + x[7] 
        #print("x0: [" , ", ".join( [str("{0:.3f}".format(i)) for i in x0 ]), "]"  )
        print( "Solution: A= %.3f, B=%.3f, Crash=%.2f days/%s, m=%.2f ,c1=%.2f ,w=%.2f ,c2=%.2f. Cost=%.5f" 
                %(x[0], x[1], x[2], to_year_from_fraction(crash) , x[3],x[4],x[5],x[6], solution.fun))

        # print( "ConA: ", str( self.conA(solution.x)) )
        # print( "ConB: ", str( self.conB(solution.x)) )
        # print( "ConC1: ", str( self.conC1(solution.x)) )
        # print( "ConC2: ", str( self.conC2(solution.x)) )
        return solution, crash, x[2]  
    def plot_solution (self, scale = 1, method= 'basinhopping', niter= 10, lppl_func = lpplc1 ):
        """
        Scale represents the number of hours 
        """
        # self.data_series = self.d.get_data_series( direction = 1, col = col, fraction= 1)
        solution, crash_time, crash = self.solve ( method = method , niter= niter)
        model_data = lpplc1( self.data_series[0], solution.x )
        crash_time = to_year_from_fraction( crash_time).strftime( "%d/%m/%Y" )
        label = method + " optimization - crash date: " + crash_time
        plt.plot( self.data_series[0], model_data, label= label )
        #plt.legend(loc='upper center',shadow=True, fontsize='medium')
        plt.xlabel("t" )
        plt.ylabel("Ln P")
        #A , #B, Tc, m, c, omega, phi
        return crash

    def plot_solution2 (self, scale = 1, method= 'basinhopping', niter= 10 ):
        """
        Scale represents the number of hours 
        """
        # self.data_series = self.d.get_data_series( direction = 1, col = col, fraction= 1)
        solution, crash_time, crash = self.solve2 ( method = method , niter= niter)
        model_data = self.reduced_lppl( self.data_series[0], solution.x )
        # label = method + " optimization - crash date: " + to_year_from_fraction( crash_time).strftime( "%d/%m/%Y" )
        # plt.plot( self.data_series[0], model_data ,label="LPPL Fit", color="blue") #, label= label
        #plt.legend(loc='upper center',shadow=True, fontsize='medium')
        # plt.xlabel("t" )
        # plt.ylabel("Ln P")
        #A , #B, Tc, m, c, omega, phi
        return crash_time
    def plot_cost (self, col = 'LogClose'):
        # from mpl_toolkits.mplot3d import Axes3D
        # TODO Need maint
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        # self.data_series = self.get_data_series( direction = 1, col = col)
        solution, _, _ = self.solve( )
        x = solution.x
        print(solution)
        tc = x[2]
        # tc_range = np.linspace(self.d.data_size, 1.2*tc, 500)        
        m_range = np.linspace(.1, .9, 100)     
        w_range = np.linspace(3, 25, 100)
        M,W = np.meshgrid( m_range, w_range )
        obj = np.zeros(10000)
        index =0
        for m,w in [(m,w) for m in m_range for w in w_range]:
            x[3]=m
            x[5]=w
            obj[index]=self.objective(x)
            index+=1
        print("min", np.min(obj))
        obj = obj.reshape(100,100)
        #CS = plt.contour(M, W, obj)
        #CS = plt.contour(, np.arange(10), np.power( np.arange(10), 2))
        #plt.clabel(CS, inline=1, fontsize=10)
        #plt.title('Objective Contour')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(M, W, obj, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)



        # for i,tc in enumerate(tc_range):
        #     x[2] = tc
        #     obj[i] = self.objective(x)
        # plt.plot(tc_range, obj)

def describe( data):
    df = DataFrame( data )
    print(df.describe())


def omx30():
    date1, date2 = "1996-4-20 00:00:00", "1998-8-15 00:00:00"
    p = Pipeline( date1, date2, data_source='OMXS30', hourly=False, count=0)
    data_series = p.data_series
    # plt.plot( data_series[0], data_series[1], label='Data' )
    # plt.xlabel("t")
    # plt.ylabel("Ln P")
    # plt.title("OMX3 Crash")
    print("Actual crash: ", to_year_from_fraction(1998.686) )
    crashes = []
    # dt = []
    for _ in np.arange(5):
        for day in np.arange(1, 60): #Advance the start date and fit
            ds = [ data_series[0][day:], data_series[1][day:] ]
            l = Nonlinear_Fit ( ds)
            # crash = l.plot_solution( method= 'differential_evolution', niter=5 )
            crash = l.plot_solution2( method= 'basinhopping', niter=20 )
            crashes.append( crash)
            # dt.append( data_series[0][day:].size )  

    # for crash ,dt in crashes:
        # print("Crash time %s, for period of %d" % (to_year_from_fraction( crash), dt))
    
    # plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )
    # plt.show()
    # plt.close('all')
    describe( crashes )
    hist( crashes )
    print("Actual crash: ", to_year_from_fraction(1998.686) )
    plt.show()

def hist( data):
    plt.hist( data )

def scatter_plot(crashes):
    """
    List of tuples
    """
    plt.scatter( *zip( *crashes) )
    plt.gca().set_xlabel('tc')
    plt.gca().set_ylabel('dt = t2-t1')
    plt.title("Crash times Cluster. Actual crash on %s" % str_from_date( to_year_from_fraction(1998.686) ) )
    plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )

def synthetic_stats( trials=1000):
    x = [ 569.988, -266.943, 1980.218, 0.445 , 8.186642647	,7.877 ,-11.65390262]
    t = np.linspace(1977.5, 1980, 1000)
    v = lpplc1tc(t, x) + np.random.normal(0, 10, t.shape)

    data_series = [t, v]

    l = Nonlinear_Fit ( data_series)
    crashes = np.zeros( trials)
    for i in np.arange( trials):
        crashes[i] = l.plot_solution2( method= 'basinhopping', niter=10)
    df = DataFrame( crashes, columns = ['Crash time'])
    print(df.describe())

def synthetic_trial():
    """
    Synthetic trial from LPPL model with added normal random noise
    Data from Dija 1929 crash and time was advanced 50 years to avoid   
    dealing with dates before the epoch starting at 1970 
    """
    # x= [ 9.554, -2.931, 29.42 ,0.75 ,0.16 ,12.74 ,-0.20,  ]#
    x = [ 569.988, -266.943, 1980.218, 0.445 , 8.186642647	,7.877 ,-11.65390262]
    t = np.linspace(1977.5, 1980, 1000)
    v = lpplc1tc(t, x) + np.random.normal(0, 10, t.shape)

    data_series = [t, v]

    l = Nonlinear_Fit ( data_series)
    crash = l.plot_solution2( method= 'basinhopping', niter=30)
    print( str_from_date( to_year_from_fraction(1980.218) ) )
    plt.plot(t, v, label="Sythesized Data", color="red" )
    plt.xlabel( "t - Actual crash on %s" % str_from_date( to_year_from_fraction(1980.218) ) )
    plt.ylabel("Ln P")
    plt.title("Synthetic trial. Predicted crash on %s" % str_from_date( to_year_from_fraction( crash ) ) )
    plt.legend(loc='upper left',shadow=True, fontsize='medium')
    plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )

    plt.show()
    # cost = []
    # for tc in np.linspace( x[2]/2, 1.5*x[2], 100):
    #     x[2] = tc
    #     cost.append( np.sum( np.power( lpplc1( data_series[0], x)-data_series[1] , 2) ) )
    # plt.plot( np.linspace( x[2]/2, 1.5*x[2], 100) , cost)
    # print( np.amin(cost) )
    # plt.show()
    # 

def Btc_trial_daily():
    date1, date2 = "2017-5-1 00:00:00", "2017-11-15 00:00:00"
    p = Pipeline( date1, date2, data_source='BTC', hourly=False, count=0)
    data_series = p.data_series
    # plt.plot( data_series[0], data_series[1], label='Data' )
    # plt.xlabel("t")
    # plt.ylabel("Ln P")
    # plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )
    crashes = []
    # plt.show()

    for _ in np.arange(1, 5): #Advance the start date and fit
        for day in np.arange(1, 20): #Advance the start date and fit
            ds = [ data_series[0][day:], data_series[1][day:] ]
            l = Nonlinear_Fit ( ds)
            # crash = l.plot_solution( method= 'differential_evolution', niter=5 )
            crash = l.plot_solution2( method= 'basinhopping', niter=30 )
            crashes.append( ( crash, data_series[0][day:].size) ) 

    # for crash ,dt in crashes:
    #     print("Crash time %s, for period of %d" % (to_year_from_fraction( crash), dt))
    describe( crashes)
    plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )
    # plt.close('all')
    plt.scatter( *zip( *crashes) )
    plt.gca().set_xlabel('tc')
    plt.gca().set_ylabel('dt = t2-t1')
    plt.title("Crash times Cluster. Actual crash on %s" % str_from_date( datetime.datetime(2017,12,17) ) )
    plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )
    plt.show()

if __name__ == "__main__":
    #Btc_trial_daily()
    #synthetic_trial()
    synthetic_stats( 100)

if __name__ == "__main__1":
    # start date 17/9/2013
    # plt.show(block=True)
    wavelet_flag = False
    date1, date2 = "2017-1-15 00:00:00", "2017-12-1 00:00:00"
    #p = Pipeline(date1, date2, data_source='BTC', count=32768)
    # p = Pipeline(date1, date2, data_source='BTC', hourly=False, count=-16384)
    p = Pipeline( date1, date2, data_source='BTC', hourly=False, count=0)
    p.run(level=1)
    # log_file = shift_to_log( "message-%d.log" %int(time.time()) )
    crashes = []
    #wavelet recon
    # data_series = p.data_wrapper.get_data_series( p.data_wrapper.data , col='Recon')
    data_series = p.data_series
    plt.plot( data_series[0], data_series[1], label='Data' )
    l = Nonlinear_Fit ( data_series)
    l.plot_solution2( method= 'basinhopping', niter=30)
    plt.xlabel("t")
    plt.ylabel("Ln P")
    plt.title("Reduced Solution")
    plt.show()
    # x = solution.x
    #A= 9.554, B=-2.931, Crash=29.42 m=0.75 ,c1=0.16 ,w=12.74 ,c2=-0.20.
        
    # plt.plot( data_series[0], data_series[1], label='Tc=29 days' )
    # x= [ 9.554, -2.931, 9.42 ,0.75 ,0.16 ,12.74 ,-0.20, data_series[0][data_series[0].size-1] ]
    # data_series[1] = lpplc1(data_series[0], x)
    # plt.plot( data_series[0], data_series[1], label='Tc=9 days' )
    # x= [ 9.554, -2.931, 59.42 ,0.75 ,0.16 ,12.74 ,-0.20, data_series[0][data_series[0].size-1] ]
    # data_series[1] = lpplc1(data_series[0], x)
    # plt.plot( data_series[0], data_series[1], label='Tc=59 days' )
    # plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )
    # plt.xlabel("t")
    # plt.ylabel("Ln P")
    # plt.title("LPPL fit of differenet Tc")
    # plt.legend()
    # plt.show()
    # for day in 24*np.arange(1, 76, 5): #Advance the start date and fit
    # for day in np.arange(1, 20): #Advance the start date and fit
    #     ds = [ data_series[0][day:], data_series[1][day:] ]
    #     l = Nonlinear_Fit ( ds)
    #     # crash = l.plot_solution( method= 'differential_evolution', niter=5 )
    #     crash = l.plot_solution( method= 'basinhopping', niter=30 )
    #     crashes.append( ( crash, data_series[0][day:].size) ) 

    # for crash ,dt in crashes:
    #     print("Crash in %.2f days, for period of %d" % (crash, dt))
    
    # plt.show()
    # # plt.close('all')
    # plt.scatter( *zip( *crashes) )
    # plt.gca().set_xlabel('tc-t2')
    # plt.gca().set_ylabel('dt = t2-t1')

    # clusters, labels = cluster(points, 4)
    # plt.scatter( *zip( *clusters) )
    # plt.show(block=True)
    # print( metrics.silhouette_score(points, labels) )
    # print ( "Clusters: ", clusters )

    # plt.title("BTC LPPL Fit - 10 Different Periods")
    # log_file.close()
    sys.stdout = old_stdout
    # plt.show()

    # if wavelet_flag:
    #     l.wavelet_recon()
    #     l.plot_solution( method= 'basinhopping' )
    # else:
    #     # l.plot_solution(col = 'LogClose', method= 'basinhopping' )
    #     l.plot_solution( method= 'basinhopping' )
    #plt.plot(p.data_series[0], p.data_series[1])
    #plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))

