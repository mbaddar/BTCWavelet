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

from epsilon import Data_Wrapper, Epsilon_Drawdown, to_year_from_fraction
from decomposition import Wavelet_Wrapper  

def format_func(x, pos):
    t= to_year_from_fraction(x)
    return t.strftime( "%Y-%m-%d" )

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

    def __init__ (self, date1=None, date2=None, data_source = 'BTC' ):
        self.data_wrapper = Data_Wrapper( hourly = True, data_source = data_source )
        if date1 and date2:
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
        print( plot_points )

        plt.scatter(*zip(*plot_points), color='red' )
        if level>0:
            data_series = self.data_wrapper.get_data_series( recon, col='Recon')
        else:
            data_series = self.data_series

        plt.plot( data_series[0], data_series[1] )

        # plt.savefig("Bubblepoints-level-%2d.png" % level )
        #plt.show()
        return potential_bubble_points
    
    def run(self, Test = False ):
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
        self.do_pass(level=0)
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

if __name__ == "__main__":
    # start date 17/9/2013
    # plt.show(block=True)
    wavelet_flag = False
    random.seed()
    # plt.plot(l.data_series[0], l.data_series[1])
    date1, date2 = "2017-11-01 00:00:00", "2017-12-10 00:00:00"
    #p = Pipeline(date1, date2, data_source='BTC')
    p = Pipeline()
    p.run()
    #plt.plot(p.data_series[0], p.data_series[1])
    #plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    plt.gca().xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_func) )
    plt.show()
    # l = Nonlinear_Fit()
    # l.set_dataseries (date1=date1, date2= date2, data_source= 'BTC')
    # #l = Nonlinear_Fit("1984-07-30 00:00:00", "1987-06-12 00:00:00" , data_source= 'SP500')
    # # l = Nonlinear_Fit("1984-09-21 00:00:00", "1987-08-06 00:00:00" , data_source= 'SP500')
    # if wavelet_flag:
    #     l.wavelet_recon()
    #     l.plot_solution( method= 'basinhopping' )
    # else:
    #     # l.plot_solution(col = 'LogClose', method= 'basinhopping' )
    #     l.plot_solution( method= 'basinhopping' )



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
        lg = w*np.log( tc-t )
        fun = a + pwr * (b  + c1 * np.cos( lg ) + c2* np.sin( lg ) )
        return fun
    except BaseException:
        print( "(tc=%d,t=%d)"% (tc,t) )

class Nonlinear_Fit:
    # @property
    # def d(self):
    #     return self.__d
    # @d.setter
    # def d(self, d):
    #     self.__d = d
    def __init__(self, data_series = None ):
        #self.data_wrapper = data_wrapper
        self.data_series = data_series

    @property
    def data_series(self):
        return self.__data_series
    @data_series.setter
    def data_series(self, data_series):
        self.__data_series = data_series
    
    @property
    def data_wrapper(self):
        return self.__data_wrapper
    @data_wrapper.setter
    def data_wrapper(self, data_wrapper):
        self.__data_wrapper = data_wrapper

    # def set_dataseries(self, date1 = "2017-12-1 00:00:00", date2= "2017-12-5 00:00:00", data_source = 'BTC', fraction = 1):
    #     #Move up
    #     d = Data_Wrapper( hourly = True, data_source = data_source )
    #     d.filter_by_date( date1, date2) #returns None. updates the underlying data 
    #     # self.d = d
    #     data_series = d.get_data_series( direction = 1, fraction = fraction)
    #     self.data_wrapper = d
    #     self.data_series = data_series
    
    # def wavelet_recon( self, level=1):
    #     # move up. Dataseries now only holds wavelet or original data
    #     if self.data_series:
    #         wavelet = Wavelet_Wrapper( self.data_series[1].tolist() , padding = False)
    #         recon = pd.DataFrame( wavelet.reconstruct( level = level ) , columns = ['Recon'] )
    #         self.data_wrapper.data['Recon'] = recon['Recon']
    #         self.data_series = self.data_wrapper.get_data_series( direction = 1, col='Recon', fraction =1)

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
    def sse(self, y, yest):
        """
        Both are array-like of the same shape (n,)
        Returns: Sum-squared error 
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
        obj = self.rms(y, yest) 
        #print("objective= ", obj )
        return obj
    def solve (self, method = 'basinhopping', niter = 10 ):
        """
        Set con = True. to search with the below constraints
        """
        def print_fun(x, f, accepted):
            print("Found minima %.8f, accepted %d" % (f, int(accepted)))
        class MyTakeStep(object):
            def __init__(self, stepsize=0.5): #0.5 for hourly
                self.stepsize = stepsize
            def __call__(self, x):
                s = self.stepsize
                x[0] += np.random.uniform(-5.*s, 5.*s) #a
                x[1] += np.random.uniform(-4.*s, 4.*s) #b
                x[2] += np.random.uniform(-24.*s, 24.*s) #tc
                x[3] += np.random.uniform(-.2*s, .2*s) #m
                x[4] += np.random.uniform(-s, s) #c1
                x[5] += np.random.uniform(-6.*s, 6.*s) #w
                x[6] += np.random.uniform(-s, s) #c2
                #x[1:] += np.random.uniform(-s, s, x[1:].shape)
                return x    

           
        a = (.5, 1000)
        b = (-1000, -0.001)
        # Now tc represet a fractional year instead of a time index
        data_size = self.data_series[0].size
        time_head = self.data_series[0][data_size-1]+0.01
        tc = (time_head, time_head+0.4 ) #look 3 months in advance (.25)
        # old time index limits
        #tc = (self.d.data_size, 1.4*self.d.data_size )
        m = (0.1, 0.9)
        c1 = c2 =  (-1, 1)
        w = (3,25)

        limits = (a, b, tc, m, c1, w, c2)
        bounds = list(limits) 
        # bounds = [(.5, 10000), (-10000, -0.001), (time_head, time_head +.4 )
        # , (0.1, 0.9), (-100, 100), (3,25), (-100, 100) ]
        #initial guess inline with what parameters represent
        x0 = np.array( 
            [ np.average(self.data_series[1]) , 
             - (np.amax(self.data_series[1])-np.amin(self.data_series[1]) ),
             time_head, 0.5, 1, 3, -1])
        print("Initial guess: ", x0)
        xmin = [a[0], b[0], tc[0], m[0], c1[0], w[0], c2[0]]
        xmax = [a[1], b[1], tc[1], m[1], c1[1], w[1], c2[1]]

        class MyBounds(object):
         def __init__(self, xmin, xman ):
             self.xmin = np.array(xmin)
             self.xmax = np.array(xmax)
         def __call__(self, **kwargs):
             x = kwargs["x_new"]
             tmax = bool(np.all(x <= self.xmax))
             tmin = bool(np.all(x >= self.xmin))
             return tmax and tmin

        # then = time.time()
        constraints = [ { 'type': 'eq', 'fun': self.conA },
                        { 'type': 'eq', 'fun': self.conB },
                        { 'type': 'eq', 'fun': self.conC1 },
                        { 'type': 'eq', 'fun': self.conC2 },
                    ]
        options = { 'maxiter': 10000 ,'ftol': 1e-5}
        print("Minimizing..., method: ", method)
        # methods: SLSQP, 
        # Nelder-Mead: Simplex does not work in the below form
        # Bounded: L-BFGS-B, TNC, SLSQP
        minimizer_kwargs = { "method": "SLSQP", "constraints": constraints ,
                             "options": options, "bounds": limits
                           } #
        mytakestep = MyTakeStep()
        # mybounds = MyBounds( xmin, xmax )
        if method == 'basinhopping':
            solution = basinhopping( self.objective, x0, minimizer_kwargs=minimizer_kwargs,
                        T= 0.5 , niter=niter, take_step = mytakestep, callback= print_fun )   
        elif method == 'SLSQP':     
            solution = minimize( self.objective ,x0 ,method='SLSQP',bounds=limits,\
                                constraints=constraints,\
                                options= options )
        else: # Either basinhopping or differential evolution 
            solution = differential_evolution( self.objective, bounds= bounds, 
                        tol=1e-5, maxiter=10000)        
        #print( "Minimization completed in %.2f seconds" % (time.time()-then) )
        # Now crash is a real point in time
        crash = solution.x[2] 
        # old crash calculation based on time index
        #crash = (solution.x[2]- self.d.data_size)/24. 
        x = solution.x
        print( "x0: ", x0 )
        print( "Solution: A= %.3f, B=%.3f, Crash=%.2f, m=%.2f ,c1=%.2f ,w=%.2f ,c2=%.2f. Cost=%.5f" 
                %(x[0],x[1],x[2], x[3],x[4],x[5],x[6], solution.fun))
        print( "ConA: ", str( self.conA(solution.x)) )
        print( "ConB: ", str( self.conB(solution.x)) )
        print( "ConC1: ", str( self.conC1(solution.x)) )
        print( "ConC2: ", str( self.conC2(solution.x)) )
        return solution, crash 
    def plot_solution (self, scale = 1, method= 'basinhopping'):
        """
        Scale represents the number of hours 
        """
        # self.data_series = self.d.get_data_series( direction = 1, col = col, fraction= 1)
        plt.plot(self.data_series[0], self.data_series[1], label='data' )
        solution, crash = self.solve ( method = method )
        model_data = lpplc1( self.data_series[0], solution.x )
        from epsilon import to_year_from_fraction
        crash_time = to_year_from_fraction(crash).strftime( "%d/%m/%Y" )
        label = method + " optimization - crash date: " + crash_time
        # Old time index
        #label = method + " optimization - crash in %.2f days" % (crash*scale) 
        plt.plot( self.data_series[0], model_data, label= label )
        plt.legend(loc='upper center',shadow=True, fontsize='medium')
        plt.xlabel("t" )
        plt.ylabel("Log P(t)")
        #A , #B, Tc, m, c, omega, phi
    def plot_cost (self, col = 'LogClose'):
        # from mpl_toolkits.mplot3d import Axes3D
        # TODO Need maint
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        # self.data_series = self.get_data_series( direction = 1, col = col)
        solution, _ = self.solve( )
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


