import numpy as np
import matplotlib.pyplot as plt
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

from epsilon import Data_Wrapper, Epsilon_Drawdown
from decomposition import Wavelet_Wrapper  
import time

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
        sol = None
        mat = np.array( 
            [[ self.d.data_size, self.sum_fi(tc, m), self.sum_gi( tc, m, w), self.sum_hi( tc, m, w)],
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

    def solve (self, method = 'basinhopping' ):
        """
        Set con = True. to search with the below constraints
        """
        def print_fun(x, f, accepted):
            print("at minima %.8f accepted %d" % (f, int(accepted)))
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

           
        a = (.5, np.inf)
        b = (-np.inf, -0.001)
        tc = (self.d.data_size, 1.4*self.d.data_size )
        m = (0.1, 0.9)
        c1 = (-np.inf, np.inf)
        w = (3,25)
        c2 = (-np.inf, np.inf)

        limits = (a, b, tc, m, c1, w, c2)
        bounds = [(.5, 10000), (-10000, -0.001), (self.d.data_size, 1.4*self.d.data_size )
        , (0.1, 0.9), (-100, 100), (3,25), (-100, 100) ]
        #initial guess inline with what parameters represent
        x0 = np.array( 
            [ np.average(self.data_series[1]) , 
             - (np.amax(self.data_series[1])-np.amin(self.data_series[1]) ),
             self.d.data_size, 0.5, 1, 7, -1])
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
        options = { 'maxiter': 10000 ,'ftol': 1e-8}
        print("Minimizing..., method: ", method)
        # methods: SLSQP, 
        # Nelder-Mead: Simplex does not work in the below form
        # Bounded: L-BFGS-B, TNC, SLSQP
        # solution = minimize( self.objective ,x0 ,method='SLSQP',bounds=limits,\
        #                     constraints=[conA, conB, conC1, conC2],\
        #                     options= options )
        minimizer_kwargs = { "method": "SLSQP", "constraints": constraints ,
                             "options": options, "bounds": limits
                           } #
        mytakestep = MyTakeStep()
        # mybounds = MyBounds( xmin, xmax )
        if method == 'basinhopping':
            solution = basinhopping( self.objective, x0, minimizer_kwargs=minimizer_kwargs,
                        T= 1 ,niter=20, niter_success = 3, take_step = mytakestep, callback= print_fun )        
        else: # Either basinhopping or differential evolution 
            solution = differential_evolution( self.objective, bounds= bounds, 
                        tol=1e-4, maxiter=10000)        

        #print( "Minimization completed in %.2f seconds" % (time.time()-then) )
        crash = (solution.x[2]- self.d.data_size)/24. 
        x = solution.x
        print( "x0: ", x0 )
        print( "Solution: A= %.3f, B=%.3f, Crash=%.2f, m=%.2f ,c1=%.2f ,w=%.2f ,c2=%.2f. Cost=%.5f" 
                %(x[0],x[1],x[2], x[3],x[4],x[5],x[6], solution.fun))
        print( "ConA: ", str( self.conA(solution.x)) )
        print( "ConB: ", str( self.conB(solution.x)) )
        print( "ConC1: ", str( self.conC1(solution.x)) )
        print( "ConC2: ", str( self.conC2(solution.x)) )
        return solution, crash 

    def plot_solution (self, col = 'LogClose', scale = 1, method= 'basinhopping'):
        """
        Scale represents the number of hours 
        """
        plt.plot(self.d.data[ col ])
        self.data_series = self.d.get_data_series( direction = 1, col = col)
        solution, crash = self.solve ( method = method )
        model_data = lpplc1( self.data_series[0], solution.x )
        label = method + " optimization - crash in %.2f days" % (crash*scale) 
        plt.plot( self.data_series[0], model_data, label= label )
        plt.legend(loc='upper center',shadow=True, fontsize='medium')
        plt.xlabel("t" )
        plt.ylabel("Log P(t)")
        # solution = self.minimize ( data_series[0], data_series[1])
        #A , #B, Tc, m, c, omega, phi

    def plot_cost (self, col = 'LogClose'):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        self.data_series = self.d.get_data_series( direction = 1, col = col)
        solution, _ = self.solve( )
        x = solution.x
        print(solution)
        tc = x[2]
        tc_range = np.linspace(self.d.data_size, 1.2*tc, 500)        
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
                #lppl = pl.model_lppl( data_series ) # returns 3 fits
                print("Step: %d, dt: %d" % (step, dt))
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
    random.seed(1)
    # l = Grid_Fit("2017-1-1 00:00:00", "2017-12-10 00:00:00" )
    # plt.plot(l.data_series[0], l.data_series[1])
    l = Nonlinear_Fit("2017-10-1 00:00:00", "2017-12-10 00:00:00" )
    # Parameters 
    # c1 = c *cos(phi), c2 = c * sin (phi)
    # 
    # a, b, tc, m, c, w, phi
    #x = [36960, -35902, 91.81, 0.16, -18839, 4.8, -.3]
    #l.plot_solution( con = True, col = 'LogClose',scale = 1) #Recon data size issue
    #l.plot_solution(  col = 'LogClose',scale = 1) #Recon data size issue
    l.plot_solution(col = 'LogClose', method= 'basinhopping' )
    # plt.close('all')
    # l = Grid_Fit("2017-10-15 00:00:00", "2017-12-10 00:00:00" )
    # l.plot_solution( con = True, col = 'LogClose',scale = 1) #Recon data size issue
    # plt.show()

    #l = Grid_Fit("2017-1-15 00:00:00", "2017-12-10 00:00:00" )
    #l = Grid_Fit("2017-06-15 00:00:00", "2017-12-10 00:00:00" )
    #l = Grid_Fit("2013-11-10 00:00:00", "2013-11-18 00:00:00" )
    # l = Grid_Fit("1927-05-1 00:00:00", "1930-12-31 00:00:00" , data_source= 'DIJA')
    # plt.plot(l.data_series[0], l.data_series[1])
    # l = Grid_Fit("1927-05-1 00:00:00", "1929-10-24 00:00:00" , data_source= 'DIJA')
    # l.plot_solution( con = True, col = 'LogClose',scale = 24) #Recon data size issue
    #l = Grid_Fit("2007-03-12 00:00:00", "2007-10-10 00:00:00" , data_source= 'SSE')
    # wavelet = Wavelet_Wrapper( l.d.data['LogClose'].tolist() , padding = False)
    # recon = pd.DataFrame( wavelet.reconstruct( level = 1 ) , columns = ['Recon'] )
    # l.d.data['Recon'] = recon['Recon']
    # l.data_series = l.d.get_data_series( direction = 1)
    # print(" Dataseries shape =", l.data_series[0].shape) 
    #pl = Pipeline( l.d )
    #lpplfit = pl.model_lppl( l.data_series ) # returns 3 fits
    plt.show()

