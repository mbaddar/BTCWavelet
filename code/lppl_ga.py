import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc, least_squares
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



class Individual:
    """base class for individuals"""

    def __init__ (self, InitValues, limits, data_series):
        self.fit = 0
        self.cof = InitValues
        lim = [ (a[0],a[1]) for a in limits] 
        self.limits = lim
        self.data_series = data_series

    def func(self, x):
        """
        The fitness function returns the SSE between lppl and the log price list
        """
        lppl_values = lppl( self.data_series[0], x)
        #lppl_values = [lppl(t,x) for t in DataSeries[0]]
        #actuals = DataSeries[1]
        delta = np.subtract( lppl_values, self.data_series[1] )
        # print("Lppl values: ", self.data_series[1])
        # print("Data: ", lppl_values)
        delta = np.power( delta, 2)
        sse = np.sum( delta) #SSE
        return sse
        #return mean_squared_error(actuals, lppl_values )

    def fitness(self):
        try:
            cofs, nfeval, rc = fmin_tnc( self.func,
                                        self.cof, 
                                        fprime=None,
                                        approx_grad=True,
                                        bounds =  self.limits, #Added to respect boundaries
                                        messages=0)
            self.fit = self.func(cofs)
            self.cof = cofs
        except:
            #does not converge
            print("Does not converge")
            return False

    def mate(self, partner):
        reply = []
        for i in range(0, len(self.cof)):
            if (random.randint(0,1) == 1):
                reply.append(self.cof[i])
            else:
                reply.append(partner.cof[i])
        #Limit do not change
        return Individual( reply, self.limits, self.data_series) 

    def mutate(self):
        for i in range(0, len(self.cof)-1):
            if (random.randint(0,len(self.cof)) <= 2):
                self.cof[i] += random.choice([-1,1]) * .05 * i

    def PrintIndividual(self):
        #t, a, b, tc, m, c, w, phi
        cofs = "A: " + str(round(self.cof[0], 3))
        cofs += " B: " + str(round(self.cof[1],3))
        cofs += " Critical Time: " + str(round(self.cof[2], 3))
        cofs += " m: " + str(round(self.cof[3], 3))
        cofs += " c: " + str(round(self.cof[4], 3))
        cofs += " omega: " + str(round(self.cof[5], 3))
        cofs += " phi: " + str(round(self.cof[6], 3))
        return " fitness: " + str(self.fit) +"\n" + cofs
        
    def getDataSeries(self):
        return self.data_series

    def getExpData(self):
        #Return a list of lppl values For every time point in t 
        ds = [lppl(t,self.cof) for t in self.data_series[0]]
        return ds

    # def getTradeDate(self):
    #     return date

# def fitFunc(t, a, b, tc, m, c, w, phi):
#     val = a + ( b*np.power( np.abs(tc-t), m) ) *(1 + (c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
#     return val

class Population:
    """base class for a population"""

    LOOP_MAX = 500

    def __init__ (self, limits, size, eliminate, mate, probmutate, vsize, data_series, verbose = False ):
        'seeds the population'
        'limits is a tuple holding the lower and upper limits of the cofs'
        'size is the size of the seed population'
        self.populous = []
        self.eliminate = eliminate
        self.size = size
        self.mate = mate
        self.probmutate = probmutate
        self.fitness = []
        self.data_series = data_series
        self.verbose = verbose 
        for _ in range(size):
            SeedCofs = [ random.uniform(a[0], a[1]) for a in limits ]
            self.populous.append(Individual(SeedCofs , limits, data_series ))

    def PopulationPrint(self):
        for x in self.populous:
            print(x.cof)

    def SetFitness(self):
        self.fitness = [x.fit for x in self.populous]
 
    def FitnessStats(self):
        #returns an array with high, low, mean
        return [np.amax(self.fitness), np.amin(self.fitness), np.mean(self.fitness)]

    def Fitness(self):
        counter = 0
        false = 0
        for individual in list(self.populous):
            #print('Fitness Evaluating: ' + str(counter) +  " of " + str(len(self.populous)) + "        \r"),
            state = individual.fitness()
            counter += 1
            if ((state == False)):
                false += 1
                self.populous.remove(individual)
        self.SetFitness()
        if self.verbose:
            print("\n fitness out size: " + str(len(self.populous)) + " " + str(false))

    def Eliminate(self):
        a = len(self.populous)
        self.populous.sort(key=lambda ind: ind.fit)
        while (len(self.populous) > self.size * self.eliminate):
            self.populous.pop()
        if self.verbose:
            print("Eliminate: " + str(a- len(self.populous)))

    def Mate(self):
        counter = 0
        if not self.populous and self.verbose:
            print("Empty populous")
        while ( self.populous and len(self.populous) <= self.mate * self.size):
            counter += 1
            #TODO What if populous is empty?
            i = self.populous[random.randint(0, len(self.populous)-1)]
            j = self.populous[random.randint(0, len(self.populous)-1)]
            diff = abs(i.fit-j.fit)
            if (diff < random.uniform( 
                            np.amin(self.fitness), 
                            np.amax(self.fitness) - np.amin(self.fitness)) ):
                self.populous.append(i.mate(j))
            if counter > Population.LOOP_MAX and self.verbose:
                print("loop broken: mate")
                while (len(self.populous) <= self.mate * self.size):
                    i = self.populous[random.randint(0, len(self.populous)-1)]
                    j = self.populous[random.randint(0, len(self.populous)-1)]
                    self.populous.append(i.mate(j))
        if self.verbose:
            print("Mate Loop complete: " + str(counter))

    def Mutate(self):
        counter = 0
        for ind in self.populous:
            if (random.uniform(0, 1) < self.probmutate):
                ind.mutate()
                ind.fitness()
                counter +=1
        if self.verbose:
            print("Mutate: " + str(counter))
        self.SetFitness()

    def BestSolutions(self, num):
        reply = []
        if not self.populous:
            print("No best solution found. Empty populous")
            return None 
        self.populous.sort(key=lambda ind: ind.fit)
        for i in range(num):
            reply.append(self.populous[i])
        return reply


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

    def model_lppl ( self, dataSeries ):
        data_size = dataSeries[0].size
        # dt = data_size - step
        limits = (
            [1, 1000],     # A :
            [-100, -0.1],     # B :
            [data_size, 2*data_size],     # Critical Time : between the end of the TS part and 1 length away 
            [0.01, .999],       # m :
            [-1, 1],        # c :
            [6, 13],       # omega
            #[12, 25],       # omega
            [0, 2 * np.pi]  # phi : up to 8.83
        )

        x = Population( limits, 20, 0.3, 1.5, .05, 4, \
        [ dataSeries[0], dataSeries[1] ] )
        for _ in range(2):
            x.Fitness()
            x.Eliminate()
            x.Mate()
            x.Mutate()

        x.Fitness()
        values = x.BestSolutions(3)
        wrappers = []
        if values:
            for x in values:
                print(x.PrintIndividual())
                model = [ x.cof[i] for i in range(7)]
                wrappers.append( Lppl_Wrapper( model, x.fit ) )
        else:
            print("No values")
        return wrappers #The first fit model
        #return wrappers[0] #The first fit model

def run( Test = False ):
    d = Data_Wrapper( hourly = True)
    d.filter_by_date( "2017-12-1 00:00:00", "2018-1-10 00:00:00")
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
    plt.show(block=True)
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

        # a = x[0]
        # b = x[1]
        # tc = x[2]
        # m = x[3]
        # c = x[4]
        # w = x[5]
        # phi = x[6]
        # try:
        #     yield a + ( b*np.power( np.abs(tc-ts), m) ) *( 1 + ( c*np.cos((w*np.log( np.abs( tc-ts)))+ phi)))
        # except BaseException:
        #     print( "(tc=%d,t=%d)"% (tc,ts) )
        values = lppl(ts, x)
        return values 

    def plot(self, ts, offset=0):
        # Offset is to plot it on a full TS 
        #plt.plot(ts + offset , list( self.generator(ts)[0] )  )
        ts = ts + offset
        print("plotting ts from point: ", ts[0])
        plt.plot(ts , list( self.generator(ts) )  )

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
    except BaseException:
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
        fun = a + pwr * (b  + c1 * np.cos( ( w*np.log( tc-t ))) + c2 * np.sin( ( w*np.log( tc-t ))) )
        return fun
    except BaseException:
        print( "(tc=%d,t=%d)"% (tc,t) )

def lppl_res (x, t, y): #return fitting result using LPPL parameters
    """
    t and y must have the same dimensions
    """
    try:
        return lppl(t, x) - y
    except BaseException:
        print( "(tc=%d,t=%d)"% (x[2],t) )

search_ranges = {
    'm': np.linspace(0.1,.9, 100),
    'w': np.linspace( 6,13, 100)
}

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
    
    def __init__(self):
        d = Data_Wrapper( hourly = True)
        d.filter_by_date( "2017-11-01 00:00:00", "2017-12-17 00:00:00")
        self.d = d
        self.data_series = d.get_data_series()


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
        return np.sum( np.multiply( np.power( tc-t, m) , np.cos( ( w*np.log( tc-t ))) ) )

    def sum_fi_square(self, tc, m):
        t = self.data_series[0]
        return np.sum( np.power( tc-t, 2*m) )  
    def sum_gi_square (self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.power( np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ), 2 ) )
    def sum_hi_square ( self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.power( np.multiply( np.power( tc-t, m) , np.cos( ( w*np.log( tc-t ))) ),2 ) )

    def sum_yi(self):
        t = self.data_series[0]
        return np.sum( self.data_series[1] )
    def sum_yi_fi(self, tc, m):
        t = self.data_series[0]
        return np.sum( np.multiply( self.data_series[1], np.power( tc-self.data_series[0], m) ) )  
    def sum_yi_gi(self, tc, m, w):
        t = self.data_series[0]
        return np.sum( np.multiply( self.data_series[1], np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ) ) )  
    def sum_yi_hi(self, tc, m, w):
        t = self.data_series[0]
        return np.sum( np.multiply( self.data_series[1], np.multiply( np.power( tc-t, m) , np.cos( ( w*np.log( tc-t ))) ) ) )  

    def sum_fi_gi(self, tc, m, w):
        t = self.data_series[0]
        return np.sum( np.multiply( np.power( tc-self.data_series[0], m), np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ) ) )  
    def sum_gi_hi (self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply( np.multiply( np.power( tc-t, m), np.cos( ( w*np.log( tc-t ))) ) ), np.multiply( np.power( tc-t, m) , np.cos( ( w*np.log( tc-t ))) ) )
    def sum_fi_hi( self, tc, m, w ):
        t = self.data_series[0]
        return np.sum( np.multiply ( np.power( tc-self.data_series[0], 2*m), np.multiply( np.power( tc-t, m) , np.cos( ( w*np.log( tc-t ))) ) ) )
    

    def fit_linear_params (self, tc, m, w):
        """
        Fits A, B, C1, C2 according to an analytic solution given tc, m, w
        Returns a vector of A, B, C1, C2
        """
        from numpy.linalg import inv
        b = np.array( [self.sum_yi(), self.sum_yi_fi(tc, m), self.sum_yi_gi (tc, m, w), self.sum_yi_gi (tc, m, w) ] ).T
        # Summetric 
        a = inv( np.array( [ self.d.data_size, self.sum_fi(tc, m), self.sum_gi( tc, m, w), self.sum_hi( tc, m, w)],
                           [self.sum_fi( tc, m), self.sum_fi_square(tc, m), self.sum_fi_gi(tc, m, w), self.sum_fi_hi(tc, m, w)],
                           [self.sum_gi(tc, m, w), self.sum_fi_gi(tc, m, w), self.sum_gi_square( tc, m, w), self.sum_gi_hi( tc, m, w)],
                           [self.sum_hi( tc, m, w) , self.sum_fi_hi( tc, m, w) , self.sum_gi_hi ( tc, m, w) , self.sum_hi_square (tc, m, w)]
                    ) )
        return np.dot( a, b)

    def sse(self, y, yest):
        """
        Both are array-like of the same shape (n,)
        """
        return  np.sum( np.power( y-yest, 2) )

    def f(self, x):
        t = self.data_series[0]
        y = self.data_series[1]
        yest = lpplc1(t, x)
        return self.sse(y, yest)
        
    def f1(self, tc, m, w):
        """
        min of f. Searching over m and w
        """
        for (m,w) in [ (m,w) for m in search_ranges['m'] for w in search_ranges['w'] ]:
            pass 
    def f2(self):
        pass 

    def minimize( self, t_train, y_train, method = 'lm'):
        limits = (  
            #A , #B, Tc, m, c, omega, phi
            [1, -100, self.d.data_size, 0.01, -1, 6, 0],
            [1000, -0.1, 2*self.d.data_size, .999, 1, 13, 2 * np.pi]
        )
        x0 = np.array([1.0, -1, self.d.data_size, 0.5, .5, 6, 0])
        res_lsq = least_squares(lppl_res, x0, f_scale=0.1, method= method, args=(t_train, y_train) ) if method=='lm'\
        else least_squares(lppl_res, x0, f_scale=0.1, method= method, bounds= limits, args=(t_train, y_train) )
        return res_lsq
    

    def plot_solution (self):
        data_series = self.d.get_data_series( direction = 1)
        plt.plot(self.d.data['LogClose'])
        print("data size = %d" % self.d.data_size )
        print("dataseries size = %d" % len(data_series[0]) )
        solution = self.minimize ( data_series[0], data_series[1])
        #A , #B, Tc, m, c, omega, phi
        print( "solution: ", np.around(solution.x,3) )
        model_data = lppl(data_series[0], solution.x )
        plt.plot( data_series[0], model_data )

        solution = self.minimize ( data_series[0], data_series[1], method = 'dogbox')
        model_data = lppl(data_series[0], solution.x )
        plt.plot( data_series[0], model_data )
        print( "solution dogbox: ", np.around(solution.x,3) )

        plt.xlabel("t")
        plt.ylabel("Log P(t)")
        plt.show()

if __name__ == "__main__":

    # plt.show(block=True)
    random.seed()
    #run()
    l = Grid_Fit()
    l.plot_solution()

# Junk Code 
def run1( search = True):
    # Get hourly data
    
    d = Data_Wrapper( hourly = True)
    t1 = "2017-03-24 00:00:00"
    t2 = "2017-05-25 00:00:00"
    dataSeries = d.get_lppl_data(date_from= t1, date_to = t2)
    data_size = d.data_size
    points = []
    for step in range(0, data_size-10 ): #skip 5 time points and recalculate
        # respective minimum and maximum values ​​of the seven parameters fitting process
        dt = data_size - step
        print("Step: %d, dt: %d" % (step, dt))
        limits = (
            [1, 200],     # A :
            [-1000, -0.1],     # B :
            [dt, dt+50],     # Critical Time : between the end of the TS part and 1 length away 
            #[dt, 2*dt],     # Critical Time : between the end of the TS part and 1 length away 
            #[350, 400],    # Critical Time :
            [0.01, .999],       # m :
            [-1, 1],        # c :
            [25, 50],       # omega
            #[12, 25],       # omega
            [0, 2 * np.pi]  # phi : up to 8.83
        )
        #x = lppl.Population(limits, 20, 0.3, 1.5, .05, 4)
        x = Population( limits, 20, 0.3, 1.5, .05, 4, \
        [ dataSeries[0][step:], dataSeries[1][step:] ] )
        #for i in range(2):
        for _ in range(2):
            x.Fitness()
            x.Eliminate()
            x.Mate()
            x.Mutate()

        x.Fitness()
        values = x.BestSolutions(3)
        wrappers = []
        if values:
            for x in values:
                print(x.PrintIndividual())
                model = [ x.cof[i] for i in range(7)]
                wrappers.append( Lppl_Wrapper( model, x.fit ) )

            #TODO Buggy
            # data = pd.DataFrame({'Date':values[0].getDataSeries()[0],
            #                     'Index': values[0].getDataSeries()[1], #Display the price instead of log p
            #                     'Fit1' : values[0].getExpData() ,
            #                     'Fit2' : values[1].getExpData() ,
            #                     'Fit3' : values[2].getExpData()  })
            # data = data.set_index('Date')
            #data.to_csv("Data/lppl_fit.csv")
            # if step ==0: # plot the TS 
            #     plt.plot( values[0].getDataSeries()[0],  values[0].getDataSeries()[1] )
            # Plot the lppl fits
            # wrappers[0].plot( values[0].getDataSeries()[0] )
            # wrappers[1].plot( values[0].getDataSeries()[0] )
            # wrappers[2].plot( values[0].getDataSeries()[0] )
        else:
            print("No values")
        points.append( (wrappers[0].tc-dt, dt ) )
    plt.scatter( *zip( *points) )
    plt.gca().set_xlabel('tc-t2')
    plt.gca().set_ylabel('dt = t2-t1')
    clusters, labels = cluster(points, 4)
    plt.scatter( *zip( *clusters) )
    plt.show(block=True)
    print( metrics.silhouette_score(points, labels) )
    print ( "Clusters: ", clusters )
