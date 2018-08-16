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

def lppl_res (x, t, y): #return fitting result using LPPL parameters
    """
    t and y must have the same dimensions
    """
    try:
        return lppl(t, x) - y
    except BaseException:
        print( "(tc=%d,t=%d)"% (x[2],t) )


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

def model_lppl ( dataSeries ):
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
    # def getTradeDate(self):
    #     return date

# def fitFunc(t, a, b, tc, m, c, w, phi):
#     val = a + ( b*np.power( np.abs(tc-t), m) ) *(1 + (c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
#     return val
