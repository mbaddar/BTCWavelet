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
#sz = pdr.get_data_yahoo('159915.SZ', start=datetime.datetime(2014, 1, 1),end=datetime.datetime(2015, 6, 10))
# print( sz)

#Nasdaq
# daily_data = pd.read_csv( "Data/Stock/NASDAQCOM.csv", sep=',', index_col = 0, names=['Close'], header=None )
# daily_data['Close'] = daily_data['Close'].apply( lambda x: np.nan if x=='.' else float(x) )
# # Lppl works on log prices
# daily_data['Close'] = daily_data['Close'].apply( lambda x: np.log(x) )
# # Remove nan values
# daily_data = daily_data [  np.isnan(daily_data['Close'])==False  ]
# date = sz.index

# time = np.linspace(0, len(sz)-1, len(sz))
# close = [sz.Close[i] for i in range(len(sz.Close))]
# print(sz.Close.describe() )
# BTC
daily_data = pd.read_csv( "Data/cmc/daily.csv", sep='\t', parse_dates=['Date'], index_col='Date', 
                            names=[ 'Date', 'Open', 'High', 'Low', 'PirceClose', 'Volume', 'MarketCap'],
                            header=0)

daily_data = daily_data.loc[daily_data.index >= '2017-01-01 00:00:00']
daily_data = daily_data.loc[daily_data.index <= '2017-12-15 00:00:00']

daily_data['Open'] = daily_data['Open'].apply(lambda x: float( x.replace(',','') ) )
daily_data['High'] = daily_data['High'].apply(lambda x: float( x.replace(',','') ) )
daily_data['Low'] = daily_data['Low'].apply(lambda x: float( x.replace(',','') ) )
daily_data['PirceClose'] = daily_data['PirceClose'].apply(lambda x: float( x.replace(',','') ) )
# Lppl works on log prices
daily_data['Close'] = daily_data['PirceClose'].apply( lambda x: np.log(x) )
#filter some dates

#reverse index
# daily_data.index = reversed(daily_data.index)
# daily_data= daily_data.sort_index()

#plt.plot(daily_data['Close'])

date = daily_data.index
time = np.linspace( 0, len(daily_data)-1, len(daily_data)) #just a sequence 
close = [daily_data.Close[-i] for i in range(1,len(daily_data.Close)+1)]
DataSeries = [time, close]

# plt.plot(DataSeries[0], DataSeries[1])
# plt.show()
# # sz = get_price(order_book_ids='159915.XSHE',start_date='20140101',end_date='20150610',fields="ClosingPx")
# date = sz.index
# time = np.linspace(0, len(sz)-1, len(sz))
# close = np.array(np.log(sz))
# DataSeries = [time, close]

def lppl (t,x): #return fitting result using LPPL parameters
    a = x[0]
    b = x[1]
    tc = x[2]
    m = x[3]
    c = x[4]
    w = x[5]
    phi = x[6]
    try:
        return a + ( b*np.power( np.abs(tc-t), m) ) *(1 + (c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
    except BaseException:
        print( "(tc=%d,t=%d)"% (tc,t) )


def func(x):
    """
    The fitness function returns the SSE between lppl and the log price list
    TODO Still doesn't respect boundaries
    """
    lppl_values = lppl( DataSeries[0],x)
    #lppl_values = [lppl(t,x) for t in DataSeries[0]]

    #actuals = DataSeries[1]
    delta = np.subtract( lppl_values, DataSeries[1])
    delta = np.power( delta, 2)
    sse = np.sum( delta) #SSE

    return sse
    #return mean_squared_error(actuals, lppl_values )

class Individual:
    """base class for individuals"""

    def __init__ (self, InitValues, limits):
        self.fit = 0
        self.cof = InitValues
        lim = [ (a[0],a[1]) for a in limits] 
        self.limits = lim

    def fitness(self):
        try:
            cofs, nfeval, rc = fmin_tnc( func,
                                        self.cof, 
                                        fprime=None,
                                        approx_grad=True,
                                        bounds =  self.limits, #Added to respect boundaries
                                        messages=0)
            self.fit = func(cofs)
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
        return Individual( reply, self.limits) 

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
        return DataSeries

    def getExpData(self):
        #Return a list of lppl values For every time point in t 
        ds = [lppl(t,self.cof) for t in DataSeries[0]]
        return ds

    def getTradeDate(self):
        return date

def fitFunc(t, a, b, tc, m, c, w, phi):
    val = a + ( b*np.power( np.abs(tc-t), m) ) *(1 + (c*np.cos((w*np.log( np.abs( tc-t ) ))+phi)))
    return val

class Population:
    """base class for a population"""

    LOOP_MAX = 500

    def __init__ (self, limits, size, eliminate, mate, probmutate, vsize):
        'seeds the population'
        'limits is a tuple holding the lower and upper limits of the cofs'
        'size is the size of the seed population'
        self.populous = []
        self.eliminate = eliminate
        self.size = size
        self.mate = mate
        self.probmutate = probmutate
        self.fitness = []
        for _ in range(size):
            SeedCofs = [ random.uniform(a[0], a[1]) for a in limits ]
            self.populous.append(Individual(SeedCofs , limits))

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
            print('Fitness Evaluating: ' + str(counter) +  " of " + str(len(self.populous)) + "        \r"),
            state = individual.fitness()
            counter += 1
            if ((state == False)):
                false += 1
                self.populous.remove(individual)
        self.SetFitness()
        print("\n fitness out size: " + str(len(self.populous)) + " " + str(false))

    def Eliminate(self):
        a = len(self.populous)
        self.populous.sort(key=lambda ind: ind.fit)
        while (len(self.populous) > self.size * self.eliminate):
            self.populous.pop()
        print("Eliminate: " + str(a- len(self.populous)))

    def Mate(self):
        counter = 0
        if not self.populous:
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
            if (counter > Population.LOOP_MAX):
                print("loop broken: mate")
                while (len(self.populous) <= self.mate * self.size):
                    i = self.populous[random.randint(0, len(self.populous)-1)]
                    j = self.populous[random.randint(0, len(self.populous)-1)]
                    self.populous.append(i.mate(j))
        print("Mate Loop complete: " + str(counter))

    def Mutate(self):
        counter = 0
        for ind in self.populous:
            if (random.uniform(0, 1) < self.probmutate):
                ind.mutate()
                ind.fitness()
                counter +=1
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

    random.seed()