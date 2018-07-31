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
from epsilon import Data_Wrapper #Data crawler class 
from sklearn import metrics
from sklearn.cluster import KMeans

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

# plt.plot(DataSeries[0], DataSeries[1])
# plt.show()
# # sz = get_price(order_book_ids='159915.XSHE',start_date='20140101',end_date='20150610',fields="ClosingPx")
# date = sz.index
# time = np.linspace(0, len(sz)-1, len(sz))
# close = np.array(np.log(sz))
# DataSeries = [time, close]
# d = Data_Wrapper()
# DataSeries = d.get_lppl_data()

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
        lppl_values = lppl( self.data_series[0],x)
        #lppl_values = [lppl(t,x) for t in DataSeries[0]]
        #actuals = DataSeries[1]
        delta = np.subtract( lppl_values, self.data_series[1])
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

    def __init__ (self, limits, size, eliminate, mate, probmutate, vsize, data_series ):
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

def run( search = True):
    
    d = Data_Wrapper()
    t1 = "2017-03-24 00:00:00"
    t2 = "2017-05-25 00:00:00"
    dataSeries = d.get_lppl_data(date_from= t1, date_to = t2)
    data_size = d.data_size
    cluster = []
    for step in range(0, data_size-10 ): #skip 5 time points and recalculate
        # respective minimum and maximum values ​​of the seven parameters fitting process
        dt = data_size - step
        print("Step: %d, dt: %d" % (step, dt))
        limits = (
            [1, 200],     # A :
            [-100, -0.1],     # B :
            [dt, 2*dt],     # Critical Time : between the end of the TS part and 1 length away 
            #[350, 400],    # Critical Time :
            [0.01, .999],       # m :
            [-1, 1],        # c :
            [4, 25],       # omega
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
        cluster.append( (wrappers[0].tc-dt, dt ) )
    plt.scatter(*zip(*cluster) )
    plt.show(block=True)
    print(cluster )

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
        x = self.model

        a = x[0]
        b = x[1]
        tc = x[2]
        m = x[3]
        c = x[4]
        w = x[5]
        phi = x[6]
        try:
            yield a + ( b*np.power( np.abs(tc-ts), m) ) *( 1 + ( c*np.cos((w*np.log( np.abs( tc-ts)))+ phi)))
        except BaseException:
            print( "(tc=%d,t=%d)"% (tc,ts) )

    def plot(self, ts):
        plt.plot(ts, list(self.generator(ts))[0]  )

if __name__ == "__main__":

    # data = pd.read_csv("Data/lppl_fit.csv", header=0)
    # data = data.set_index('Date')
    # data['Fit1'] = data['Fit1'].apply(lambda x: np.exp(x) )
    # data['Fit2'] = data['Fit2'].apply(lambda x: np.exp(x) )
    # data['Fit3'] = data['Fit3'].apply(lambda x: np.exp(x) )
    # data.Fit2.plot(figsize=(14,8))
    # plt.show(block=True)
    random.seed()
    run()


