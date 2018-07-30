'''
Help from https://www.ricequant.com/community/topic/427/
Thanks ricequant!
Updated implementation of the LPPL model running on the s&p 500
'''
import lppl as lppl
from matplotlib import pyplot as plt
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')

def run( search = True):
    # respective minimum and maximum values ​​of the seven parameters fitting process
    limits = (
        [1, 20],     # A :
        [-10, -0.1],     # B :
        [340, 410],     # Critical Time :
        #[350, 400],     # Critical Time :
        [.01, .99],       # m :
        [-1, 1],        # c :
        [6, 25],       # omega
        #[12, 25],       # omega
        [0, 2 * np.pi]  # phi : up to 8.83
    )
    #x = lppl.Population(limits, 20, 0.3, 1.5, .05, 4)
    x = lppl.Population( limits, 50, 0.3, 1.5, .05, 4)
    #for i in range(2):
    for _ in range(2):
        x.Fitness()
        x.Eliminate()
        x.Mate()
        x.Mutate()

    x.Fitness()
    values = x.BestSolutions(3)
    if values:
        for x in values:
            print(x.PrintIndividual())

        #TODO Buggy
        data = pd.DataFrame({'Date':values[0].getDataSeries()[0],
                            'Index':np.exp( values[0].getDataSeries()[1]), #Display the price instead of log p
                            'Fit1' :np.exp( values[0].getExpData() ),
                            'Fit2' :np.exp( values[1].getExpData() ),
                            'Fit3' :np.exp( values[2].getExpData() ) })
        data = data.set_index('Date')
        data.to_csv("Data/lppl_fit.csv")
        data.plot(figsize=(14,8))
        plt.show(block=True)
    else:
        print("No values")

if __name__ == "__main__":

    # data = pd.read_csv("Data/lppl_fit.csv", header=0)
    # data = data.set_index('Date')
    # data['Fit1'] = data['Fit1'].apply(lambda x: np.exp(x) )
    # data['Fit2'] = data['Fit2'].apply(lambda x: np.exp(x) )
    # data['Fit3'] = data['Fit3'].apply(lambda x: np.exp(x) )
    # data.Fit2.plot(figsize=(14,8))
    # plt.show(block=True)
    run()