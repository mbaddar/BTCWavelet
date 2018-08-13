#lppl_ga.py
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

#epsilon

################# Junk Code ############################
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

#Data_Wrapper
  # def get_data_series( self, index =0, to = -1, direction = 1, col = 'LogClose', fraction = 0):
    #     """
    #     Direction: +/- 1
    #     fraction is a flag. if set to 0 (default): dataSeries[0] is time points indexed from 0
    #     if set to 1: return fractional year. Example: 2018.604060 for 9/8/2018
    #     """
    #     if direction not in [-1,1]:
    #         direction = 1 #Should raise some error 
    #     # Remove na first 
    #     data = self.data[ col ][self.data[ col ].notna()]
    #     data = np.array( data[index: to] if to>-1 else data[index:] ) 
    #     data_size = data.size 
    #     #time = np.linspace( 0, data_size-1, data_size) #just a sequence 
    #     time = None
    #     if fraction: #apply a filter then convert to numpy array
    #         time = self.data['Date'].apply( lambda epoch: toYearFraction( epoch) ).values[:data_size]
    #     else:
    #         time = np.arange( data_size )
    #     values = (data if direction==1 else np.flip(data, axis=0) )
    #     dataSeries = [time, values]
    #     # Reset data size
    #     self.data_size = data_size        
    #     return dataSeries