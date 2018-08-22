#lppl_ga.py
# Junk Code 

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
    l.plot_solution2( method= 'basinhopping', niter=20)
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

