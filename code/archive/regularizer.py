class Lagrange_regularizer:
    """
    Copyright: G.Demos @ ETH-Zurich - Jan.2017
    Swiss Finance Institute
    SSRN-id3007070
    Guilherme DEMOS
    Didier SORNETTE
    """
    __B_max = 0 #Power law amplitude
    __m_range = np.arange(0,1,20).tolist()  #PL exponent [0,1]
    __omeaga_range = np.arange(4, 25, 2).tolist() #LP frequency [4,25]
    __D_min = 0.5 #Damping [0.5, inf]
    __O_min = 2.5 #No of oscillations [2.5, inf]

    def simulateOLS( self):
        """ Generate synthetic OLS as presented in the paper """
        nobs = 200
        X = np.arange(0,nobs,1)
        e = np.random.normal(0, 10, nobs)
        beta = 0.5
        Y = [beta*X[i] + e[i] for i in range(len(X))]
        Y = np.array(Y)
        X = np.array(X)
        Y[:100] = Y[:100] + 4*e[:100]
        Y[100:200] = Y[100:200]*8
        return X, Y

    def fitDataViaOlsGetBetaAndLine( self, X, Y):
        """ Fit synthetic OLS """
        beta_hat = np.dot(X.T,X)**-1. * np.dot(X.T,Y) # get beta
        Y = [beta_hat*X[i] for i in range(len(X))]
        # generate fit
        return Y

    def getSSE( self, Y, Yhat, p=1, normed=False):
        """
        Obtain SSE (chi^2)
        p -> No. of parameters
        Y -> Data
        Yhat -> Model
        """
        error = (Y-Yhat)**2. #SSE
        obj = np.sum(error) 
        if normed == False:
            obj = np.sum(error) 
        else:
            obj = 1/np.float(len(Y) - p) * np.sum(error)
        return obj

    def getSSE_and_SSEN_as_a_func_of_dt( self, normed=False, plot=False):
        """ Obtain SSE and SSE/N for a given shrinking fitting window w """
        # Simulate Initial Data
        X, Y = self.simulateOLS()
        # Get a piece of it: Shrinking Window
        _sse = []
        _ssen = []
        for i in range(len(X)-10): # loop t1 until: t1 = (t2 - 10):
            xBatch = X[i:-1]
            yBatch = Y[i:-1]
            YhatBatch = self.fitDataViaOlsGetBetaAndLine( xBatch, yBatch)
            sse = self.getSSE(yBatch, YhatBatch, normed=False)
            sseN = self.getSSE(yBatch, YhatBatch, normed=True)
            _sse.append(sse)
            _ssen.append(sseN)
        if plot == False:
            pass
        else:
            f, ax = plt.subplots( 1,1,figsize=(6,3) )
            ax.plot( _sse, color= 'k')
            a = ax.twinx()
            a.plot( _ssen, color='b')
            plt.tight_layout()
        if normed == False: 
            return _sse, _ssen, X, Y # returns results + data
        else:
            return _sse/max(_sse), _ssen/max(_ssen), X, Y # returns results + data
    ########################    

    def LagrangeMethod( self, sse):
        """ Obtain the Lagrange regulariser for a given SSE/N """
        # Fit the decreasing trend of the cost function
        slope = self.calculate_slope_of_normed_cost(sse)
        return slope[0]        

    def calculate_slope_of_normed_cost( self, sse):
        #Create linear regression object using statsmodels package
        regr = linear_model.LinearRegression( fit_intercept=False)
        # create x range for the sse_ds
        x_sse = np.arange(len(sse))
        x_sse = x_sse.reshape(len(sse),1)
        # Train the model using the training sets
        res = regr.fit(x_sse, sse)
        return res.coef_
    ########################

    def obtainLagrangeRegularizedNormedCost( self, X, Y, slope):
        """ Obtain the Lagrange regulariser for a given SSE/N Pt. III"""
        Yhat = self.fitDataViaOlsGetBetaAndLine(X,Y) # Get Model fit
        ssrn_reg = self.getSSE(Y, Yhat, normed=True) # Classical SSE
        ssrn_lgrn = ssrn_reg - slope*len(Y) # SSE lagrange
        return ssrn_lgrn
    
    def GetSSEREGvectorForLagrangeMethod( self, X, Y, slope):
        """
        X and Y used for calculating the original SSEN
        slope is the beta of fitting OLS to the SSEN
        """
        # Estimate the cost function pondered by lambda using a Shrinking Window.
        _ssenReg = []
        for i in range(len(X)-10):
            xBatch = X[i:-1]
            yBatch = Y[i:-1]
            regLag = self.obtainLagrangeRegularizedNormedCost(xBatch, yBatch, slope)
            _ssenReg.append(regLag)
        return _ssenReg
