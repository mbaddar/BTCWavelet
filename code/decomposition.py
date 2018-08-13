import numpy as np 
from pandas import DataFrame as DF 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec, idwt, waverec, downcoef


from crawlers.crawler import Crawler
import math
import json
import copy 

class Wavelet_Wrapper:

    @property 
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    @property 
    def coeffs(self):
        return self.__coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        self.__coeffs = coeffs

    @property 
    def wavelet_name(self):
        return self.__wavelet_name

    @wavelet_name.setter
    def wavelet_name(self, wavelet_name):
        self.__wavelet_name = wavelet_name

    @property 
    def coeffs_size(self):
        return self.__coeffs_size
    @coeffs_size.setter
    def coeffs_size(self, coeffs_size):
        self.__coeffs_size = coeffs_size

    def __init__ (self, data, wavelet_name ='db8', padding = True):
        data_size = len(data)
        # Either 0 pad or trim
        if not self.power2( data_size):
            next_power2 = self.next_power_of_2( data_size)
            if padding:
                pad = next_power2 - data_size
                for _ in range(pad): 
                    data.append( 0 )
            else:
                trimmed_size = int(next_power2/2)
                #Take the last power2 data points
                data = data[ -trimmed_size: ]
        
        self.data = data
        self.wavelet_name = wavelet_name
        self.coeffs = self.wavelet_decomposition( wavelet_name)
        self.coeffs_size = len( self.coeffs )


    def power2(self, num):
        return ((num & (num - 1)) == 0) and num != 0

    def next_power_of_2(self, x):  
        return 1 if x == 0 else 2**(x - 1).bit_length()

    def zeros(self, n):
        if isinstance(n, int):
            return [0]*n
        else:
            raise TypeError("Number must be integer")

    def wavelet_decomposition(self, level = None):
        """
        2-D Array of coefficients
        [cA_n, cD_n, cD_n-1, …, cD2, cD1]
        cD1 is the highest (coarsest) level
        cD_n is the most detailed. Applied first 
        Each coeff level is half in length to the next one starting from n 
        """
        #TODO: raise an error of name not in family of wavelets
        coeffs = wavedec( self.data , self.wavelet_name , level=None )
        self.__coeffs = coeffs
        return coeffs
    def get_decomposition_level (self, level = 1):
        """
        Detail level
        """
        return downcoef('d', self.data, self.wavelet_name, mode='smooth', level = level )
    def plot_coefficients( self, wavelet_name, coeffs ):

        #Construct a DataFrame from the coeffs
        coeff_dict = dict( ("Coefficient (%d)"% d, coeffs[d].tolist() ) 
                        for d in range( len( coeffs)) )
        # Save plots
        # for d in range( len( coeffs)): 
        #     coeffs[d] = coeffs[d].tolist()  

        for d in range( len(coeffs) ):
            plt.close('all')
            plt.plot(coeffs[d])
            plt.title( "Coeff[%d]"%d)
            plt.savefig("Wavelets-%s-%2d.png"% (wavelet_name,d ) )
        # Save coeffs
        with open('coeffs.json', 'w+') as fp:
            json.dump( coeff_dict, fp, sort_keys=True, indent=4)

    def reconstruct (self, level =1 ):
        """
        Reconstruct the original TS by removing up to n levels of details and 
        applying the inverse DWT. 
        Level 1 is the highest (coarsest) and level n is the lowest (most detailed)
        0 is the summary level and cannot be removed. 
        """
        recon = None
        coeffs = copy.deepcopy( self.coeffs)
        print("coeff len: ", len(self.coeffs))
        #wavelet_decomposition [cA_n, cD_n, cD_n-1, …, cD2, cD1]
        if self.coeffs:
            #remove (len-level) coefficients out of len. The smoothest 
            for i in range(level , len(self.coeffs) ): 
                coeffs[-i] = np.zeros_like(coeffs[-i]) 
            recon = waverec( coeffs, self.wavelet_name , mode= "smooth")
        return recon 

if __name__ == "__main__":
    #filename = "2018/hourly_1530608400.json"
    path = "2018"
    c = Crawler()
    # Create a DataFrame from the crawled files
    # 17/09/2013 to 06/07/2018
    df2 = c.get_complete_df ( path, ['close', 'high'] )

    #Extracts a list of the close data
    # data = df2['close'].tolist()[-4096:]
    data = df2['close'][-8096:]
    print(data.head())
    print(data.tail())
    # data = df2['close'].tolist()[-32768:]
    d = Wavelet_Wrapper( data.tolist(), padding = False)
    coeffs = d.wavelet_decomposition( )
    recon = d.reconstruct( level = 1)
    #wavelet_decomposition [cA_n, cD_n, cD_n-1, …, cD2, cD1]

    # reconstruction 
    #plt.subplot(2,1,1) #row, col, index
    plt.plot( d.data, label='Original Data' )
    plt.plot (recon, color='red', label='Wavelet Reconstruction' )
    plt.xlabel("t: 22/10/2017 to 06/07/2018")
    plt.ylabel("USD")
    plt.title("Wavelet reconstruction of Bitcoin USD Price")
    plt.legend(loc='upper right',shadow=True, fontsize='medium')
    plt.show()
    
    #d.plot_coefficients( coeffs, "db8")
    # from pywt import wavelist
    # for wavelet_name in wavelist():
    #     print("Wavelet name:", wavelet_name)
    #     try:
    #         coeffs = d.wavelet_decomposition( wavelet_name= wavelet_name)
    #     except: 
    #         print('Bad wavelet name: ', wavelet_name)
    #     d.plot_coefficients( coeffs, wavelet_name)
    

    #plt.show()
