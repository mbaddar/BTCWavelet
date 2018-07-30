import numpy as np 
from pandas import DataFrame as DF 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec, idwt, waverec


from crawlers.crawler import Crawler
import math
import json

class Decomposition:
    # __data = []
    def __init__ (self, data):
        length = len(data)
        if not self.power2( length):
            pad = self.next_power_of_2(length) - length
            for _ in range(pad): 
                data.append( 0 )
        self.__data = data

    def power2(self, num):
        return ((num & (num - 1)) == 0) and num != 0

    def next_power_of_2(self, x):  
        return 1 if x == 0 else 2**(x - 1).bit_length()

    def zeros(self, n):
        if isinstance(n, int):
            return [0]*n
        else:
            raise TypeError("Number must be integer")

    def wavelet_decomposition(self, wavelet_name ='haar' , level = None):
        """
        2-D Array of coefficients
        [cA_n, cD_n, cD_n-1, …, cD2, cD1]
        """
        #TODO: raise an error of name not in family of wavelets
        coeffs = wavedec( self.__data , wavelet_name , level= level)
        return coeffs
    
    def plot_coefficients( self, coeffs, wavelet_name ):

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

if __name__ == "__main__":
    #filename = "2018/hourly_1530608400.json"
    path = "2018"
    c = Crawler()
    # Create a DataFrame from the crawled files
    df2 = c.get_complete_df ( path, ['close', 'high'] )
    print("data size :" ,df2.close.size)
    #Extracts a list of the close data
    data = df2['close'].tolist()[-4096:]
    # data = df2['close'].tolist()[-32768:]
    d = Decomposition( data)
    coeffs = d.wavelet_decomposition( wavelet_name= "db8")
    #wavelet_decomposition [cA_n, cD_n, cD_n-1, …, cD2, cD1]

    # reconstruction 
    for i in range(2,8): #remove 6 coefficients
        coeffs[-i] = np.zeros_like(coeffs[-i]) 
    recon = waverec( coeffs, "db8", mode= "smooth")
    plt.plot(data)
    plt.plot(recon)
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
