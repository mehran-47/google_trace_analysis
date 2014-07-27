'''
KERNEL DENSITY ESTIMATION EXAMPLE
This code can be divided into 3 parts:
1) To calculate the moving average of a given random distribution
2) Calculate the outlier data points based on average and standard deviation
3) Estimate the density of the data occurence by Kernel Density

HOW TO USE THIS CODE
This code reads the configuration parameters from the file config.txt
config has the following parameters

1) Input_type - Generate data or Read data
    Generate data - generates random data in gaussian distribution of the size specified in Num_of_data_points
    Read data - reads from the file in json format (file name should be sample_json)
2) Num_of_intervals - defines the number of intervals needed to analyze the kernel density
    Eg) if the data has 10000 points, then Num_of_intervals =4 means it is splitted into 0-2500, 2500-5000, 5000-7500, 7500-10000
    and Kernel density for that data range is drawn
3) Output_type - this does not change anything, Included for future work. Right now the output is plotted and
    written to files in json format
4) Num_of_data_points - can set the number of data points needed if we need to generate sample data
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas.stats.moments import ewma,ewmstd,ewmvar
from pandas.core.api import DataFrame, Series, notnull
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import json
import jsonpickle
import jsonpickle.pickler as pick
import jsonpickle.unpickler as unpick
from sklearn.linear_model import RidgeCV
import pylab as pl
from scipy import interpolate
from collections import OrderedDict


#defining global variables
#these values will be used if config file did not have the parameters
Input_type = 'Generate data'
num_of_intervals = 5
Output_type = 'Plot'
N = 10000
#to identify whether the data is regular or sparse data
data_type = ''


#All function definitions
#Main function
def main():

    #declaring these global variables again
    global Input_type
    global num_of_intervals
    global Output_type
    global N
    global data_type

    '''
    This is the main function
    '''
    
    mean=50
    std=12
    np.random.seed(1)
    data_dict = {}
    fig_title = ''

    #read config parameters from config file
    Input_type, num_of_intervals, Output_type, N = read_config()

    #gen_sparse_data(mean, std)
    #sparse_data = read_sparse_data()
    gen_json(mean, std)

    if(Input_type == 'Generate data'):
        '''For generate data, random data points are generated in json format and that data is used to do the analysis'''
        #write data to a json file
        gen_json(mean, std)

        #read data from a json file
        data_dict = read_json('sample_json')
        fig_title = 'Gaussian distribution'

    if(Input_type == 'Read data'):
        '''Read data reads the actual data from the json file'''
        #read data from a json file
        data_dict = read_json('sample_json')
        #data_dict = read_json('sample_json_sparse.txt')
        fig_title = 'Gaussian distribution'

    #actual length of the data got by any of the means above
    N = len(data_dict)
    data_dict = OrderedDict(sorted(data_dict.items(), key=lambda t: t[0]))

    #analysis differs slightly for Sparse and Regular data 
    if(N<1000):
        data_type = 'Sparse'
        num_of_intervals = 1
    else:
        data_type = 'Regular'

    import json
    with open('data_dict.json', 'w') as outfile:
        json.dump(data_dict, outfile)    
    #dict to dataframe
    df=dict_to_df(data_dict)
    #df = pd.read_json('data_dict.json')
    
    #plotting the entire data series with upper lower thresholds
    df2 = exp_smoothing(df)
    linestyle=['rx','','','','']
    df3=df2.drop('Std Deviation', axis=1)
    DataFrame.plot(df3,title=fig_title+" %d points " %N,use_index=True,legend="true",style=linestyle)

    #calculate points above and below threshold
    above_threshold_dict, below_threshold_dict = calculate_outliers(df3)

    #KDE for above threshold
    #outliers are distributed wrt to the range specified by the user
    ab_thresh_range_dict = outliers_with_range(above_threshold_dict)
    #outliers with range and time to plot the data with respect to time
    ab_thresh_range_time_dict = outliers_with_range_and_time(above_threshold_dict)
    #plot the kernel density and the outlier distribution
    plot_KDE(ab_thresh_range_dict, ab_thresh_range_time_dict, 'above')
    kde_dict_above = KDE_data(ab_thresh_range_dict, ab_thresh_range_time_dict, 'above')

    #KDE for below threshold
    be_thresh_range_dict = outliers_with_range(below_threshold_dict)
    be_thresh_range_time_dict = outliers_with_range_and_time(below_threshold_dict)
    plot_KDE(be_thresh_range_dict, be_thresh_range_time_dict, 'below')
    kde_dict_below = KDE_data(be_thresh_range_dict, be_thresh_range_time_dict, 'below')

    #writing all output data in the form of json
    output_dict_to_json(kde_dict_above, 'KDE', 'above', depth = 2)
    output_dict_to_json(kde_dict_below, 'KDE', 'below', depth = 2)
    output_dict_to_json(ab_thresh_range_time_dict, 'outlier_data', 'above', depth = 2)
    output_dict_to_json(be_thresh_range_time_dict, 'outlier_data', 'below', depth = 2)

    if(data_type == 'Sparse'):
        #Also plotting the entire data and its KDE (only for sparse data)
        ab_thresh_range_dict = outliers_with_range(data_dict)
        #outliers with range and time to plot the data with respect to time
        ab_thresh_range_time_dict = outliers_with_range_and_time(data_dict)
        #plot the kernel density and the outlier distribution
        plot_KDE(ab_thresh_range_dict, ab_thresh_range_time_dict, 'sparse')
        kde_dict_above = KDE_data(ab_thresh_range_dict, ab_thresh_range_time_dict, 'sparse')

    #Finally displaying the plots'''
    plt.show()
    
    
def read_config():
    '''
    To read config parameters from config file
    '''

    with open('config.txt', 'r') as config:
        for line in config: 
            config_data = json.loads(line)
            Input_type = config_data[0]['Input_type']
            Num_of_intervals = config_data[0]['Num_of_intervals']
            Output_type = config_data[0]['Output_type']
            Num_of_data_points = config_data[0]['Num_of_data_points']
            return Input_type, Num_of_intervals, Output_type, Num_of_data_points 

def gen_json(mean, std):
    '''
    To generate a sample json data file if no input is given, this will be useful
    if we are just simulating instead of providing actual data
    '''
    #index generated based on the number of intervals
    index = np.arange(start=0,stop=N,step=1,dtype=int)

    #Sample data with gaussian distribution
    data = np.random.normal(loc=mean, scale=std, size=N)
    data_dict = {}

    for i in range(index.size):
        data_dict[index[i]]=data[i]
    
    import json
    with open('sample_json', 'w') as outfile:
        json.dump(data_dict, outfile)    
    #output_dict_to_json(data_dict, 'sample','json', depth = 1)    


def read_json(file_name):

    '''
    To read the given data in the form of json file
    '''

    u = unpick.Unpickler()
    data_dict = {}
    data = []
    result_dict = {}
    with open(file_name, 'r') as json_input:
        for line in json_input:
            data = json.loads(line)
            data_dict = u.restore(data)

    #conversion from string to int and float
            
    for keys, values in data_dict.items():
        result_dict[int(keys)] = float(values)

    return result_dict 



def dict_to_df(data_dict):
    '''
    Function to convert a given dictionary to dataframe using pandas Dataframe
    '''
    
    df = DataFrame.from_dict(data_dict, orient='index')
    df.columns = ['Original data']
    return df

def exp_smoothing(df):
    '''
    Function to calculate exponential smoothing
    Pandas functions are used to calculate moving average and standard deviation
    Upper and lower thresholds are calculated by
    Upper threshold = moving average + standard deviation
    Lower threshold = moving average - standard deviation
    '''
    #Calculate exp weighted moving average
    ExpWeightedMovingAvg = ewma(df,span=N)
    ExpWeightedMovingAvg.columns=['Exp Weighted Moving Avg']

    #Calculate exp weighted moving std
    ExpWeightedMovingStd = ewmstd(df,span=N)
    ExpWeightedMovingStd.columns=['Std Deviation']

    s1=df['Original data']
    s2=ExpWeightedMovingAvg['Exp Weighted Moving Avg']
    s3=ExpWeightedMovingStd['Std Deviation']
    s4=s2.add(s3, fill_value=0)
    s5=s2.sub(s3, fill_value=0)

    df2=DataFrame(s1,columns=['Original data'])
    df2['Exp Weighted Moving Avg']=s2
    df2['Std Deviation']=s3
    df2['Upper Threshold']=s4
    df2['Lower Threshold']=s5

    return df2

def calculate_outliers(df):
    '''
    constructing dictionary of above and lower threshold values
    '''
    above_threshold_dict=dict()
    below_threshold_dict=dict()

    for index, row in df.iterrows():
        if(df['Original data'].ix[index]>df['Upper Threshold'].ix[index]):
            above_threshold_dict[index]=df['Original data'].ix[index]
        elif(df['Original data'].ix[index]<df['Lower Threshold'].ix[index]):
            below_threshold_dict[index]=df['Original data'].ix[index]     
    return above_threshold_dict, below_threshold_dict

def outliers_with_range(outlier_dict):
    '''
    populating dictionary which will later be used for kernel distribution
    this dictionary will have time split into ranges defined by the user
    Eg) dictionary will have data in the range 0-2000, 2000-4000 and so on
    '''
    intervals = (N/num_of_intervals)
    outlier_range_dict=dict()

    for keys, values in outlier_dict.items():
        low_lim=0
        high_lim=intervals
        for i in np.arange(0, N, intervals):
            if(low_lim<=keys<high_lim):
                temp_key=(str(low_lim)+'-'+str(high_lim))
                if(temp_key in outlier_range_dict.keys()):
                    temp_list=list(outlier_range_dict[temp_key])
                    temp_list.append(values)
                    outlier_range_dict[temp_key]=temp_list
                else:
                    temp_list=list()
                    temp_list.append(values)
                    outlier_range_dict[temp_key]=temp_list                    
                break
            else:
                low_lim+=intervals
                high_lim+=intervals

    return outlier_range_dict

def outliers_with_range_and_time(outlier_dict):
    '''
    Function to populate a dictionary with range and time scale (dictionary of dictionaries)
    Used to plot the outlier data with respect to time
    '''
    intervals = (N/num_of_intervals)
    outlier_range_time_dict=dict()

    for keys, values in outlier_dict.items():
        low_lim=0
        high_lim=intervals
        for i in np.arange(0, N, intervals):
            if(low_lim<=keys<high_lim):
                temp_key=(str(low_lim)+'-'+str(high_lim))
                if(temp_key in outlier_range_time_dict.keys()):
                    outlier_range_time_dict[temp_key][keys]=outlier_dict[keys]
                else:
                    outlier_range_time_dict[temp_key]={}
                    outlier_range_time_dict[temp_key][keys]=outlier_dict[keys]
                break
            else:
                low_lim+=intervals
                high_lim+=intervals
    return outlier_range_time_dict

def plot_KDE(outlier_range_dict, outlier_range_time_dict, name):

    '''
    To estimate kernel density and to plot the data vs density curves with respect to the time ranges
    '''
    
    intervals = N/num_of_intervals
    low_lim=0
    high_lim=intervals
    j=1
    k=num_of_intervals+1

    fig=plt.figure()

    #following statements are used to set the title for the figure

    if(name == 'above'):
        fig.suptitle("Outlier distribution (Above Threshold) and corresponding KDE", fontsize=20)
    elif(name == 'below'):
        fig.suptitle("Outlier distribution (Below Threshold) and corresponding KDE", fontsize=20)
    elif(name == 'sparse'):
        fig.suptitle("CPU data distribution and corresponding KDE", fontsize=20)

    lowest_value_list = list()
    highest_value_list = list()
    #to set a fixed data axis based on lowest and highest values
    for i in np.arange(0, N, intervals):
        temp_key=(str(low_lim)+'-'+str(high_lim))
        if(temp_key in outlier_range_dict.keys() and temp_key in outlier_range_time_dict.keys()):
            outlier_values=list(outlier_range_dict[temp_key])
            outlier_array=np.array(outlier_values)[:, np.newaxis]
            lowest_value_list.append(np.amin(outlier_array))
            highest_value_list.append(np.amax(outlier_array))

    #find the lowest and highest of all values so that index can be set
    lowest_value_array = np.array(lowest_value_list)
    highest_value_array = np.array(highest_value_list)
    lowest_value = np.amin(lowest_value_array)
    highest_value = np.amax(highest_value_array)
    #index_outlier=np.linspace(lowest_value, highest_value, outlier_array.size)[:, np.newaxis]
    ticks_outlier=np.arange(lowest_value, highest_value+0.1, (highest_value - lowest_value)/5)
    ticks_outlier = np.around(ticks_outlier,decimals = 2)

    for i in np.arange(0, N, intervals):
        temp_key=(str(low_lim)+'-'+str(high_lim))
        cv = 0
        temp_cv = 0
        if(temp_key in outlier_range_dict.keys() and temp_key in outlier_range_time_dict.keys()):
            outlier_values=list(outlier_range_dict[temp_key])
            outlier_array=np.array(outlier_values)[:, np.newaxis]            
            index_outlier=np.linspace(lowest_value, highest_value, outlier_array.size)[:, np.newaxis]
            if(int(outlier_array.size/20)>4):
                cv = 20
            else:
                for i in np.arange(2, 20, 1):
                    if(int(outlier_array.size/i) > 4):
                        temp_cv = int(outlier_array.size/i)
                        cv = temp_cv
                    elif(outlier_array.size > 4):
                        cv = temp_cv
                        break
                    else:
                        cv = 2
            
            # for selecting the bandwidth
            grid_outlier = GridSearchCV(KernelDensity(),
                                {'bandwidth': np.linspace(0.1, 1.0, 30)},
                                cv = cv) # n-fold cross-validation
            grid_outlier.fit(outlier_array)
            kde_outlier=grid_outlier.best_estimator_
            log_dens_outlier = kde_outlier.score_samples(index_outlier)
            #1st series of plots (Kernel density plots)
            plt.subplot(2,num_of_intervals,j)
            plt.plot(index_outlier, np.exp(log_dens_outlier), '-')
            plt.title("Optimal bandwidth = %0.4f" %kde_outlier.bandwidth)
            plt.xlabel("CPU Utilization")
            plt.ylabel("Density")
            plt.xlim(xmin=lowest_value, xmax=highest_value)
            plt.xticks(ticks_outlier)
            #increment j by 1 so that the next plot can be drawn
            j+=1

            #2nd series of plots (outlier data)
            temp_dict={}
            temp_dict=outlier_range_time_dict[temp_key]
            
            a=list(temp_dict.keys())
            b=list(temp_dict.values())
            plt.subplot(2,num_of_intervals,k)
            plt.plot(a, b, 'rx')
            if(name == 'sparse'):
                plt.title(temp_key+" CPU utilization data")
            else: 
                plt.title(temp_key+" Outliers")
            plt.xlabel("Time")
            plt.ylabel("CPU Utilization")
            plt.ylim(ymin=lowest_value, ymax=highest_value)
            plt.yticks(ticks_outlier)
            #increment k by 1 so that the next plot can be drawn
            k+=1
           
            low_lim+=intervals
            high_lim+=intervals



def KDE_data(outlier_range_dict, outlier_range_time_dict, name):

    '''
    To get the Kernel density data in the form of a dictionary with time ranges
    '''

    intervals = N/num_of_intervals
    low_lim=0
    high_lim=intervals
    j=1
    k=num_of_intervals+1

    #dictionary for kde
    kde_dict = dict()

    trd_list = list()
    for i in np.arange(0, N, intervals):
        temp_key=(str(low_lim)+'-'+str(high_lim))
        cv = 0
        temp_cv = 0             
        if(temp_key in outlier_range_dict.keys() and temp_key in outlier_range_time_dict.keys()):
            outlier_values=list(outlier_range_dict[temp_key])
            outlier_array=np.array(outlier_values)[:, np.newaxis]
            #to find the lowest and highest values in the array
            lowest_value = np.amin(outlier_array)
            highest_value = np.amax(outlier_array)
            test = (highest_value - lowest_value)/5
            test2 = int(test)
            index_outlier=np.linspace(lowest_value, highest_value, outlier_array.size)[:, np.newaxis]

            if(int(outlier_array.size/20)>4):
                cv = 20
            else:
                for i in np.arange(2, 20, 1):
                    if(int(outlier_array.size/i) > 4):
                        temp_cv = int(outlier_array.size/i)
                        cv = temp_cv
                    elif(outlier_array.size > 4):
                        cv = temp_cv
                        break
                    else:
                        cv = 2

            # for selecting the bandwidth
            grid_outlier = GridSearchCV(KernelDensity(),
                                {'bandwidth': np.linspace(0.1, 1.0, 30)},
                                cv = cv) # n-fold cross-validation
            grid_outlier.fit(outlier_array)
            kde_outlier=grid_outlier.best_estimator_
            log_dens_outlier = kde_outlier.score_samples(index_outlier)

            dens_outlier = np.exp(log_dens_outlier)
            for i in np.arange(0, index_outlier.shape[0], 1):
                if(temp_key in kde_dict.keys()):
                    kde_dict[temp_key][float(index_outlier[i])]=float(dens_outlier[i])
                else:
                    kde_dict[temp_key]={}

            low_lim+=intervals
            high_lim+=intervals

    return kde_dict


def output_dict_to_json(source_dict, name, type, depth):
    '''
    Function to write the source dictionary to a json format
    Output json file is named with 2 parameters name and type
    json pickle API is used to convert a dictionary to json format
    '''

    p = pick.Pickler(max_depth = depth)
    u = unpick.Unpickler()
    data = p.flatten(source_dict)
    
    #clear the file contents before writing
    open(name+'_'+type, 'w').close()

    #converting the dictionary to json file
    with open(name+'_'+type, 'a') as json_output:
        json_output.write(json.dumps(data))

#Calling main function
if __name__ == '__main__':
    main()