# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:32:45 2023

@author: manuv
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn import cluster
import err_ranges as err
import numpy as np

def curve(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1960))
    return f
def read_data(fn):
    """ 
    This function read data 
    return transposed value and original value
    """
    df = pd.read_csv(fn)
    dd = df.drop(columns=["Country Code","Indicator Name","Indicator Code"])
    print(dd)
    # replacing the nul vaules with 0
    dd = dd.replace(np.nan,0)
    u = ["Colombia","India"]
    # selecting the countries from the list
    dd = dd["Country Name"].isin(u)
    dd = df[dd]
    # dropping some columns
    dd = dd.drop(columns={"Country Code","Indicator Name","Indicator Code","Country Name"})
    print(dd)
    # transposing 
    dt = np.transpose(dd)
    dt = dt.reset_index()
    print(dt)
    # renaming the data with index as year
    dt = dt.rename(columns={"index":"year"})
    dt = dt.rename(columns={109:"INDIA",45:"COLOMBIA"})
    dt = dt.dropna()
    dt["COLOMBIA"] = pd.to_numeric(dt["COLOMBIA"])
    dt["INDIA"] = pd.to_numeric(dt["INDIA"])
    dt["year"] = pd.to_numeric(dt["year"])
    print(dt)
    return df,dt;
data,dt = read_data("C:\\ads 2\\CO2 emissions from liquid fuel consumption (kt).csv")
sdata,sdt = read_data("C:\\ads 2\\CO2 emissions from solid fuel consumption (kt).csv")
print(sdt)

# optimising the curve
param,cp = opt.curve_fit(curve,dt["year"],dt["INDIA"],p0=[4e8, 0.1])
print(*param)
# taking the error value
sigma = np.sqrt(np.diag(cp))
# low and up values for error ranges
low,up = err.err_ranges(dt["year"],curve,param,sigma)
# fitting
dt["fit"] = curve(dt["year"],*param)
plt.plot(dt["year"],dt["INDIA"],label="data")
plt.plot(dt["year"],dt["fit"],c="red",label="fit")
# plot the error ranges in the graph
plt.fill_between(dt["year"],low,up,alpha=0.6)
plt.legend()
plt.show()
# predicting 
plt.figure()
plt.title("PREDICTION[2035]")
pred_year = np.arange(1960,2035)
pred_ind = curve(pred_year,*param)
plt.plot(dt["year"],dt["INDIA"],label="data")
plt.plot(pred_year,pred_ind,label="prediction")
plt.title("INDIA")
plt.legend()
plt.show()


plt.figure()
plt.title("INDIA")
param,cp = opt.curve_fit(curve,sdt["year"],sdt["INDIA"],p0=[4e8, 0.1])
print(*param)
sigma = np.sqrt(np.diag(cp))
low,up = err.err_ranges(sdt["year"],curve,param,sigma)
sdt["fit"] = curve(sdt["year"],*param)
plt.plot(sdt["year"],sdt["INDIA"],label="data")
plt.plot(sdt["year"],sdt["fit"],c="red",label="fit")
plt.fill_between(sdt["year"],low,up,alpha=0.6)
plt.legend()
plt.show()
plt.figure()
plt.title("PREDICTION[2035]")
pred_year = np.arange(1960,2035)
pred_ind = curve(pred_year,*param)
plt.plot(sdt["year"],sdt["INDIA"],label="data")
plt.plot(pred_year,pred_ind,label="prediction")
plt.legend()
plt.show()


dt = dt.iloc[:,1:3]
print(dt)
kmean = cluster.KMeans(n_clusters=2).fit(dt)
label = kmean.labels_
plt.scatter(dt["COLOMBIA"],dt["INDIA"],c=label,cmap="jet")
plt.title("INDIA VS COLOMBIA")
c = kmean.cluster_centers_
for s in range(2):
    xc,yc = c[s,:]
    plt.plot(xc,yc,"dk",markersize=13)
plt.figure()
plt.show()
india = pd.DataFrame()
india["co2_liquid"] = dt["INDIA"]
india["co2_solid"] = sdt["INDIA"]
kmean = cluster.KMeans(n_clusters=2).fit(india)
label = kmean.labels_
plt.scatter(india["co2_liquid"],india["co2_solid"],c=label,cmap="jet")
plt.title("co2_liquid VS co2_solid")
c = kmean.cluster_centers_
for s in range(2):
    xc,yc = c[s,:]
    plt.plot(xc,yc,"dk",markersize=13)
plt.figure()
plt.show()



