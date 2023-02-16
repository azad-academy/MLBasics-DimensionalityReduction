'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''

import sys
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import sklearn
import sklearn.manifold as manifold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import *
from scipy.stats import multivariate_normal
from time import time

'''
X = Data
Y = Cluster Labels (1,2,...), 0 for noise
'''
def plot_data(X,Y=None,canvas=None,xlabel=None,ylabel=None,zlabel=None,plt_title=None,colmap=plt.cm.coolwarm,show_legend=False,view_point=(0,10,5),show_axis=True):
        
    if(canvas is None):
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        ax = canvas
        ax.cla()
    
    if(plt_title is not None):
        ax.set_title(plt_title)  

    if(X.shape[1]>2):
        scatter = ax.scatter(X[:,0],X[:,1],X[:,2],c=Y,cmap=colmap,edgecolors='black',alpha=0.7)
        ax.view_init(elev=view_point[1],azim=view_point[0])
        ax.dist = view_point[2]
    else:
        scatter = ax.scatter(X[:,0],X[:,1],c=Y,cmap=colmap,edgecolors='black',alpha=0.7)
    if(show_legend):
        L = ax.legend(*scatter.legend_elements(),loc="lower right", title="Class")
        
    
               
    if(xlabel is not None):
        ax.set_xlabel(xlabel,fontweight='bold',fontsize=16)
    
    if(ylabel is not None):
        ax.set_ylabel(ylabel,fontweight='bold',fontsize=16)
    if(zlabel is not None):
        ax.set_zlabel(zlabel,fontweight='bold',fontsize=16)
    
    if(not show_axis):
        ax.grid(False)
        ax.axis('off')        

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def generate_3d_curve(N,rstate_seed=111):

    n = int(N/3)

    X1, Y1 = make_s_curve(n, random_state=rstate_seed+222)
    X1 = (X1-np.mean(X1,axis=0))/np.std(X1,axis=0)
    X2, Y2 = make_s_curve(n, random_state=rstate_seed+333)
    X2 = (X2-np.mean(X2,axis=0))/np.std(X2,axis=0)
    X2[:,0] += 2.5
    X2[:,2] += 2

    X3, Y3 = make_s_curve(n, random_state=rstate_seed+444)
    X3 = (X3-np.mean(X3,axis=0))/np.std(X3,axis=0)
    X3[:,0] += -2.7
    X3[:,1] += -1 
    X3[:,2] += -1.7

    X = np.concatenate((X3,X1,X2))
    Y = np.concatenate((Y3,Y1,Y2))
    min_Y = Y.min()
    max_Y = Y.max()
    Y = np.linspace(min_Y,max_Y,len(Y))

    return X,Y

def generate_twin_rolls(N,rstate_seed=111):

    n = int(N/2)
    X1, Y1 = make_swiss_roll(n, random_state=rstate_seed+111)
    X1 = (X1-np.mean(X1,axis=0))/np.std(X1,axis=0)
    X2, Y2 = make_swiss_roll(n, random_state=rstate_seed+222)
    X2 = -(X2-np.mean(X2,axis=0))/np.std(X2,axis=0)
    X2[:,0] += 1
    X2[:,2] += 3.5

    X = np.concatenate((X1,X2))
    Y = np.concatenate((Y1,Y2))
    min_Y = Y.min()
    max_Y = Y.max()
    Y = np.linspace(min_Y,max_Y,len(Y))

    return X,Y


from matplotlib import offsetbox
def plot_embedding(X,Y, imgs , ax, title):
    
    X = MinMaxScaler().fit_transform(X)
    
    for y in np.unique(Y):
        ax.scatter(X[y == Y,0],X[y == Y,1],
            color=plt.cm.tab10(y),
            alpha=0.8,
            zorder=1,
        )
    
    shown_images = np.array([[1.0, 1.0]]) 
    for i in range(X.shape[0]):
        
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1e-2:
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(imgs[i], cmap=plt.cm.hot), X[i])
        imagebox.set(zorder=2)
        ax.add_artist(imagebox)
    
    ax.set_title(title)
    ax.axis("off")
   
