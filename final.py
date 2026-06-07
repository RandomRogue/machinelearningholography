import matplotlib.pyplot as plt
import numpy as np
import helpers
from scipy.stats import describe
from sklearn.model_selection import train_test_split
import joblib
from time import time
import sklearn.tree
import sklearn.metrics


type_list=["paramecium stained", "lillium anther", "usaf_lr", "bee wing", "ipomoea leaf",None]

def plot_band(depths,values,errors):
    plt.fill_between(depths,values-errors,values+errors,alpha=0.3)

def plot(depths,values,labelx:str,labely:str,line_type:str,band=False,variance=None,save=False,filename=None,folderpath=None,shows=True):
    plt.plot(depths,values,line_type)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    if band:
        plot_band(depths,values,variance)
    if save:
        plt.savefig("{}{}.png".format(folderpath,filename))
    if shows:
        plt.show()
    plt.clf()

def absdiff(num1,num2):
    return np.abs(num1-num2)

def data_extract(types:str,cutoff=10):
    return(helpers.load_focus_data("portfolio/project/focus_score_curves_dataset.pkl",offset=cutoff,types=types))

def data_split(data:helpers.FocusData,test_frac:float=0.2):
    total=np.arange(data.num_images)
    train_idx, test_idx=train_test_split(total,test_size=test_frac)
    return(train_idx, test_idx)

def data_prep(data:helpers.FocusData,idx_list):
    prepped=([],[])
    for idx in idx_list:
        for depth in range(len(data.norm_scores[idx,:,:])):
            prepped[0].append(data.norm_scores[idx,depth,:])
            prepped[1].append(absdiff(data.depths[depth],data.true_depths[idx]))
    return prepped

# def train(model,train_set):
    # model.fit(train)

def complete_tree(mac_depth,min_leaves):
    for ty in type_list:
        data=data_extract(types=ty)
        train_idx,test_idx=data_split(data=data)
        train,train_target=data_prep(data,train_idx)
        test,test_target=data_prep(data,test_idx)
        model=sklearn.tree.DecisionTreeRegressor(max_depth=mac_depth,min_samples_leaf=min_leaves)
    
        model.fit(train,train_target)
        pred=model.predict(test)
        print()
        print(describe(pred).mean/sklearn.metrics.root_mean_squared_error(test_target,pred))


complete_tree(7,10)