from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import helpers
from sklearn.model_selection import train_test_split
import scipy.stats
import joblib
from matplotlib.axes import Axes
from time import time

type_list=["paramecium stained", "lillium anther", "usaf_lr", "bee wing", "ipomoea leaf",None]

def data_extract(types:str):
    global data
    data=helpers.load_focus_data("portfolio/project/focus_score_curves_dataset.pkl",offset=10,types=types)

def absdiff(num1,num2):
    return np.abs(num1-num2)

def full(test_frac=0.2,maxdepth=12):
    total=np.arange(data.num_images)
    train_idx, test_idx=train_test_split(total,test_size=test_frac)

    mean=[]
    variance=[]
    max_depths=[i for i in range(1,maxdepth+1)]

    training_data=[[],[]]

    for idx in train_idx:
        for depth in range(len(data.norm_scores[idx,:,:])):
            training_data[0].append(data.norm_scores[idx,depth,:]+[p**2 for p in data.norm_scores[idx,depth,:]])
            training_data[1].append(absdiff(data.depths[depth],data.true_depths[idx]))

    test_data=[[],[]]

    for idx in test_idx:
        for depth in range(len(data.norm_scores[idx,:,:])):
            test_data[0].append(data.norm_scores[idx,depth,:]+[p**2 for p in data.norm_scores[idx,depth,:]])
            test_data[1].append(absdiff(data.depths[depth],data.true_depths[idx]))

    def trainerpred(maxdepth):
        tree_reg=DecisionTreeRegressor(max_depth=maxdepth,min_samples_leaf=20)

        tree_reg.fit(training_data[0],training_data[1])

        tested=tree_reg.predict(test_data[0])

        sta=scipy.stats.describe(np.abs(np.array(test_data[1])-np.array(tested)))
        # print("max depth {}".format(i))
        return(sta.mean,sta.variance)
    # print("##############################")
    paral=joblib.Parallel(n_jobs=12,prefer="threads")
    objec=paral(joblib.delayed(trainerpred)(i) for i in range(1,13))
    # print("##############################")
    # print(objec)
    # print("##############################")
    mean=[i[0] for i in objec]
    variance=[i[1] for i in objec]

    return (np.array(max_depths),np.array(mean),np.array(variance))

def plot_band(depths,values,errors):
    # Axes.fill_between()
    plt.fill_between(depths,values-errors,values+errors,alpha=0.3)

def ex_plot(max_depths,mean,variance,band=False,mean_band=None,variance_band=None,save=False):
    plt.clf()
    plt.plot(max_depths,mean)
    plt.xlabel("max Depths")
    plt.ylabel("Mean")
    if band:
        plot_band(max_depths,mean,mean_band)
    if save:
        plt.savefig("{}mean.png".format(save_path))
    else:    
        plt.show()
    plt.clf()

    plt.plot(max_depths,variance)
    plt.xlabel("max Depths")
    plt.ylabel("Variance")
    if band:
        plot_band(max_depths,variance,variance_band)
    if save:
        plt.savefig("{}var.png".format(save_path))
    else:
        plt.show()
    plt.clf()

    plt.plot(10*variance,mean,'x')
    plt.plot([min(10*variance),max(10*variance)],[min(mean),max(mean)],'-')
    plt.xlabel("Variance")
    plt.ylabel("Mean")
    if save:
        plt.savefig("{}varmean.png".format(save_path))
    else:
        plt.show()
    plt.clf()
# max_depths,mean,variance=full()
# ex_plot(max_depths,mean,variance)

def averaging_optimising(attempts,maxdepth=12,test_size=0.2):
    assert attempts>0
    max_depths=[i for i in range(1,maxdepth+1)]
    MEMmeans=[]
    MEMvariances=[]
    for i in range(attempts):
        max_depths,mean,variance=full(maxdepth=maxdepth,test_frac=test_size)
        MEMmeans.append(mean)
        MEMvariances.append(variance)
    # print(MEMmeans)
    MEMmeans=np.array(MEMmeans).reshape(maxdepth,attempts)
    # print(MEMmeans)
    MEMvariances=np.array(MEMvariances).reshape(maxdepth,attempts)
    return(max_depths,MEMmeans,MEMvariances)

# max_depths,means,variances=averaging_optimising(5)

def multi_post_processing(attempts,maxdepth=12,test_size=0.2):
    max_depths,means,variances=averaging_optimising(attempts,maxdepth,test_size)
    final_mean=[]
    final_variance=[]
    mean_band=[]
    variance_band=[]
    for i in means:
        final_mean.append(scipy.stats.describe(i).mean)
        mean_band.append(scipy.stats.describe(i).variance)
    for i in variances:
        final_variance.append(scipy.stats.describe(i).mean)
        variance_band.append(scipy.stats.describe(i).variance)
    # print(means)
    # print(variance_band)
    ex_plot(max_depths=max_depths,mean=np.array(final_mean),variance=np.array(final_variance),band=True,mean_band=np.array(mean_band),variance_band=np.array(variance_band),save=True)

#just running it all
for current_type in type_list:
    if type(current_type)!=str:
        print("all classes")
        save_path="portfolio/project/img_all/all"
    else:
        print(current_type)
        save_path="portfolio/project/img_{}/{}".format(current_type,current_type)    
    data_extract(current_type)
    start=time()
    multi_post_processing(100)
    end=time()
    print(end-start)