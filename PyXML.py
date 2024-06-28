# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:17:52 2024

@author: ali_d
"""
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import font
from tkinter import messagebox 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler 
from sklearn.preprocessing import RobustScaler, Normalizer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
import math
from tkinter import PhotoImage
import webbrowser
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lime
import shap
from anchor import anchor_tabular


root = tk.Tk()
root.title("PyXML")
root.geometry("640x510")

my_notebook = ttk.Notebook(root)
my_notebook.pack(pady=5)
# Create font object
label_font = font.Font(family="Segoe UI", weight="bold", size=9)

importData = Frame(my_notebook, width=640, height=510)
preprocess = Frame(my_notebook, width=640, height=510) 
dataProperty = Frame(my_notebook, width=640, height=510)  
learning = Frame(my_notebook, width=640, height=510) 
perfMetric = Frame(my_notebook, width=640, height=510) 
expML = Frame(my_notebook, width=640, height=510) 
info = Frame(my_notebook, width=640, height=510) 

importData.pack(fill="both", expand=1)
preprocess.pack(fill="both", expand=1)
dataProperty.pack(fill="both", expand=1)
learning.pack(fill="both", expand=1)
perfMetric.pack(fill="both", expand=1)
expML.pack(fill="both", expand=1)
info.pack(fill="both", expand=1)

my_notebook.add(importData, text="Import Data")
my_notebook.add(preprocess, text="Preprocessing")
my_notebook.add(dataProperty, text="Dataset Property")
my_notebook.add(learning, text="Training")
my_notebook.add(perfMetric, text="Evaluate")
my_notebook.add(expML, text="Explainable Machine Learning")
my_notebook.add(info, text="Credits")

fileName = tk.StringVar()
saveName = tk.StringVar()
randomState =tk.IntVar(value=42)
split_ratio = tk.IntVar()
split_ratio.set(75)
mlMethod = tk.StringVar()
constInput = tk.DoubleVar()
constInput.set(0.0)
scaleMethod = tk.StringVar()
data = ""
X_train_scaled = ""
xmlMethod = tk.StringVar()
shap_L_G = tk.StringVar()
shapPlot = tk.StringVar()



############################################################################
# Data loading functions tab


def openFile():
    global df, data, X_train, X_test, y_train, y_test
    df = data = X_train = X_test = y_train = y_test= None

    
    filePath = filedialog.askopenfilename(initialdir="D:\PyXML",
                                          title="Open file",
                                          filetypes= (("csv files","*.csv"),
                                          ("all files","*.*")))
    fileName.set(filePath)
    
    filePath = r"{}".format(filePath)
    df = pd.read_csv(filePath)
    

    clear_tree(tree1)
    

    tree1["column"] = list(df.columns)
    tree1["show"] = "headings"

    for column in tree1["column"]:
        tree1.heading(column, text=column)
    

    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tree1.insert("", "end", values= row)
    
def saveFile():
    filePath2 = filedialog.askopenfilename(title='Open a file')
    saveName.set(filePath2)
    
def clear_tree(del_tree):
    del_tree.delete(*del_tree.get_children())
    
def show_data(data, showTree):

    clear_tree(showTree)
    

    showTree["column"] = list(df.columns)
    showTree["show"] = "headings"

    for column in showTree["column"]:
        showTree.heading(column, text=column)
    

    data_rows = data.tolist()
    for row in data_rows:
        showTree.insert("", "end", values= row)
        
############################################################################
# Preprocessing tab functions
    
def showMissing():
    global df, data
    
    pp_fig_frame.tkraise()
    ax_PP.clear()
    missing_data = df.isnull().mean().to_numpy()*100
    bars = ax_PP.barh(df.columns, missing_data)
    ax_PP.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_PP.xaxis.set_visible(False)
    ax_PP.bar_label(bars, padding=-50, color='white', label_type='edge', 
                 fmt='%.2f%%') 
    canvas_PP.draw()
        
def meanImputation():
    global df, data
    pp_tree_frame.tkraise()
    imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
    imputer = imputer.fit(df)
    data = imputer.transform(df)
    show_data(data, tree2)

def medianImputation():
    global df, data
    pp_tree_frame.tkraise()
    imputer = SimpleImputer(missing_values = np.nan, strategy ='median')
    imputer = imputer.fit(df)
    data = imputer.transform(df)
    show_data(data, tree2)

def mostFrequent():
    global df, data
    pp_tree_frame.tkraise()
    imputer = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')
    imputer = imputer.fit(df)
    data = imputer.transform(df)
    show_data(data, tree2)

def constantImputation():
    global df, data
    pp_tree_frame.tkraise()
    imputer = SimpleImputer(missing_values = np.nan, strategy ='constant',
                            fill_value=float(constVal.get()))
    imputer = imputer.fit(df)
    data = imputer.transform(df)
    show_data(data, tree2)

def scaleSelect():
    global df, data, X_train, X_test, y_train, y_test
    global X_train_scaled, X_test_scaled
    pp_tree_frame.tkraise()


    if randomState.get() < 0:
        randVal = "None"
        shuffle_cond = True
    else:
        randVal = randomState.get()
        shuffle_cond = False

    
    cols = df.shape[1]
    X = df.iloc[:,0:cols-1]
    y = df.iloc[:,cols-1].values
    
    print(split_ratio.get()/100)
    
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, random_state=randVal, train_size = split_ratio.get()/100, 
        shuffle=True)
    
    if mlMethod.get() == 'standartScale':
        sScaler = StandardScaler()
        X_train_scaled = sScaler.fit_transform(X_train)
        X_test_scaled = sScaler.transform(X_test)
    elif mlMethod.get() == 'minMaxScale':
        mmScaler = MinMaxScaler()
        X_train_scaled = mmScaler.fit_transform(X_train)
        X_test_scaled = mmScaler.transform(X_test)
    elif mlMethod.get() == 'normalizeScale':
        nScaler = Normalizer()
        X_train_scaled = nScaler.fit_transform(X_train)
        X_test_scaled = nScaler.transform(X_test)
    elif mlMethod.get() == 'maxAbsScale':
        maScaler = MaxAbsScaler()
        X_train_scaled = maScaler.fit_transform(X_train)
        X_test_scaled = maScaler.transform(X_test)
    elif mlMethod.get() == 'medQuantileScale':
        rScaler = RobustScaler()
        X_train_scaled = rScaler.fit_transform(X_train)
        X_test_scaled = rScaler.transform(X_test)

def showTrain():
    global df, data, X_train, X_test, y_train, y_test
    global X_train_scaled, X_test_scaled
    pp_tree_frame.tkraise()
    print(X_train_scaled)
    if isinstance(X_train_scaled, str):
        cols = df.shape[1]
        X = df.iloc[:,0:cols-1]
        y = df.iloc[:,cols-1].values
        X_train, X_test, y_train, y_test = train_test_split( 
            X, y, random_state=randVal, train_size = split_ratio.get()/100, 
            shuffle=shuffle_cond)
        X_train_scaled = X_train
        X_test_scaled = X_test
    

    clear_tree(tree2)
    

    tree2["column"] = list(df.columns)
    tree2["show"] = "headings"

    for column in tree2["column"]:
        tree2.heading(column, text=column)
        
    train_data = np.hstack((X_train_scaled, y_train.reshape(-1,1)))

    data_rows = train_data.tolist()
    for row in data_rows:
        tree2.insert("", "end", values= row)

def showTest():
    global df, data, X_train, X_test, y_train, y_test
    global X_train_scaled, X_test_scaled
    pp_tree_frame.tkraise()
    
    if isinstance(X_test_scaled, str):
        X_train_scaled = X_train
        X_test_scaled = X_test
        

    clear_tree(tree2)
    

    tree2["column"] = list(df.columns)
    tree2["show"] = "headings"

    for column in tree2["column"]:
        tree2.heading(column, text=column)
        
    test_data = np.hstack((X_test_scaled, y_test.reshape(-1,1)))

    data_rows = test_data.tolist()
    for row in data_rows:
        tree2.insert("", "end", values= row)

############################################################################
# Dataset property tab functions

def dataStats():
    global df
    X = df.iloc[:,:-1]
    
    properties_df2 = pd.concat({'Feature': X.columns.to_series(), 'Min': X.min(axis=0), 'Max': X.max(axis=0), 
                                'Mean': X.mean(axis=0),'Median': X.median(axis=0),
                                'Std dev': X.std(axis=0), 'Variance':X.var(axis=0), 
                                'Range': (X.max(axis=0)- X.min(axis=0))},axis=1)

    dp_tree_frame.tkraise()

    clear_tree(tree3)
  

    tree3["column"] = list(properties_df2.columns)
    tree3["show"] = "headings"

    for column in tree3["column"]:
        tree3.heading(column, text=column)


    data_rows = properties_df2.values.tolist()
    for row in data_rows:
        tree3.insert("", "end", values= row)
    
def dataVisualize():
    global df
    dp_fig_frame.tkraise()
    clear_frame(dp_fig_frame)
    
    fig_DPP, ax_DPP = plt.subplots()
    s = sns.boxplot(data = df.iloc[:,:-1])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    s.set(yscale="log")
    

    canvas= FigureCanvasTkAgg(fig_DPP, master=dp_fig_frame)  
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    
    toolbar_DPP = NavigationToolbar2Tk(canvas, dp_fig_frame, pack_toolbar = False)
    toolbar_DPP.update()
    toolbar_DPP.pack(anchor = "w", fill = tk.X)

def correlation():
    global df
    dp_fig_frame.tkraise()
    clear_frame(dp_fig_frame)
    
    fig_DPP, ax_DPP = plt.subplots()
    sns.heatmap(df.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')
    

    canvas= FigureCanvasTkAgg(fig_DPP, master=dp_fig_frame)  
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    
    toolbar_DPP = NavigationToolbar2Tk(canvas, dp_fig_frame, pack_toolbar = False)
    toolbar_DPP.update()
    toolbar_DPP.pack(anchor = "w", fill = tk.X)
 
def anova():
    global df, data
    dp_fig_frame.tkraise()
    clear_frame(dp_fig_frame)
    
    data_fs = df
    cols = data_fs.shape[1]
    X_fs = data_fs.iloc[:,0:cols-1]
    y_fs = data_fs.iloc[:,cols-1].values
    
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X_fs, y_fs)
    
    fig_DPP, ax_DPP = plt.subplots()
    bars = ax_DPP.bar(X_fs.columns, fs.scores_)
    ax_DPP.tick_params(axis='x', labelrotation=90)
    
    canvas = FigureCanvasTkAgg(fig_DPP, master=dp_fig_frame)  
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def mutualInformation():
    global df, data
    dp_fig_frame.tkraise()
    clear_frame(dp_fig_frame)
    
    data_fs = df
    cols = data_fs.shape[1]
    X_fs = data_fs.iloc[:,0:cols-1]
    y_fs = data_fs.iloc[:,cols-1].values
    
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_fs, y_fs)
    
    fig_DPP, ax_DPP = plt.subplots()
    bars = ax_DPP.bar(X_fs.columns, fs.scores_)
    ax_DPP.tick_params(axis='x', labelrotation=90)
    
    canvas = FigureCanvasTkAgg(fig_DPP, master=dp_fig_frame)  
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    
def clear_frame(c_frame):
    for widget in c_frame.winfo_children():
        widget.destroy()

############################################################################
# Training tab functions
def trainTestSplit():
    print("1")

def mlSelect():
    if mlMethod.get() == 'LogReg':
        learning_LR.tkraise()
    elif mlMethod.get() == "SGD":
        learning_SGD.tkraise()
    elif mlMethod.get() == "NB":
        learning_NB.tkraise()
    elif mlMethod.get() == "kNN":
        learning_kNN.tkraise()
    elif mlMethod.get() == "DT":
        learning_DT.tkraise()
    elif mlMethod.get() == "RF":
        learning_RF.tkraise()
    elif mlMethod.get() == "AdaBoost":
        learning_adaBoost.tkraise()
    elif mlMethod.get() == "GB":
        learning_GB.tkraise()
    elif mlMethod.get() == "XGB":
        learning_XGB.tkraise()
    elif mlMethod.get() == "SVM":
        learning_SVM.tkraise()
        
def applyML():
    global df, data, y_train, y_test, X_train_scaled, X_test_scaled, clf
    global X_train_scaled_df, X_test_scaled_df, feature_names
    
    feature_names = df.columns
    feature_names = feature_names[:-1]
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns =feature_names) 
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns =feature_names) 
       
    if mlMethod.get() == 'LogReg':
        print("lr_penalty: ", lr_penalty.get() , "max_iter_lr: ", lr_max_iter.get(), "lr_solver: ", lr_solver.get() )
        clf = LogisticRegression(penalty =lr_penalty.get(),solver = lr_solver.get(),
                                 max_iter = lr_max_iter.get())
        ml_txt = 'The Logistic Regression model is successfully built.'
    elif mlMethod.get() == 'SGD':
        clf = SGDClassifier(loss = sgd_loss .get(), penalty = sgd_penalty.get(),
                            alpha = float(sgd_alpha.get()), 
                            l1_ratio = float(sgd_l1_ratio.get()),
                            max_iter = sgd_max_iter.get())
        ml_txt = 'The Stochastic Gradient Descent model is successfully built.'        
    elif mlMethod.get() == "NB":
        clf = GaussianNB(var_smoothing=float(nb_var_smoothing.get()))
        ml_txt = 'The Naive Bayes model is successfully built.'
    elif mlMethod.get() == "kNN":
        clf = KNeighborsClassifier(n_neighbors = knn_neighbors.get(),
                                   weights = knn_weights.get(),
                                   algorithm = knn_algorithm.get(),
                                   leaf_size = knn_leaf_size.get(),
                                   metric = knn_metric.get())  
        ml_txt = 'The k Nearest Neighbors model is successfully built.'
    elif mlMethod.get() == "DT":
        if dt_max_depth.get() == "None":
            clf = DecisionTreeClassifier(criterion = dt_criterion.get(),
                                          splitter = dt_splitter.get(),
                                          max_depth = None,
                                          min_samples_split = dt_min_samples_split.get(),
                                          min_samples_leaf = dt_min_samples_leaf.get(),
                                          max_features =  dt_max_features.get())
        else:
            clf = DecisionTreeClassifier(criterion = dt_criterion.get(),
                                          splitter = dt_splitter.get(),
                                          max_depth = int(dt_max_depth.get()),
                                          min_samples_split = dt_min_samples_split.get(),
                                          min_samples_leaf = dt_min_samples_leaf.get(),
                                          max_features =  dt_max_features.get())
        ml_txt = 'The Decision Tree model is successfully built.'
    elif mlMethod.get() == "RF":
        if rf_max_depth.get() == "None":
            clf = RandomForestClassifier(n_estimators = rf_n_estimators.get(),
                                         criterion = rf_criterion.get(),
                                         max_depth = None,
                                         min_samples_split = rf_min_samples_split.get(),
                                         min_samples_leaf = rf_min_samples_leaf.get(),
                                         max_features = rf_max_features.get())
        else:
            clf = RandomForestClassifier(n_estimators = rf_n_estimators.get(),
                                         criterion = rf_criterion.get(),
                                         max_depth = int(rf_max_depth.get()),
                                         min_samples_split = rf_min_samples_split.get(),
                                         min_samples_leaf = rf_min_samples_leaf.get(),
                                         max_features = rf_max_features.get())
        ml_txt = 'The Random Forest model is successfully built.'
    elif mlMethod.get() == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators = adaBoost_n_estimators.get(),
                                 learning_rate = float(adaBoost_learning_rate.get()),
                                 algorithm = adaBoost_algorithm.get())
        ml_txt = 'The AdaBoost model is successfully built.'
    elif mlMethod.get() == "GB":
        if gb_max_leaf_nodes.get() == "None":
            clf = GradientBoostingClassifier(loss = gb_loss.get(),
                                         learning_rate = float(gb_learning_rate.get()),
                                         n_estimators = gb_n_estimators.get(),
                                         subsample = float(gb_subsample.get()),
                                         criterion = gb_criterion.get(),
                                         min_samples_split = gb_min_samples_split.get(),
                                         min_samples_leaf = gb_min_samples_leaf.get(),
                                         max_depth = gb_max_depth.get(),
                                         min_impurity_decrease = float(gb_min_impurity_decrease .get()),
                                         max_features = gb_max_features .get(),
                                         max_leaf_nodes = None)
        else:
            clf = GradientBoostingClassifier(loss = gb_loss.get(),
                                         learning_rate = float(gb_learning_rate.get()),
                                         n_estimators = gb_n_estimators.get(),
                                         subsample = float(gb_subsample.get()),
                                         criterion = gb_criterion.get(),
                                         min_samples_split = gb_min_samples_split.get(),
                                         min_samples_leaf = gb_min_samples_leaf.get(),
                                         max_depth = gb_max_depth.get(),
                                         min_impurity_decrease = float(gb_min_impurity_decrease .get()),
                                         max_features = gb_max_features .get(),
                                         max_leaf_nodes = int(gb_max_leaf_nodes.get()))
        ml_txt = 'The Gradient Boosting model is successfully built.'
    elif mlMethod.get() == "XGB":
        clf = XGBClassifier(learning_rate = float(xgb_learning_rate.get()),
                            min_split_loss = xgb_min_split_loss.get(),
                            max_depth = xgb_max_depth.get(),
                            min_child_weight = xgb_min_child_weight.get(),
                            max_delta_step = xgb_max_delta_step.get(),
                            subsample = float(xgb_subsample.get()),
                            reg_lambda = xgb_lambda.get(),
                            reg_alpha = xgb_alpha.get(),
                            tree_method = xgb_tree_method.get(),
                            max_leaves = xgb_max_leaves.get(),
                            max_bin = xgb_max_bin.get())
        ml_txt = 'The XGBoost model is successfully built.'
    elif mlMethod.get() == "SVM":
        gamma_v = svm_gamma.get()
        if not(gamma_v == "scale" or gamma_v == "auto"):
            gamma_v = float(svm_gamma.get())
        if svm_kernel.get() == "linear":
            clf = svm.SVC(kernel=svm_kernel.get(), C = float(svm_C.get()),
                          probability=True)
        else:
            clf = svm.SVC(kernel=svm_kernel.get(), C = float(svm_C.get()),
                          gamma = gamma_v, probability=True)
        ml_txt = 'The Support Vector Machine model is successfully built.'
    clf.fit(X_train_scaled_df,y_train)
    
# compute performance metrics
    y_pred = clf.predict(X_test_scaled_df)
    y_pred_proba = clf.predict_proba(X_test_scaled_df)[::,1]
    cm = metrics.confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    accuracy = (cm[1,1]+cm[0,0]) / (cm[1,1]+cm[0,0]+cm[0,1]+cm[1,0]) 
    specificity = cm[0,0] / (cm[0,0]+cm[0,1])
    precision = cm[1,1] / (cm[1,1]+cm[0,1])
    recall = cm[1,1] / (cm[1,1]+cm[1,0])
    F1_score = 2*(precision*recall) / (precision+recall)
    g_mean = math.sqrt(recall * specificity)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    pm_accuracy.set(round(accuracy,6))
    pm_precision.set(round(precision,6))
    pm_recall.set(round(recall,6))
    pm_specificity.set(round(specificity,6))
    pm_f1_score.set(round(F1_score,6))
    pm_g_mean.set(round(g_mean,6))
    pm_roc_auc.set(round(auc,6))
    
    pm_cm_frame.tkraise()
    clear_frame(pm_cm_frame)
    
    fig_PM_cm, ax_PM_cm = plt.subplots()
    group_names = ['True Negatives','False Positives','False Negatives','True Positives']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    s = sns.heatmap(cm, annot=labels, fmt='', cmap='tab20', annot_kws={"size":10},
                xticklabels=['False', 'True'], yticklabels=['False', 'True'], cbar=False)
    # s.set_xlabel('Predicted Class', fontsize=10)
    s.set_ylabel('Actual Class', fontsize=10)
    s.set_title('Predicted Class', fontsize=10)
    # create tkinter canvas from figure
    canvas_PM_cm= FigureCanvasTkAgg(fig_PM_cm, master=pm_cm_frame)  
    canvas_PM_cm_widget = canvas_PM_cm.get_tk_widget()
    canvas_PM_cm_widget.pack(fill=tk.BOTH, expand=True)
    canvas_PM_cm.draw()
       
    pm_roc_frame.tkraise()
    clear_frame(pm_roc_frame)
   
    fig_PM_roc, ax_PM_roc = plt.subplots()
    ax_PM_roc.plot(fpr, tpr)
    ax_PM_roc.plot([0, 1], ls="--")
    ax_PM_roc.set_title('Receiver Operating Characteristic')
    ax_PM_roc.plot([0, 0], [1, 0] , c=".7"), ax_PM_roc.plot([1, 1] , c=".7")
    ax_PM_roc.set_xlabel('False Positive Rate')
    ax_PM_roc.set_ylabel('True Positive Rate')
        
    canvas_PM_roc = FigureCanvasTkAgg(fig_PM_roc, master=pm_roc_frame) 
    canvas_PM_roc_widget = canvas_PM_roc.get_tk_widget()
    canvas_PM_roc_widget.pack(fill=tk.BOTH, expand=True)
    canvas_PM_roc.draw()

    toolbar_PM = NavigationToolbar2Tk(canvas_PM_roc, pm_roc_frame, pack_toolbar = False)
    toolbar_PM.update()
    toolbar_PM.pack(anchor = "w", fill = tk.X)
        
    print("auc= ", auc)
    print(clf.get_params()) # for printing the parameters of the model
    # messagebox.showinfo("Machine Learning", clf.get_params()) 
    messagebox.showinfo("Machine Learning", ml_txt)


############################################################################
# XML tab functions
    
def xmlSelect():
    if xmlMethod.get() == 'Lime':
        expML_lime.tkraise()
    elif xmlMethod.get() == "Shap":
        expML_shap.tkraise()
    elif xmlMethod.get() == "Anchors":
        expML_anchors.tkraise()
    
def applyLime():
    global df, X_train_scaled, X_test_scaled, y_train, y_test, clf
    entry_text = lime_k_width.get()
    if not entry_text.strip(): 
        lime_k_width.set(np.sqrt(X_train_scaled.shape[1]) * .75)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled,
                                                        discretizer=lime_discretizer.get(),
                                                        feature_names=df.columns[:-1].tolist(),
                                                        kernel_width= float(lime_k_width.get()),
                                                        class_names=['Negative','Positive'], 
                                                        discretize_continuous=True)

    exp = explainer.explain_instance(X_test_scaled[lime_n_sample.get()], 
                                     clf.predict_proba, 
                                     num_features=lime_n_featrues.get(),
                                     distance_metric = lime_d_metric.get(), 
                                     num_samples = lime_n_samples.get())

    lime_save = filedialog.asksaveasfile(filetypes = [('HTML', '*.html')])
    html_file_path = lime_save.name + '.html'
    exp.save_to_file(html_file_path)
    webbrowser.open_new_tab(f"file://{html_file_path}")
    
def applyShap():
    global df, explainer, X_train_scaled, X_test_scaled, clf, shap_values
    global s_ML, X_train_scaled_df, X_test_scaled_df, shap_EV
    
    s_ML = mlMethod.get()
    feature_names = df.columns
    feature_names = feature_names[:-1]
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns =feature_names) 
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns =feature_names) 
    
    if  s_ML == "DT" or s_ML == "RF":
        explainer = shap.TreeExplainer(clf,X_train_scaled_df)
        shap_values = explainer(X_test_scaled_df,check_additivity=False)
        shap_EV = explainer.expected_value[1]
    else:
        explainer = shap.Explainer(clf.predict_proba, X_train_scaled_df)
        shap_values = explainer(X_test_scaled_df)
        shap_EV = np.mean(clf.predict(X_train))
    
    messagebox.showinfo("Shap process", "Shap successfully implemented.") 
    
def shapSelect():
    if shap_L_G.get() == 'Local':
        local_shap.tkraise()
    elif shap_L_G.get() == 'Global':
        global_shap.tkraise()
        
def displayShap():
    global df, data, clf, shap_values, X_test_scaled_df, explainer, shap_EV
    
    feature_names = df.columns
    feature_names = feature_names[:-1]
    shapley_vals = [shap_values.values[:,:,0], shap_values.values[:,:,1]]
    
    if shapPlot.get() == 'l_bar_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.bar(shap_values[shap_s_number.get(),:,1],max_display = shap_l_max_disp.get())
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'l_decision_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.decision_plot(shap_EV, shapley_vals[1][shap_s_number.get()], 
                           X_test_scaled_df, feature_names=list(feature_names))
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'l_force_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.force_plot(shap_EV, shapley_vals[1][shap_s_number.get()],
                        X_test_scaled_df.iloc[shap_s_number.get()], 
                        feature_names=list(feature_names), matplotlib = True)
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'l_waterfall_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.waterfall(shap_values[shap_s_number.get(),:,1],  
                             max_display = shap_l_max_disp.get())
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'g_bar_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.bar(shap_values[:,:,1], max_display = shap_g_max_disp.get())
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'g_beeswarm_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.beeswarm(shap_values[:,:,1], max_display = shap_g_max_disp.get())
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'g_decision_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.decision_plot(shap_EV, shapley_vals[1], X_test_scaled_df, feature_names=list(feature_names))
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'g_heatmap_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.heatmap(shap_values[:,:,1], max_display = shap_g_max_disp.get())
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'g_violin_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.violin(shap_values[:,:,1], max_display=shap_g_max_disp.get(), feature_names=list(feature_names))
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()
    elif shapPlot.get() == 'g_summary_plot':
        shap_plot.tkraise()
        clear_frame(shap_plot)
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values[:,:,1], X_test_scaled_df)
        canvas_shap = FigureCanvasTkAgg(fig_shap, master=shap_plot) 
        canvas_shap_widget = canvas_shap.get_tk_widget()
        canvas_shap_widget.pack(fill=tk.BOTH, expand=True)
        canvas_shap.draw()

       
    
def applyAnchors():
    global df, X_train_scaled, X_test_scaled, y_train, y_test, clf
    anchors_plot.tkraise()
    clear_frame(anchors_plot)
    idx = anchors_n_sample.get()
    explainer_a = anchor_tabular.AnchorTabularExplainer(class_names=['Negative','Positive'],
                                                      feature_names=df.columns[:-1].tolist(),
                                                      train_data=X_train_scaled)
    exp_a = explainer_a.explain_instance(X_test_scaled[idx], 
                                         clf.predict, 
                                         threshold=float(anchors_threshold.get()),
                                         delta=float(anchors_delta.get()),
                                         tau=float(anchors_tau.get()),
                                         batch_size=anchors_batch_size.get())
    
    canvas_a = Canvas(anchors_plot)
    canvas_a.pack()
    XBASE, YBASE, DISTANCE = 10, 20, 20
    i = 0
    for i in range(len(exp_a.names())):
        if i == 0:
            canvas_a.create_text((XBASE, YBASE + i * DISTANCE), text="IF", anchor=W, fill='red', font=('Helvetica','11', 'bold'))
            canvas_a.create_text((XBASE+20, YBASE + i * DISTANCE), text=exp_a.names()[i], anchor=W, fill='blue', font=('Helvetica','12'))
        else:
            canvas_a.create_text((XBASE, YBASE + i * DISTANCE), text="AND", anchor=W, fill='red', font=('Helvetica','11', 'bold'))
            canvas_a.create_text((XBASE+40, YBASE + i * DISTANCE), text=exp_a.names()[i], anchor=W, fill='blue', font=('Helvetica','12'))

    canvas_a.create_text((XBASE, YBASE + (i+1) * DISTANCE), text="THEN PREDICT", anchor=W, fill='red', font=('Helvetica','11', 'bold'))
    canvas_a.create_text((XBASE+135, YBASE + (i+1) * DISTANCE), text=explainer_a.class_names[int(clf.predict(X_test_scaled[idx].reshape(1, -1))[0])], anchor=W, fill='blue', font=('Helvetica','12'))

    canvas_a.create_text((XBASE, YBASE + (i+2) * DISTANCE), text="WITH PRECISION", anchor=W, fill='red', font=('Helvetica','11', 'bold'))
    canvas_a.create_text((XBASE+135, YBASE + (i+2) * DISTANCE), text=f'{exp_a.precision():.2f}', anchor=W, fill='blue', font=('Helvetica','12'))

    canvas_a.create_text((XBASE, YBASE + (i+3) * DISTANCE), text="AND COVERAGE", anchor=W, fill='red', font=('Helvetica','11', 'bold'))
    canvas_a.create_text((XBASE+135, YBASE + (i+3) * DISTANCE), text=f'{exp_a.coverage():.2f}', anchor=W, fill='blue', font=('Helvetica','12'))

############################################################################
# Credits tab function
def open_url(url):
    webbrowser.open_new(url)

############################################################################
############################################################################
# Imnport data tab 

lbl1ID = ttk.Label(importData, text="Load Input Data")
lbl1ID.place(x = 10, y = 5, width=100)
dataPath = ttk.Entry(importData, textvariable=fileName)
dataPath.place(x = 110, y = 5, width=430)
browse = ttk.Button(importData, text="Browse", command=openFile)
browse.place(x = 550, y = 5, width=60)


tree1 = ttk.Treeview(importData)
tree1.place(x = 10, y = 35, width=605, height=410)

vsb = ttk.Scrollbar(importData, orient="vertical", command=tree1.yview)
vsb.place(x=620, y=30, height=420)
tree1.configure(yscrollcommand=vsb.set)

hsb = ttk.Scrollbar(importData, orient="horizontal", command=tree1.xview)
hsb.place(x=10, y=455, width=600)
tree1.configure(xscrollcommand=hsb.set)

############################################################################
# Preprocessing tab

# Frame for missing data visualization
pp_fig_frame = tk.Frame(preprocess,width=475,height=460)
pp_fig_frame.place(x = 155, y = 5, width=475, height=460)

# Create Canvas
fig_PP, ax_PP = plt.subplots()
canvas_PP = FigureCanvasTkAgg(fig_PP, master=pp_fig_frame)  
canvas_PP.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas_PP, pp_fig_frame, pack_toolbar = False)
toolbar.update()
toolbar.pack(anchor = "w", fill = tk.X)
 
# # Frame  for displaying dataset
pp_tree_frame = tk.Frame(preprocess,width=475,height=460)
pp_tree_frame.place(x = 155, y = 5, width=475, height=460)

tree2 = ttk.Treeview(pp_tree_frame)
tree2.place(x = 0, y = 0, width=459, height=443)

vsb2 = ttk.Scrollbar(pp_tree_frame, orient="vertical", command=tree2.yview)
vsb2.place(x=460, y=0, height=440)
tree2.configure(yscrollcommand=vsb2.set)

hsb2 = ttk.Scrollbar(pp_tree_frame, orient="horizontal", command=tree2.xview)
hsb2.place(x=0, y=443, width=460)
tree2.configure(xscrollcommand=hsb2.set)

lbl2 = ttk.Label(preprocess, text="Data Imputation", font=label_font)
lbl2.place(x = 5, y = 0, width=140, height=25)

iMean = ttk.Button(preprocess, text="Check Missing Data", command=showMissing)
iMean.place(x = 15, y = 25, width=133, height=25)

iMean = ttk.Button(preprocess, text="Mean", command=meanImputation)
iMean.place(x = 15, y = 55, width=133, height=25)

iMedian = ttk.Button(preprocess, text="Median", command=medianImputation)
iMedian.place(x = 15, y = 85, width=133, height=25)

imostFrequent = ttk.Button(preprocess, text="Most Frequent", command=mostFrequent)
imostFrequent.place(x = 15, y = 115, width=133, height=25)

iConstant = ttk.Button(preprocess, text="Constant", command=constantImputation)
iConstant.place(x = 15, y = 145, width=77, height=25)

constVal = ttk.Entry(preprocess, textvariable=constInput)
constVal.place(x = 98, y = 146, width=48, height=22)

separator1 = ttk.Separator(preprocess, orient="horizontal")
separator1.place(x = 15, y = 172, width=132)

lbl3 = ttk.Label(preprocess, text="Train/Test Split", font=label_font)
lbl3.place(x = 5, y = 173, width=120, height=25)

lbl4 = ttk.Label(preprocess, text="Random State: ")
lbl4.place(x = 15, y = 192, width=140, height=25)

spin_box = ttk.Spinbox(preprocess, from_=-1, to=4294967295,textvariable=randomState, wrap=True)
spin_box.place(x = 97, y = 193, width=49, height=20)

scale = tk.Scale(preprocess, variable= split_ratio, from_=0, to=100, 
                 orient="horizontal", showvalue=True, tickinterval= 20, 
                 length= 130) 
scale.place(x = 15, y = 214)

separator2 = ttk.Separator(preprocess, orient="horizontal")
separator2.place(x = 15, y = 272, width=132)



lf_DS = tk.LabelFrame(preprocess, text="Data Scaling",
                       font=label_font)
lf_DS.place(x = 5, y = 275, width=143, height=200)

ds_Standart= tk.Radiobutton( lf_DS, text="Standart",
                            wraplength=120, variable=mlMethod, 
                            value="standartScale", command=scaleSelect)
ds_Standart.grid(row=0, column=0, padx=0, pady=0, sticky='W' )

ds_MinMax= tk.Radiobutton( lf_DS, text="Min/Max",
                            wraplength=120, variable=mlMethod, 
                            value="minMaxScale", command=scaleSelect)
ds_MinMax.grid(row=1, column=0, padx=0, pady=0, sticky='W' )

ds_Mean= tk.Radiobutton( lf_DS, text="Normalize",
                            wraplength=120, variable=mlMethod, 
                            value="normalizeScale", command=scaleSelect)
ds_Mean.grid(row=2, column=0, padx=0, pady=0, sticky='W' )

ds_MaxAbs= tk.Radiobutton( lf_DS, text="Maximum Absolute",
                            wraplength=120, variable=mlMethod, 
                            value="maxAbsScale", command=scaleSelect)
ds_MaxAbs.grid(row=3, column=0, padx=0, pady=0, sticky='W' )

ds_MedQuantile= tk.Radiobutton(lf_DS, text="Median and Quantile",
                            wraplength=120, variable=mlMethod, 
                            value="medQuantileScale", command=scaleSelect)
ds_MedQuantile.grid(row=4, column=0, padx=0, pady=0, sticky='W' )

trainBtn = ttk.Button(lf_DS, text="Show Train Data", command=showTrain)
trainBtn.grid(row=5, column=0, padx=0, pady=0, sticky='EW' )

testBtn = ttk.Button(lf_DS, text="Show Test Data", command=showTest)
testBtn.grid(row=6, column=0, padx=0, pady=5, sticky='EW' )

############################################################################
# Dataset property tab

# Frame 1 for data visualization
dp_fig_frame = tk.Frame(dataProperty,bg="black",width=625,height=425)
dp_fig_frame.place(x = 5, y = 45,width=625,height=425)
 
# Frame 2 for displaying fearture statistics
dp_tree_frame = tk.Frame(dataProperty,bg='#d9d9d9',width=625,height=425)
dp_tree_frame.place(x = 5, y = 45, width=625, height=425)

tree3 = ttk.Treeview(dp_tree_frame)
tree3.place(x = 0, y = 0, width=610, height=413)

vsb3 = ttk.Scrollbar(dp_tree_frame, orient="vertical", command=tree3.yview)
vsb3.place(x=610, y=0, height=410)
tree3.configure(yscrollcommand=vsb3.set)

hsb3 = ttk.Scrollbar(dp_tree_frame, orient="horizontal", command=tree3.xview)
hsb3.place(x=0, y=413, width=610)
tree3.configure(xscrollcommand=hsb3.set)

lf_DP = tk.LabelFrame(dataProperty)
lf_DP.place(x = 5, y = 10, width=600, height=30)

statBtn = ttk.Button(lf_DP, text="Dataset Statistics", command=dataStats)
statBtn.grid(row=0, column=0, padx=10, pady=0, sticky='EW' )

dvBtn = ttk.Button(lf_DP, text="Dataset Visualization", command=dataVisualize)
dvBtn.grid(row=0, column=1, padx=10, pady=0, sticky='EW' )

corrBtn = ttk.Button(lf_DP, text="Correlation", command=correlation)
corrBtn.grid(row=0, column=2, padx=10, pady=0, sticky='EW' )

anovaBtn = ttk.Button(lf_DP, text="Anova", command=anova)
anovaBtn.grid(row=0, column=3, padx=10, pady=0, sticky='EW' )

miBtn = ttk.Button(lf_DP, text="Muttual Information", command=mutualInformation)
miBtn.grid(row=0, column=4, padx=10, pady=0, sticky='EW' )

############################################################################
# Training tab

# Frames of the machine learning radio buttons
# Logisitic Regression
learning_LR = Frame(learning, relief="raised", borderwidth = 3) 
learning_LR.place(x = 205, y = 40, width=420, height=420)

lr_penalty = tk.StringVar()
lr_solver = tk.StringVar()
lr_max_iter = tk.IntVar()
lr_max_iter.set(100)

lr_penalty_lbl= ttk.Label(learning_LR, text="Penalty: ")
lr_penalty_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
comboBox_penalty = ttk.Combobox(learning_LR, textvariable=lr_penalty, 
                        values=("none", "l1", "l2", "elasticnet"),
                        state="readonly")
comboBox_penalty.current(2)
comboBox_penalty.grid(row=0, column=1,  padx=20, pady=20, sticky='W')


lr_solver_lbl = ttk.Label(learning_LR, text="Solver: ")
lr_solver_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
comboBox_solver = ttk.Combobox(learning_LR, textvariable=lr_solver, 
                        values=("lbfgs", "liblinear", "newton-cg", "newton-cholesky",
                                "sag", "saga"),
                        state="readonly")
comboBox_solver.current(0)
comboBox_solver.grid(row=1, column=1, padx=20, pady=20, sticky='W')

lr_max_iter_lbl = ttk.Label(learning_LR, text="Maximum iterations: ")
lr_max_iter_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
max_iter_lr = ttk.Entry(learning_LR, textvariable=lr_max_iter)
max_iter_lr.grid(row=2, column=1, padx=20, pady=20, sticky='W')

# Stochastic Gradient Descent
learning_SGD = Frame(learning, relief="raised", borderwidth = 3) 
learning_SGD.place(x = 205, y = 40, width=420, height=420)

sgd_loss = tk.StringVar()
sgd_penalty = tk.StringVar()
sgd_alpha = tk.StringVar() 
sgd_alpha.set("0.0001")
sgd_l1_ratio = tk.StringVar() 
sgd_l1_ratio.set("0.15")
sgd_max_iter = tk.IntVar()
sgd_max_iter.set(1000)

sgd_loss_lbl = ttk.Label(learning_SGD, text="Splitter: ")
sgd_loss_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
sgd_loss_cb = ttk.Combobox(learning_SGD, textvariable=sgd_loss, 
                        values=("hinge", "log_loss", "modified_huber",
                                "squared_hinge", "perceptron", "squared_error",
                                "huber", "epsilon_insensitive"), state="readonly")
sgd_loss_cb.current(0)
sgd_loss_cb.grid(row=0, column=1, padx=20, pady=20, sticky='W')

sgd_penalty_lbl = ttk.Label(learning_SGD, text="Penalty: ")
sgd_penalty_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
sgd_penalty_cb = ttk.Combobox(learning_SGD, textvariable=sgd_penalty, 
                        values=("l2", "l1", "elasticnet", "none"), state="readonly")
sgd_penalty_cb.current(0)
sgd_penalty_cb.grid(row=1, column=1, padx=20, pady=20, sticky='W')

sgd_alpha_lbl = ttk.Label(learning_SGD, text="Alpha: ")
sgd_alpha_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
sgd_alpha_e = ttk.Entry(learning_SGD, textvariable=sgd_alpha)
sgd_alpha_e.grid(row=2, column=1, padx=20, pady=20, sticky='W')

sgd_l1_ratio_lbl = ttk.Label(learning_SGD, text="L1 ratio: ")
sgd_l1_ratio_lbl.grid(row=3, column=0, padx=20, pady=20, sticky='W')
sgd_l1_ratio_e = ttk.Entry(learning_SGD, textvariable=sgd_l1_ratio)
sgd_l1_ratio_e.grid(row=3, column=1, padx=20, pady=20, sticky='W')

sgd_max_iter_lbl = ttk.Label(learning_SGD, text="Maximum iteration: ")
sgd_max_iter_lbl.grid(row=4, column=0, padx=20, pady=20, sticky='W')
sgd_max_iter_e = ttk.Entry(learning_SGD, textvariable=sgd_max_iter)
sgd_max_iter_e.grid(row=4, column=1, padx=20, pady=20, sticky='W')

# Naive Bayes
learning_NB = Frame(learning, relief="raised", borderwidth = 3) 
learning_NB.place(x = 205, y = 40, width=420, height=420)

nb_var_smoothing = tk.StringVar() 
nb_var_smoothing.set("1e-9")

nb_var_smoothing_lbl = ttk.Label(learning_NB, text="Var_smoothing: ")
nb_var_smoothing_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
nb_var_smoothing_e = ttk.Entry(learning_NB, textvariable=nb_var_smoothing)
nb_var_smoothing_e.grid(row=1, column=1, padx=20, pady=20, sticky='W')

# k Nearest Neighbors
learning_kNN = Frame(learning, width=240, relief="raised", borderwidth = 3) 
learning_kNN.place(x = 205, y = 40, width=420, height=420)

knn_neighbors = tk.IntVar()
knn_neighbors.set(5)
knn_weights = tk.StringVar()
knn_algorithm = tk.StringVar()
knn_leaf_size = tk.IntVar()
knn_leaf_size.set(30)
knn_metric = tk.StringVar()

knn_n_neighbors_lbl = ttk.Label(learning_kNN, text="Number of neighbors: ")
knn_n_neighbors_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
knn_neighbors_e = ttk.Entry(learning_kNN, textvariable=knn_neighbors)
knn_neighbors_e.grid(row=0, column=1, padx=20, pady=20, sticky='W')

knn_weights_lbl = ttk.Label(learning_kNN, text="Weights: ")
knn_weights_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
knn_weights_cb = ttk.Combobox(learning_kNN, textvariable=knn_weights, 
                        values=("uniform", "distance"), state="readonly")
knn_weights_cb.current(0)
knn_weights_cb.grid(row=1, column=1, padx=20, pady=20, sticky='W')

knn_algorithm_lbl = ttk.Label(learning_kNN, text="Algorithm: ")
knn_algorithm_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
knn_algorithm_cb = ttk.Combobox(learning_kNN, textvariable=knn_algorithm, 
                        values=("auto", "ball_tree","kd_tree","brute"),
                        state="readonly")
knn_algorithm_cb.current(0)
knn_algorithm_cb.grid(row=2, column=1, padx=20, pady=20, sticky='W')

knn_leaf_size_lbl = ttk.Label(learning_kNN, text="Leaf size: ")
knn_leaf_size_lbl.grid(row=3, column=0, padx=20, pady=20, sticky='W')
knn_leaf_size_e = ttk.Entry(learning_kNN, textvariable=knn_leaf_size)
knn_leaf_size_e.grid(row=3, column=1, padx=20, pady=20, sticky='W')

knn_metric_lbl = ttk.Label(learning_kNN, text="Distance metric: ")
knn_metric_lbl.grid(row=4, column=0, padx=20, pady=20, sticky='W')
knn_metric_cb = ttk.Combobox(learning_kNN, textvariable=knn_metric, 
                        values=("euclidean", "manhattan","chebyshev","minkowski",
                                "seuclidean"), state="readonly")
knn_metric_cb.current(0)
knn_metric_cb.grid(row=4, column=1, padx=20, pady=20, sticky='W')

# Decision Tree
learning_DT = Frame(learning, relief="raised", borderwidth = 3) 
learning_DT.place(x = 205, y = 40, width=420, height=420)

dt_criterion = tk.StringVar()
dt_splitter = tk.StringVar()
dt_max_depth = tk.StringVar()
dt_max_depth.set("None")
dt_min_samples_split = tk.IntVar()
dt_min_samples_split.set(2)
dt_min_samples_leaf = tk.IntVar()
dt_min_samples_leaf.set(1)
dt_max_features = tk.IntVar()
dt_max_features.set(1)

dt_criterion_lbl = ttk.Label(learning_DT, text="Criterion: ")
dt_criterion_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
dt_criterion_cb = ttk.Combobox(learning_DT, textvariable=dt_criterion, 
                        values=("gini", "entropy", "log_loss"), state="readonly")
dt_criterion_cb.current(0)
dt_criterion_cb.grid(row=0, column=1, padx=20, pady=20, sticky='W')

dt_splitter_lbl = ttk.Label(learning_DT, text="Splitter: ")
dt_splitter_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
dt_splitter_cb = ttk.Combobox(learning_DT, textvariable=dt_splitter, 
                        values=("best", "random"), state="readonly")
dt_splitter_cb.current(0)
dt_splitter_cb.grid(row=1, column=1, padx=20, pady=20, sticky='W')

dt_max_depth_lbl = ttk.Label(learning_DT, text="Maximum depth: ")
dt_max_depth_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
dt_max_depth_e = ttk.Entry(learning_DT, textvariable=dt_max_depth)
dt_max_depth_e.grid(row=2, column=1, padx=20, pady=20, sticky='W')

dt_min_samples_split_lbl = ttk.Label(learning_DT, text="Minimum samples split: ")
dt_min_samples_split_lbl.grid(row=3, column=0, padx=20, pady=20, sticky='W')
dt_min_samples_split_e = ttk.Entry(learning_DT, textvariable=dt_min_samples_split)
dt_min_samples_split_e.grid(row=3, column=1, padx=20, pady=20, sticky='W')

dt_min_samples_leaf_lbl = ttk.Label(learning_DT, text="Minimum samples leaf: ")
dt_min_samples_leaf_lbl.grid(row=4, column=0, padx=20, pady=20, sticky='W')
dt_min_samples_leaf_e = ttk.Entry(learning_DT, textvariable=dt_min_samples_leaf)
dt_min_samples_leaf_e.grid(row=4, column=1, padx=20, pady=20, sticky='W')

dt_max_features_lbl = ttk.Label(learning_DT, text="Maximum features: ")
dt_max_features_lbl.grid(row=5, column=0, padx=20, pady=20, sticky='W')
dt_max_features_e = ttk.Entry(learning_DT, textvariable=dt_max_features)
dt_max_features_e.grid(row=5, column=1, padx=20, pady=20, sticky='W')

# Random Forest
learning_RF= Frame(learning, relief="raised", borderwidth = 3)
learning_RF.place(x = 205, y = 40, width=420, height=420)

rf_n_estimators = tk.IntVar()
rf_n_estimators.set(100)
rf_criterion = tk.StringVar()
rf_max_depth = tk.StringVar()
rf_max_depth.set("None")
rf_min_samples_split = tk.IntVar()
rf_min_samples_split.set(2)
rf_min_samples_leaf = tk.IntVar()
rf_min_samples_leaf.set(1)
rf_max_features = tk.IntVar()
rf_max_features.set(1)

rf_n_estimators_lbl = ttk.Label(learning_RF, text="Number of trees: ")
rf_n_estimators_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
rf_n_estimators_e = ttk.Entry(learning_RF, textvariable=rf_n_estimators)
rf_n_estimators_e.grid(row=0, column=1, padx=20, pady=20, sticky='W')

rf_criterion_lbl = ttk.Label(learning_RF, text="Criterion: ")
rf_criterion_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
rf_criterion_cb = ttk.Combobox(learning_RF, textvariable=rf_criterion, 
                        values=("gini", "entropy", "log_loss"), state="readonly")
rf_criterion_cb.current(0)
rf_criterion_cb.grid(row=1, column=1, padx=20, pady=20, sticky='W')

rf_max_depth_lbl = ttk.Label(learning_RF, text="Maximum depth: ")
rf_max_depth_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
rf_max_depth_e = ttk.Entry(learning_RF, textvariable=rf_max_depth)
rf_max_depth_e.grid(row=2, column=1, padx=20, pady=20, sticky='W')

rf_min_samples_split_lbl = ttk.Label(learning_RF, text="Minimum samples split: ")
rf_min_samples_split_lbl.grid(row=3, column=0, padx=20, pady=20, sticky='W')
rf_min_samples_split_e = ttk.Entry(learning_RF, textvariable=rf_min_samples_split)
rf_min_samples_split_e.grid(row=3, column=1, padx=20, pady=20, sticky='W')

rf_min_samples_leaf_lbl = ttk.Label(learning_RF, text="Minimum samples leaf: ")
rf_min_samples_leaf_lbl.grid(row=4, column=0, padx=20, pady=20, sticky='W')
rf_min_samples_leaf_e = ttk.Entry(learning_RF, textvariable=rf_min_samples_leaf)
rf_min_samples_leaf_e.grid(row=4, column=1, padx=20, pady=20, sticky='W')

rf_max_features_lbl = ttk.Label(learning_RF, text="Maximum features: ")
rf_max_features_lbl.grid(row=5, column=0, padx=20, pady=20, sticky='W')
rf_max_features_e = ttk.Entry(learning_RF, textvariable=rf_max_features)
rf_max_features_e.grid(row=5, column=1, padx=20, pady=20, sticky='W')

# AdaBoost
learning_adaBoost = Frame(learning, relief="raised", borderwidth = 3) 
learning_adaBoost.place(x = 205, y = 40, width=420, height=420)

adaBoost_n_estimators = tk.IntVar()
adaBoost_n_estimators.set(50)
adaBoost_learning_rate = tk.StringVar()
adaBoost_learning_rate.set(1.0)
adaBoost_algorithm = tk.StringVar()

adaBoost_n_estimators_lbl = ttk.Label(learning_adaBoost, text="Maximum estimators: ")
adaBoost_n_estimators_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
adaBoost_n_estimators_e = ttk.Entry(learning_adaBoost, textvariable=adaBoost_n_estimators)
adaBoost_n_estimators_e.grid(row=0, column=1, padx=20, pady=20, sticky='W')

adaBoost_learning_rate_lbl = ttk.Label(learning_adaBoost, text="Maximum estimators: ")
adaBoost_learning_rate_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
adaBoost_learning_rate_e = ttk.Entry(learning_adaBoost, textvariable=adaBoost_learning_rate)
adaBoost_learning_rate_e.grid(row=1, column=1, padx=20, pady=20, sticky='W')

adaBoost_algorithm_lbl = ttk.Label(learning_adaBoost, text="Algorithm: ")
adaBoost_algorithm_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
adaBoost_algorithm_cb = ttk.Combobox(learning_adaBoost, textvariable=adaBoost_algorithm, 
                        values=("SAMME", "SAMME.R"), state="readonly")
adaBoost_algorithm_cb.current(0)
adaBoost_algorithm_cb.grid(row=2, column=1, padx=20, pady=20, sticky='W')

# Gradient Boosting
learning_GB= Frame(learning, relief="raised", borderwidth = 3)
learning_GB.place(x = 205, y = 40, width=420, height=420)

gb_loss = tk.StringVar()
gb_learning_rate = tk.StringVar()
gb_learning_rate.set('0.1')
gb_n_estimators = tk.IntVar()
gb_n_estimators.set(100)
gb_subsample = tk.StringVar()
gb_subsample.set('1.0')
gb_criterion = tk.StringVar()
gb_min_samples_split = tk.IntVar()
gb_min_samples_split.set(2)
gb_min_samples_leaf = tk.IntVar()
gb_min_samples_leaf.set(1)
gb_max_depth = tk.IntVar()
gb_max_depth.set(3)
gb_min_impurity_decrease = tk.StringVar()
gb_min_impurity_decrease.set('0.0')
gb_max_features = tk.IntVar()
gb_max_features.set(1)
gb_max_leaf_nodes = tk.StringVar()
gb_max_leaf_nodes.set("None")

gb_loss_lbl = ttk.Label(learning_GB, text="Loss: ")
gb_loss_lbl.grid(row=0, column=0, padx=20, pady=8, sticky='W')
gb_loss_cb = ttk.Combobox(learning_GB, textvariable=gb_loss, 
                        values=("log_loss", "exponential"), state="readonly")
gb_loss_cb.current(0)
gb_loss_cb.grid(row=0, column=1, padx=20, pady=8, sticky='W')

gb_learning_rate_lbl = ttk.Label(learning_GB, text="Learning rate: ")
gb_learning_rate_lbl.grid(row=1, column=0, padx=20, pady=8, sticky='W')
gb_learning_rate_e = ttk.Entry(learning_GB, textvariable=gb_learning_rate)
gb_learning_rate_e.grid(row=1, column=1, padx=20, pady=8, sticky='W')

gb_n_estimators_lbl = ttk.Label(learning_GB, text="Number of boosting stages: ")
gb_n_estimators_lbl.grid(row=2, column=0, padx=20, pady=8, sticky='W')
gb_n_estimators_e = ttk.Entry(learning_GB, textvariable=gb_n_estimators)
gb_n_estimators_e.grid(row=2, column=1, padx=20, pady=8, sticky='W')

gb_subsample_lbl = ttk.Label(learning_GB, text="Number of boosting stages: ")
gb_subsample_lbl.grid(row=3, column=0, padx=20, pady=8, sticky='W')
gb_subsample_e = ttk.Entry(learning_GB, textvariable=gb_subsample)
gb_subsample_e.grid(row=3, column=1, padx=20, pady=8, sticky='W')

gb_criterion_lbl = ttk.Label(learning_GB, text="Criterion: ")
gb_criterion_lbl.grid(row=4, column=0, padx=20, pady=8, sticky='W')
gb_criterion_cb = ttk.Combobox(learning_GB, textvariable=gb_criterion, 
                        values=("friedman_mse", "squared_error"), state="readonly")
gb_criterion_cb.current(0)
gb_criterion_cb.grid(row=4, column=1, padx=20, pady=8, sticky='W')

gb_min_samples_split_lbl = ttk.Label(learning_GB, text="Minimum samples split: ")
gb_min_samples_split_lbl.grid(row=5, column=0, padx=20, pady=8, sticky='W')
gb_min_samples_split_e = ttk.Entry(learning_GB, textvariable=gb_min_samples_split)
gb_min_samples_split_e.grid(row=5, column=1, padx=20, pady=8, sticky='W')

gb_min_samples_leaf_lbl = ttk.Label(learning_GB, text="Minimum samples leaf: ")
gb_min_samples_leaf_lbl.grid(row=6, column=0, padx=20, pady=8, sticky='W')
gb_min_samples_leaf_e = ttk.Entry(learning_GB, textvariable=gb_min_samples_leaf)
gb_min_samples_leaf_e.grid(row=6, column=1, padx=20, pady=8, sticky='W')

gb_max_depth_lbl = ttk.Label(learning_GB, text="Maximum depth: ")
gb_max_depth_lbl.grid(row=7, column=0, padx=20, pady=8, sticky='W')
gb_max_depth_e = ttk.Entry(learning_GB, textvariable=gb_max_depth)
gb_max_depth_e.grid(row=7, column=1, padx=20, pady=8, sticky='W')

gb_min_impurity_decrease_lbl = ttk.Label(learning_GB, text="Minimum impurity decrease: ")
gb_min_impurity_decrease_lbl.grid(row=8, column=0, padx=20, pady=8, sticky='W')
gb_min_impurity_decrease_e = ttk.Entry(learning_GB, textvariable=gb_min_impurity_decrease)
gb_min_impurity_decrease_e.grid(row=8, column=1, padx=20, pady=8, sticky='W')

gb_max_features_lbl = ttk.Label(learning_GB, text="Maximum features: ")
gb_max_features_lbl.grid(row=9, column=0, padx=20, pady=8, sticky='W')
gb_max_features_e = ttk.Entry(learning_GB, textvariable=gb_max_features)
gb_max_features_e.grid(row=9, column=1, padx=20, pady=8, sticky='W')

gb_max_leaf_nodes_lbl = ttk.Label(learning_GB, text="Maximum leaf nodes: ")
gb_max_leaf_nodes_lbl.grid(row=10, column=0, padx=20, pady=8, sticky='W')
gb_max_leaf_nodes_e = ttk.Entry(learning_GB, textvariable=gb_max_leaf_nodes)
gb_max_leaf_nodes_e.grid(row=10, column=1, padx=20, pady=8, sticky='W')

# XGBoost
learning_XGB= Frame(learning, relief="raised", borderwidth = 3)
learning_XGB.place(x = 205, y = 40, width=420, height=420)

xgb_learning_rate = tk.StringVar()
xgb_learning_rate.set("0.3")
xgb_min_split_loss = tk.IntVar()
xgb_min_split_loss.set(0)
xgb_max_depth = tk.IntVar()
xgb_max_depth.set(6)
xgb_min_child_weight = tk.IntVar()
xgb_min_child_weight.set(0)
xgb_max_delta_step = tk.IntVar()
xgb_max_delta_step.set(0)
xgb_subsample = tk.StringVar()
xgb_subsample.set('1.0')
xgb_lambda = tk.IntVar()
xgb_lambda.set(1)
xgb_alpha = tk.IntVar()
xgb_alpha.set(0)
xgb_tree_method = tk.StringVar()
xgb_max_leaves = tk.IntVar()
xgb_max_leaves.set(0)
xgb_max_bin = tk.IntVar()
xgb_max_bin.set(256)

xgb_learning_rate_lbl = ttk.Label(learning_XGB, text="Learning rate: ")
xgb_learning_rate_lbl.grid(row=0, column=0, padx=20, pady=8, sticky='W')
xgb_learning_rate_e = ttk.Entry(learning_XGB, textvariable=xgb_learning_rate)
xgb_learning_rate_e.grid(row=0, column=1, padx=20, pady=8, sticky='W')

xgb_min_split_loss_lbl = ttk.Label(learning_XGB, text="Minimum loss: ")
xgb_min_split_loss_lbl.grid(row=1, column=0, padx=20, pady=8, sticky='W')
xgb_min_split_loss_e = ttk.Entry(learning_XGB, textvariable=xgb_min_split_loss)
xgb_min_split_loss_e.grid(row=1, column=1, padx=20, pady=8, sticky='W')

xgb_max_depth_lbl = ttk.Label(learning_XGB, text="Maximum depth: ")
xgb_max_depth_lbl.grid(row=2, column=0, padx=20, pady=8, sticky='W')
xgb_max_depth_e = ttk.Entry(learning_XGB, textvariable=xgb_max_depth)
xgb_max_depth_e.grid(row=2, column=1, padx=20, pady=8, sticky='W')

xgb_min_child_weight_lbl = ttk.Label(learning_XGB, text="Minimum child weight: ")
xgb_min_child_weight_lbl.grid(row=3, column=0, padx=20, pady=8, sticky='W')
xgb_min_child_weight_e = ttk.Entry(learning_XGB, textvariable=xgb_min_child_weight)
xgb_min_child_weight_e.grid(row=3, column=1, padx=20, pady=8, sticky='W')

xgb_max_delta_step_lbl = ttk.Label(learning_XGB, text="Maximum delta step: ")
xgb_max_delta_step_lbl.grid(row=4, column=0, padx=20, pady=8, sticky='W')
xgb_max_delta_step_e = ttk.Entry(learning_XGB, textvariable=xgb_max_delta_step)
xgb_max_delta_step_e.grid(row=4, column=1, padx=20, pady=8, sticky='W')

xgb_subsample_lbl = ttk.Label(learning_XGB, text="Subsample ratio: ")
xgb_subsample_lbl.grid(row=5, column=0, padx=20, pady=8, sticky='W')
xgb_subsample_e = ttk.Entry(learning_XGB, textvariable=xgb_subsample)
xgb_subsample_e.grid(row=5, column=1, padx=20, pady=8, sticky='W')

xgb_lambda_lbl = ttk.Label(learning_XGB, text="Lambda: ")
xgb_lambda_lbl.grid(row=6, column=0, padx=20, pady=8, sticky='W')
xgb_lambda_e = ttk.Entry(learning_XGB, textvariable=xgb_lambda)
xgb_lambda_e.grid(row=6, column=1, padx=20, pady=8, sticky='W')

xgb_alpha_lbl = ttk.Label(learning_XGB, text="Alpha: ")
xgb_alpha_lbl.grid(row=7, column=0, padx=20, pady=8, sticky='W')
xgb_alpha_e = ttk.Entry(learning_XGB, textvariable=xgb_alpha)
xgb_alpha_e.grid(row=7, column=1, padx=20, pady=8, sticky='W')

xgb_tree_method_lbl = ttk.Label(learning_XGB, text="Tree method: ")
xgb_tree_method_lbl.grid(row=8, column=0, padx=20, pady=8, sticky='W')
xgb_tree_method_cb = ttk.Combobox(learning_XGB, textvariable=xgb_tree_method, 
                        values=('auto', 'exact', 'approx', 'hist'), state="readonly")
xgb_tree_method_cb.current(0)
xgb_tree_method_cb.grid(row=8, column=1, padx=20, pady=8, sticky='W')

xgb_max_leaves_lbl = ttk.Label(learning_XGB, text="Maximum leaf: ")
xgb_max_leaves_lbl.grid(row=9, column=0, padx=20, pady=8, sticky='W')
xgb_max_leaves_e = ttk.Entry(learning_XGB, textvariable=xgb_max_leaves)
xgb_max_leaves_e.grid(row=9, column=1, padx=20, pady=8, sticky='W')

xgb_max_bin_lbl = ttk.Label(learning_XGB, text="Maximum bins: ")
xgb_max_bin_lbl.grid(row=10, column=0, padx=20, pady=8, sticky='W')
xgb_max_bin_e = ttk.Entry(learning_XGB, textvariable=xgb_max_bin)
xgb_max_bin_e.grid(row=10, column=1, padx=20, pady=8, sticky='W')

# Support Vector Machines
learning_SVM= Frame(learning,  relief="raised", borderwidth = 3) 
learning_SVM.place(x = 205, y = 40, width=420, height=420)

svm_C = tk.StringVar() 
svm_C.set(1.0)
svm_kernel = tk.StringVar()
svm_degree = tk.IntVar()
svm_degree.set(3)
svm_gamma = tk.StringVar() 

svm_C_lbl = ttk.Label(learning_SVM, text="Regularization parameter: ")
svm_C_lbl.grid(row=0, column=0, padx=20, pady=20, sticky='W')
svm_C_e = ttk.Entry(learning_SVM, textvariable=svm_C)
svm_C_e.grid(row=0, column=1, padx=20, pady=20, sticky='W')

svm_kernel_lbl = ttk.Label(learning_SVM, text="Kernel: ")
svm_kernel_lbl.grid(row=1, column=0, padx=20, pady=20, sticky='W')
svm_kernel_cb = ttk.Combobox(learning_SVM, textvariable=svm_kernel, 
                        values=("linear", "poly", "rbf","sigmoid"), state="readonly")
svm_kernel_cb.current(2)
svm_kernel_cb.grid(row=1, column=1, padx=20, pady=20, sticky='W')

svm_degree_lbl = ttk.Label(learning_SVM, text="Degree: ")
svm_degree_lbl.grid(row=2, column=0, padx=20, pady=20, sticky='W')
svm_degree_e = ttk.Entry(learning_SVM, textvariable=svm_degree)
svm_degree_e.grid(row=2, column=1, padx=20, pady=20, sticky='W')

svm_gamma_lbl = ttk.Label(learning_SVM, text="Gamma: ")
svm_gamma_lbl.grid(row=3, column=0, padx=20, pady=20, sticky='W')
svm_gamma_cb = ttk.Combobox(learning_SVM, textvariable=svm_gamma, 
                        values=("scale", "auto"), state="normal")
svm_gamma_cb.current(0)
svm_gamma_cb.grid(row=3, column=1, padx=20, pady=20, sticky='W')

lf_ML = tk.LabelFrame(learning, text="Method", font=label_font)
lf_ML.place(x = 5, y = 35, width=185, height=385)

rb_logReg= tk.Radiobutton( lf_ML, text="Logistic Regression",
                            wraplength=165, variable=mlMethod, value="LogReg", 
                            command=mlSelect)
rb_logReg.grid(row=0, column=0, padx=5, pady=5, sticky='W' )

rb_SGD = tk.Radiobutton( lf_ML, text="Stochastic Gradient Descent",
                            wraplength=165, variable=mlMethod, value="SGD", 
                            command=mlSelect)
rb_SGD.grid(row=1, column=0, padx=5, pady=5, sticky='W' )

rb_NB = tk.Radiobutton( lf_ML, text="Naive Bayes",
                            wraplength=165, variable=mlMethod, value="NB", 
                            command=mlSelect)
rb_NB.grid(row=2, column=0, padx=5, pady=5, sticky='W')

rb_kNN = tk.Radiobutton( lf_ML, text="k Nearest Neighbor",
                            wraplength=165, variable=mlMethod, value="kNN", 
                            command=mlSelect)
rb_kNN.grid(row=3, column=0, padx=5, pady=5, sticky='W' )

rb_DT = tk.Radiobutton( lf_ML, text="Decision Tree",
                            wraplength=165, variable=mlMethod, value="DT", 
                            command=mlSelect)
rb_DT.grid(row=4, column=0, padx=5, pady=5, sticky='W' )

rb_RF = tk.Radiobutton( lf_ML, text="Random Forest",
                            wraplength=165, variable=mlMethod, value="RF", 
                            command=mlSelect)
rb_RF.grid(row=5, column=0, padx=5, pady=5, sticky='W' )

rb_AdaBoost = tk.Radiobutton( lf_ML, text="AdaBoost",
                            wraplength=165, variable=mlMethod, value="AdaBoost", 
                            command=mlSelect)
rb_AdaBoost.grid(row=6, column=0, padx=5, pady=5, sticky='W' )

rb_GB = tk.Radiobutton( lf_ML, text="Gradient Boosting",
                            wraplength=165, variable=mlMethod, value="GB", 
                            command=mlSelect)
rb_GB.grid(row=7, column=0, padx=5, pady=5, sticky='W' )

rb_XGB = tk.Radiobutton( lf_ML, text="XGBoost",
                            wraplength=165, variable=mlMethod, value="XGB", 
                            command=mlSelect)
rb_XGB.grid(row=8, column=0, padx=5, pady=5, sticky='W' )

rb_SVM = tk.Radiobutton( lf_ML, text="Support Vector Machines",
                            wraplength=165, variable=mlMethod, value="SVM", 
                            command=mlSelect)
rb_SVM.grid(row=9, column=0, padx=5, pady=5, sticky='W' )

btn_apply_ml = tk.Button(learning, text="Apply", command=applyML)
btn_apply_ml.place(x = 5, y = 430, width=185, height=30)

############################################################################
# Evaluate tab

# Frame for confusion matrix
pm_cm_frame = tk.Frame(perfMetric,bg='#d9d9d9',width=270,height=180)
pm_cm_frame.place(x = 5, y = 5,width=270,height=180)

# Frame for ROC-Curve
pm_roc_frame = tk.Frame(perfMetric,bg='#d9d9d9',width=620,height=300)
pm_roc_frame.place(x = 5, y = 190,width=620,height=300)


pm_accuracy = tk.StringVar() 
pm_accuracy.set("0.0000")
pm_precision= tk.StringVar() 
pm_precision.set("0.0000")
pm_recall = tk.StringVar() 
pm_recall.set("0.0000")
pm_specificity = tk.StringVar() 
pm_specificity.set("0.0000")
pm_f1_score = tk.StringVar() 
pm_f1_score.set("0.0000")
pm_g_mean = tk.StringVar() 
pm_g_mean.set("0.0000")
pm_roc_auc = tk.StringVar() 
pm_roc_auc.set("0.0000")

accuracy_lbl = ttk.Label(perfMetric, text="Accuracy: ")
accuracy_lbl.place(x = 290, y = 10,width=60,height=30)
accuracy_e = ttk.Entry(perfMetric, textvariable=pm_accuracy)
accuracy_e.place(x = 355, y = 10,width=60,height=30)

precision_lbl = ttk.Label(perfMetric, text="Precision: ")
precision_lbl.place(x = 290, y = 45,width=60,height=30)
precision_e = ttk.Entry(perfMetric, textvariable=pm_precision)
precision_e.place(x = 355, y = 45,width=60,height=30)

recall_lbl = ttk.Label(perfMetric, text="Recall: ")
recall_lbl.place(x = 290, y = 80,width=60,height=30)
recall_e = ttk.Entry(perfMetric, textvariable=pm_recall)
recall_e.place(x = 355, y = 80,width=60,height=30)

specificity_lbl = ttk.Label(perfMetric, text="Specificity: ")
specificity_lbl.place(x = 290, y = 115,width=60,height=30)
specificity_e = ttk.Entry(perfMetric, textvariable=pm_specificity)
specificity_e.place(x = 355, y = 115,width=60,height=30)

f1_score_lbl = ttk.Label(perfMetric, text="F1-score: ")
f1_score_lbl.place(x = 450, y = 10,width=60,height=30)
f1_score_e = ttk.Entry(perfMetric, textvariable=pm_f1_score)
f1_score_e.place(x = 515, y = 10,width=60,height=30)

g_mean_lbl = ttk.Label(perfMetric, text="G-Mean: ")
g_mean_lbl.place(x = 450, y = 45,width=60,height=30)
g_mean_e = ttk.Entry(perfMetric, textvariable=pm_g_mean)
g_mean_e.place(x = 515, y = 45,width=60,height=30)

roc_auc_lbl = ttk.Label(perfMetric, text="ROC-AUC: ")
roc_auc_lbl.place(x = 450, y = 80,width=60,height=30)
roc_auc_e = ttk.Entry(perfMetric, textvariable=pm_roc_auc)
roc_auc_e.place(x = 515, y = 80,width=60,height=30)

############################################################################
# XML tab
# lime
expML_lime= Frame(expML, relief="raised", borderwidth = 3) 
expML_lime.place(x = 5, y = 40, width=630, height=430)

lf_lime = tk.LabelFrame(expML_lime, font=label_font)
lf_lime.place(x = 5, y = 5, width=130, height=415)

lime_n_sample = tk.IntVar()
lime_n_sample.set(0)
lime_discretizer = tk.StringVar()
lime_k_width = tk.StringVar() 
lime_d_metric = tk.StringVar()
lime_n_featrues = tk.IntVar()
lime_n_featrues.set(10)
lime_n_samples = tk.IntVar()
lime_n_samples.set(5000)


lime_n_sample_lbl = ttk.Label(lf_lime, text="Sample number")
lime_n_sample_lbl.place(x = 5, y = 5, width=110, height=20)
lime_n_sample_e = ttk.Entry(lf_lime, textvariable=lime_n_sample)
lime_n_sample_e.place(x = 5, y = 25, width=110, height=20)

lime_discretizer_lbl = ttk.Label(lf_lime, text="Discretizer")
lime_discretizer_lbl.place(x = 5, y = 55, width=110, height=20)
lime_discretizer_cb = ttk.Combobox(lf_lime, textvariable=lime_discretizer, 
                        values=("quartile","decile","entropy"),state="readonly")
lime_discretizer_cb.current(0)
lime_discretizer_cb.place(x = 5, y = 75, width=110, height=20)

lime_k_width_lbl = ttk.Label(lf_lime, text="Kernel width")
lime_k_width_lbl.place(x = 5, y = 105, width=110, height=20)
lime_k_width_e = ttk.Entry(lf_lime, textvariable=lime_k_width)
lime_k_width_e.place(x = 5, y = 125, width=110, height=20)

lime_d_metric_lbl = ttk.Label(lf_lime, text="Distance metric")
lime_d_metric_lbl.place(x = 5, y = 155, width=110, height=20)
lime_d_metric_cb = ttk.Combobox(lf_lime, textvariable=lime_d_metric, 
                        values=("euclidean", "manhattan","chebyshev","minkowski",
                                "seuclidean"), state="readonly")
lime_d_metric_cb.current(0)
lime_d_metric_cb.place(x = 5, y = 175, width=110, height=20)

lime_n_featrues_lbl = ttk.Label(lf_lime, text="Number of features")
lime_n_featrues_lbl.place(x = 5, y = 205, width=110, height=20)
lime_n_featrues_e = ttk.Entry(lf_lime, textvariable=lime_n_featrues)
lime_n_featrues_e.place(x = 5, y = 225, width=110, height=20)

lime_n_samples_lbl = ttk.Label(lf_lime, text="Number of samples")
lime_n_samples_lbl.place(x = 5, y = 255, width=110, height=20)
lime_n_samples_e = ttk.Entry(lf_lime, textvariable=lime_n_samples)
lime_n_samples_e.place(x = 5, y = 275, width=110, height=20)

btn_apply_lime = tk.Button(lf_lime, text="Apply Lime", command=applyLime)
btn_apply_lime.place(x = 5, y = 305, width=110, height=30)

# shap
shap_s_number = tk.IntVar()
shap_s_number.set(0)
shap_l_max_disp = tk.IntVar()
shap_l_max_disp.set(10)
shap_g_max_disp = tk.IntVar()
shap_g_max_disp.set(10)

expML_shap= Frame(expML, relief="raised", borderwidth = 3) 
expML_shap.place(x = 5, y = 40, width=630, height=430)

lf_shap= tk.LabelFrame(expML_shap, font=label_font)
lf_shap.place(x = 5, y = 5, width=130, height=415)


shap_plot= Frame(expML_shap, relief="raised", borderwidth = 3) 
shap_plot.place(x = 140, y = 0, width=485, height=425)

btn_apply_shap = tk.Button(lf_shap, text="Apply Shap", command=applyShap)
btn_apply_shap.place(x = 5, y = 10, width=115, height=30)

local_shap_rb = tk.Radiobutton( lf_shap, text="Local",
                            wraplength=110, variable=shap_L_G, value="Local", 
                            command=shapSelect)
local_shap_rb.place(x = 5, y = 45, width=60, height=30 )

global_shap_rb = tk.Radiobutton( lf_shap, text="Global",
                            wraplength=110, variable=shap_L_G, value="Global", 
                            command=shapSelect)
global_shap_rb.place(x = 7, y = 75,width=60,height=30 )

local_shap= Frame(lf_shap, relief="raised", borderwidth = 3) 
local_shap.place(x = 1, y = 105, width=123, height=303)

shap_s_number_lbl = ttk.Label(local_shap, text="Sample number")
shap_s_number_lbl.place(x = 5, y = 5, width=110, height=25)
shap_s_number_e = ttk.Entry(local_shap, textvariable=shap_s_number)
shap_s_number_e.place(x = 5, y = 35, width=110, height=25)

shap_l_max_disp_lbl = ttk.Label(local_shap, text="Max Features")
shap_l_max_disp_lbl.place(x = 5, y = 65, width=110, height=25)
shap_l_max_disp_e = ttk.Entry(local_shap, textvariable=shap_l_max_disp)
shap_l_max_disp_e.place(x = 5, y = 90, width=110, height=25)

shap_l_bar_rb = tk.Radiobutton( local_shap, text="Bar plot",
                            wraplength=110, variable=shapPlot, value="l_bar_plot")
shap_l_bar_rb.place(x = 5, y = 120, width=60, height=30 )

shap_l_decision_rb = tk.Radiobutton( local_shap, text="Decision plot",
                            wraplength=110, variable=shapPlot, value="l_decision_plot")
shap_l_decision_rb.place(x = 5, y = 150, width=88, height=30 )

shap_l_force_rb = tk.Radiobutton( local_shap, text="Force plot",
                            wraplength=110, variable=shapPlot, value="l_force_plot")
shap_l_force_rb.place(x = 5, y = 180, width=73, height=30 )

shap_l_waterfall_rb = tk.Radiobutton( local_shap, text="Waterfall plot",
                            wraplength=110, variable=shapPlot, value="l_waterfall_plot")
shap_l_waterfall_rb.place(x = 5, y = 210, width=90, height=30 )

btn_l_disp_shap= tk.Button(local_shap, text="Display SHAP", command=displayShap)
btn_l_disp_shap.place(x = 5, y = 255, width=110, height=30)


global_shap= Frame(lf_shap, relief="raised", borderwidth = 3) 
global_shap.place(x = 1, y = 105, width=123, height=303)

shap_g_max_disp_lbl = ttk.Label(global_shap, text="Max Features")
shap_g_max_disp_lbl.place(x = 5, y = 10, width=110, height=30)
shap_g_max_disp_e = ttk.Entry(global_shap, textvariable=shap_g_max_disp)
shap_g_max_disp_e.place(x = 5, y = 40, width=110, height=30)

shap_g_bar_rb = tk.Radiobutton( global_shap, text="Bar plot",
                            wraplength=110, variable=shapPlot, value="g_bar_plot")
shap_g_bar_rb.place(x = 5, y = 70, width=60, height=30 )

shap_g_beeswarm_rb = tk.Radiobutton( global_shap, text="Beeswarm plot",
                            wraplength=110, variable=shapPlot, value="g_beeswarm_plot")
shap_g_beeswarm_rb.place(x = 5, y = 100, width=98, height=30 )

shap_g_decision_rb = tk.Radiobutton( global_shap, text="Decision plot",
                            wraplength=110, variable=shapPlot, value="g_decision_plot")
shap_g_decision_rb.place(x = 5, y = 130, width=88, height=30 )

shap_g_heatmap_rb = tk.Radiobutton( global_shap, text="Heatmap plot",
                            wraplength=110, variable=shapPlot, value="g_heatmap_plot")
shap_g_heatmap_rb.place(x = 5, y = 160, width=93, height=30 )

shap_g_violin_rb = tk.Radiobutton( global_shap, text="Violin plot",
                            wraplength=110, variable=shapPlot, value="g_violin_plot")
shap_g_violin_rb.place(x = 5, y = 190, width=74, height=30 )

shap_g_summary_rb = tk.Radiobutton( global_shap, text="Summary plot",
                            wraplength=110, variable=shapPlot, value="g_summary_plot")
shap_g_summary_rb.place(x = 5, y = 220, width=95, height=30 )

btn_g_disp_shap= tk.Button(global_shap, text="Display SHAP", command=displayShap)
btn_g_disp_shap.place(x = 5, y = 255, width=110, height=30)

# anchors
expML_anchors= Frame(expML, relief="raised", borderwidth = 3) 
expML_anchors.place(x = 5, y = 40, width=630, height=430)


lf_anchors= tk.LabelFrame(expML_anchors, font=label_font)
lf_anchors.place(x = 5, y = 5, width=130, height=415)

anchors_plot= Frame(expML_anchors, relief="raised", borderwidth = 3) 
anchors_plot.place(x = 140, y = 5, width=480, height=415)

anchors_n_sample = tk.IntVar()
anchors_n_sample.set(0)
anchors_threshold = tk.StringVar()
anchors_threshold.set(0.95)
anchors_delta = tk.StringVar()
anchors_delta.set(0.1)
anchors_tau = tk.StringVar()
anchors_tau.set(0.15)
anchors_batch_size = tk.IntVar()
anchors_batch_size.set(100)

anchors_n_sample_lbl = ttk.Label(lf_anchors, text="Sample number ")
anchors_n_sample_lbl.place(x = 5, y = 5, width=110, height=20)
anchors_n_sample_e = ttk.Entry(lf_anchors, textvariable=anchors_n_sample)
anchors_n_sample_e.place(x = 5, y = 25, width=110, height=20)

anchors_threshold_lbl = ttk.Label(lf_anchors, text="Threshold ")
anchors_threshold_lbl.place(x = 5, y = 55, width=110, height=20)
anchors_threshold_e = ttk.Entry(lf_anchors, textvariable=anchors_threshold)
anchors_threshold_e.place(x = 5, y = 75, width=110, height=20)

anchors_delta_lbl = ttk.Label(lf_anchors, text="Delta ")
anchors_delta_lbl.place(x = 5, y = 105, width=110, height=20)
anchors_delta_e = ttk.Entry(lf_anchors, textvariable=anchors_delta)
anchors_delta_e.place(x = 5, y = 125, width=110, height=20)

anchors_tau_lbl = ttk.Label(lf_anchors, text="Tau ")
anchors_tau_lbl.place(x = 5, y = 155, width=110, height=20)
anchors_tau_e = ttk.Entry(lf_anchors, textvariable=anchors_tau)
anchors_tau_e.place(x = 5, y = 175, width=110, height=20)

anchors_batch_size_lbl = ttk.Label(lf_anchors, text="Batch size ")
anchors_batch_size_lbl.place(x = 5, y = 205, width=110, height=20)
anchors_batch_size_e = ttk.Entry(lf_anchors, textvariable=anchors_batch_size)
anchors_batch_size_e.place(x = 5, y = 225, width=110, height=20)

btn_apply_anchors = tk.Button(lf_anchors, text="Apply Anchors", command=applyAnchors)
btn_apply_anchors.place(x = 5, y = 255, width=110, height=30)

lf_XML = tk.LabelFrame(expML, font=label_font)
lf_XML.place(x = 85, y = 5, width=430, height=30)

rb_lime= tk.Radiobutton( lf_XML, text="LIME",
                            wraplength=165, variable=xmlMethod, value="Lime", 
                            command=xmlSelect)
rb_lime.grid(row=0, column=0, padx=25, pady=0, sticky='W' )

rb_shap = tk.Radiobutton( lf_XML, text="SHAP",
                            wraplength=165, variable=xmlMethod, value="Shap", 
                            command=xmlSelect)
rb_shap.grid(row=0, column=1, padx=45, pady=00, sticky='W' )

rb_anchors = tk.Radiobutton( lf_XML, text="ANCHORS",
                            wraplength=165, variable=xmlMethod, value="Anchors", 
                            command=xmlSelect)
rb_anchors.grid(row=0, column=2, padx=45, pady=00, sticky='W' )

############################################################################
# Credits tab
img = PhotoImage(file = "logo.png")
urls = ["https://pyxml.readthedocs.io/en/latest/",
        "https://github.com/alidegirmenci/PyXML",
        "https://www.sciencedirect.com/journal/softwarex"]

font1 = font.Font(family="Segoe UI", weight="bold", size=10)
font2= font.Font(family="Segoe UI", size=9, underline = True)
font3= font.Font(family="Segoe UI", size=10,underline = True)
default_font = ("Helvetica", 12)
img_lbl = ttk.Label(info, image= img)
img_lbl.place(x = 10, y = 13,width=310,height=451)
text = Text(info, height=360,width=285,background='#F0F0F0', relief="raised", 
            borderwidth = 5)
text.place(x = 340, y = 50, height=380, width=285)
text.insert(tk.END, "\n{0:^34}\n".format("Python"))
text.insert(tk.END, "{0:^34}\n\n\n".format("eXplainable Machine Learning"))
text.insert(tk.END, " {0:^34}\n".format("Contact"))
text.insert(tk.END, "{0:^34}\n".format("Ph.D. Ali Degirmenci"))
text.insert(tk.END, "   {0:^34}\n\n\n".format("alidegirmenci@aybu.edu.tr"))
text.insert(tk.END, "{0:^34}\n".format("Help"))
text.insert(tk.END, "{0:^39}\n\n\n".format(urls[0]))
text.tag_add("hyperlink", '12.0', '12.39')
text.tag_config("hyperlink", foreground="blue", underline=True)
text.tag_bind("hyperlink", "<Button-1>", lambda event, url=urls[0]: open_url(urls[0]))
text.insert(tk.END, " {0:^34}\n".format("Source code"))
text.insert(tk.END, "{0:^34}\n\n\n".format(urls[1]))
text.tag_add("hyperlink1", '16.0', '17.7')
text.tag_config("hyperlink1", foreground="blue", underline=True)
text.tag_bind("hyperlink1", "<Button-1>", lambda event, url=urls[1]: open_url(urls[1]))
text.insert(tk.END, "  {0:^34}\n".format("Article link"))
text.insert(tk.END, "{0:^34}\n\n\n".format(urls[2]))
text.tag_add("hyperlink2", '20.0', '21.13')
text.tag_config("hyperlink2", foreground="blue", underline=True)
text.tag_bind("hyperlink2", "<Button-1>", lambda event, url=urls[2]: open_url(urls[2]))

text.tag_add('b_1', '2.14','2.16')
text.tag_add('b_1', '3.4','3.5')
text.tag_add('b_1', '3.15','3.16')
text.tag_add('b_1', '3.23','3.24')
text.tag_config('b_1', font= font1)
text.tag_add('b_2', '6.14','6.21')
text.tag_add('b_2', '11.15','11.19')
text.tag_add('b_2', '15.12','15.23')
text.tag_add('b_2', '19.13','19.25')
text.tag_config('b_2', font= font2)
text.tag_add('b_3', '8.7','8.32')
text.tag_add('b_3', '16.7','16.32')
text.tag_config('b_3', foreground = 'blue', font= font3)

root.mainloop()