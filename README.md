<p align="center"><img src="https://github.com/alidegirmenci/PyXML/blob/master/docs/images/pyxml_logo_main.png" alt="logo" height="300"/></p>

<h3 align="center">
<p> PyXML : Python eXplainable Machine Learning </h3>

![Python](https://img.shields.io/badge/Python-3.8-brightgreen) ![tkinter](https://img.shields.io/badge/tkinter-8.6-brightgreen) ![pandas](https://img.shields.io/badge/pandas-2.1.1-brightgreen) ![pandas](https://img.shields.io/badge/NumPy-1.24.3-brightgreen) ![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.2.2-brightgreen) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.3-blue) ![seaborn](https://img.shields.io/badge/seaborn-0.12.2-brightgreen) ![math](https://img.shields.io/badge/Scikit--learn-0.22-brightgreen) ![webbrowser](https://img.shields.io/badge/webbrowser-0.2.2-brightgreen)  ![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-brightgreen) ![LIME](https://img.shields.io/badge/LIME-0.2.0.1-brightgreen) ![SHAP](https://img.shields.io/badge/SHAP-0.43.0-brightgreen) ![ANCHOR](https://img.shields.io/badge/ANCHOR-0.0.2.0-brightgreen)  [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview
PyXML is a software tool developed on Python, it is a modular software that allows users to perform complete machine learning pipeline without requiring any additional tool. The main difference of PyXML software from its counterparts is that PyXML allows data loading, data imputation (completion of missing data), data scaling, fitting of different machine learning algorithms and hyperparameter tuning of these methods and interpretation of the results with explainable machine learning algorithms in a single GUI. PyXML, a Python GUI software, was developed to automate the machine learning application process and can be used by researchers, domain experts, and end users to explain the results of machine learning models and make decisions.  It is written with Python programming language and GUI is created with Tkinter framework. Tkinter is the Python's built-in GUI toolkit framework and it is an open-source. Since PyXML is written with Tkinter framework, it is cross-platform (operate on Windows, macOS, and Linux), do not require any programming experience, simple, and easy to modify (user's may make modification).

## Highlights

*	A comprehensive software solution for implementing the entire Machine Learning (ML) process.
*	Developed using Tkinter, Python's standard Graphical User Interface (GUI) development framework.
*	Designed with contemporary software engineering principles in mind. 
*	Facilitates the utilization of widely adopted ML techniques and allows for the tuning of method-specific hyperparameters.
*	Tailored to support state-of-the-art eXplainable Machine Learning (XML) methods.

This repository includes all the source code and associated resources for the PyXML software.

## Features

* **Data loading**: Allows users to load and see the content of the data set.
* **Preprocessing**: Enables users to prepare the dataset by applying preprocessing operations such as data imputation, training/test split, and data scaling to the uploaded dataset.
* **Dataset Property**: Provides visual and statistical information about the dataset.
* **Training**: Allows machine learning methods to be applied in combination with hyperparameter tuning.
* **Evaluate**: Displays the results of the created machine learning model.
* **XML**: Offers the generated model outputs to be explained with different XML methods.
* **Credits**: Presents information about the software.

## Requirements & Installation

**Step 1**: Download and install the most recent version of Python from the given [link](https://www.python.org/downloads/).

It is advisable to use Python version 3.8 or later. To verify the installation of Python3, open the terminal on Linux/MacOS or PowerShell on Windows, then execute the subsequent command:

```
python3 --version
```
Once Python is successfully installed, users need to install various modules necessary for running the PyXML software.
<br/>
**Step 2**: Install all the requirements before proceeding to next steps:
### Mandatory Dependencies
| **Software** | **Version** | **Official Installation Guide** |
| --- | --- | --- |
| [Python](https://www.python.org/) | >= 3.8 |  [link](https://www.python.org/downloads/) |
| [Tkinter](https://docs.python.org/3/library/tkinter.html) | >= 8.6 |  |
| [Pandas](https://pandas.pydata.org) | >= 2.1.1 | [Link](https://pandas.pydata.org/docs/getting_started/install.html#) |
| [NumPy](https://numpy.org)| >= 1.24.3 |[Link](https://numpy.org/install/)|
| [Scikit-learn](https://scikit-learn.org/stable/)| >= 1.2.2 |[Link](https://scikit-learn.org/stable/install.html)|
| [Matplotlib](https://matplotlib.org)| == 3.7.3 |[Link](https://matplotlib.org/stable/install/index.html)|
| [Seaborn](https://seaborn.pydata.org)| >= 0.12.2 |[Link](https://seaborn.pydata.org/installing.html)|
| [Math](https://docs.python.org/3/library/math.html)| >= 0.22 |[Link](https://pypi.org/project/python-math/)|
| [Webbrowser](https://docs.python.org/3/library/webbrowser.html)| >= 0.2.2 ||
| [XGBoost](https://xgboost.readthedocs.io/en/stable/python/index.html)| >= 1.7.6|[Link](https://xgboost.readthedocs.io/en/stable/install.html)|
| [LIME](https://github.com/marcotcr/lime)| >= 0.2.0.1 |[Link](https://pypi.org/project/lime/)|
| [SHAP](https://shap.readthedocs.io/en/latest/)| >= 0.43.0 |[Link](https://pypi.org/project/shap/)|
| [ANCHOR](https://github.com/marcotcr/anchor)| >= 0.0.2.0 |[Link](https://pypi.org/project/anchor/)|
<br/>

Make sure the packages are installed in the versions specified above. To check the version of a module in Python, you can use the **__version__** attribute of the module. The version of any Python module can be found by running the following code snippet. Replace **module_name** with the name of the module whose version you want to check.

```
import [packagename]
print(module_name.__version__)

```
<br/>

You should install all the python3 modules using the 
```
pip3 install [packagename]

```
or alternatively using
```
sudo apt-get install [packagename]
```
<br/>

**Step 3**: How to use PyXML?

* Download the source code either with "Open with GitHub Desktop" or "Download zip".
*	Once the .zip file is downloaded from the GitHub, extract the contents onto a desired folder.
*	Open Spyder IDE, navigate to that folder in Spyder IDE and run the "pyxml.py" and "logo.png" file which is in the extracted folders directory. Note that PyXML is only test with the Spyder IDE. It may also work with other IDEs, but it has been tested with only Spyder.

## Release Information

* **Author**:
> Ali Degirmenci, MSc, PhD Ankara Yildirim Beyazit University, Turkey.
> Email: alidegirmenci@aybu.edu.tr
> GitHub: https://github.com/alidegirmenci

* **Source Repository**:
> https://github.com/alidegirmenci/PyXML

* **Documentation**:
> https://pyxml.readthedocs.io/en/latest/

* **Citation**:
(not published yet)


## Feedback

You can contact me at the following e-mail: [alidegirmenci@aybu.edu.tr](alidegirmenci@aybu.edu.tr).
