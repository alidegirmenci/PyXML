.. PyXML documentation master file, created by
   sphinx-quickstart on Wed Jun 12 02:29:23 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyXML's documentation!
=================================

.. image:: images/pyxml_logo_main.png
   :width: 300px
   :align: center

.. raw:: html

    <br><br>


**PyXML (Python eXplainable Machine Learning)** is a software tool developed 
on Python, it is a modular software that allows users to perform complete machine learning pipeline without 
requiring any additional tool. The main difference of PyXML software 
from its counterparts is that PyXML allows data loading, data imputation 
(completion of missing data), data scaling, fitting of different Machine 
Learning (ML) algorithms and hyperparameter tuning of these methods and 
interpretation of the results with eXplainable Machine Learning (XML) algorithms 
in a single Graphical User Interface (GUI). PyXML, a Python GUI software, was developed to automate 
the ML application process and can be used by researchers, 
domain experts and end users to explain the results of machine learning 
models and make decisions.  It is written with Python programming language 
and GUI is created with Tkinter framework. Tkinter is the Python's built-in 
GUI toolkit framework and it is an open-source. Since PyXML is written with 
Tkinter framework, it is cross-platform (operate on Windows, macOS, and Linux), 
do not require any programming experience, simple and easy to modify (user's 
may make modification).

Highlights
-----------

* A comprehensive software solution for implementing the entire ML process.
* Developed using Tkinter, Python's standard GUI development framework.
* Designed with contemporary software engineering principles in mind. 
* Facilitates the utilization of widely adopted ML techniques and allows for the tuning of method-specific hyperparameters.
* Tailored to support state-of-the-art XML methods.

`GitHub repository <https://www.w3schools.com/cs/index.php>`_
includes all the source code and associated resources for the PyXML software.

Features
-----------

* **Data loading**: Allows users to load and see the content of the data set.
* **Preprocessing**: Enables users to prepare the dataset by applying preprocessing operations such as data imputation, training/test split and data scaling to the uploaded dataset.
* **Dataset Property**: Provides visual and statistical information about the dataset.
* **Training**: Allows machine learning methods to be applied in combination with hyperparameter tuning
* **Evaluate**: Displays the results of the created machine learning model.
* **XML**: Offers the generated model outputs to be explained with different XML methods.
* **Credits**: Presents information about the software.

Requirements & Installation
---------------------------

**Step 1**: Download and install the most recent version of Python from the 
given `link <https://www.python.org/downloads/>`_.

It is advisable to use Python version 3.8 or later. To verify the installation of Python3, open the terminal on Linux/MacOS or PowerShell on Windows, then execute the subsequent command:

.. code-block:: python

    python3 --version

Once Python is successfully installed, users need to install various 
modules necessary for running the PyXML software.

**Step 2**: Install all the requirements before proceeding to next steps:

.. list-table:: **Mandatory Dependencies**
   :widths: 30 30 50
   :header-rows: 1

   * - Software
     - Version
     - Official Installation Guide
   * - `Python <https://www.python.org/>`_
     - >= 3.8
     - `Link <https://www.python.org/downloads/>`_
   * - `Tkinter <https://docs.python.org/3/library/tkinter.html>`_
     - >= 8.6
     - 
   * - `NumPy <https://numpy.org>`_
     - >= 1.24.3
     - `Link <https://numpy.org/install/>`_
   * - `Scikit-learn <https://scikit-learn.org/stable/>`_
     - >= 1.2.2
     - `Link <https://scikit-learn.org/stable/install.html>`_
   * - `Matplotlib <https://matplotlib.org>`_
     - == 3.7.3
     - `Link <https://matplotlib.org/stable/install/index.html>`_
   * - `Seaborn <https://seaborn.pydata.org>`_
     - >= 0.12.2
     - `Link <https://seaborn.pydata.org/installing.html>`_
   * - `Math <https://docs.python.org/3/library/math.html>`_
     - >= 0.22
     - `Link <https://pypi.org/project/python-math/>`_
   * - `Webbrowser <https://docs.python.org/3/library/webbrowser.html>`_
     - >= 0.2.2
     - 
   * - `XGBoost <https://xgboost.readthedocs.io/en/stable/python/index.html>`_
     - >= 1.7.6
     - `Link <https://xgboost.readthedocs.io/en/stable/install.html>`_
   * - `LIME <https://github.com/marcotcr/lime>`_
     - >= 0.2.0.1
     - `Link <https://pypi.org/project/lime/>`_
   * - `SHAP <https://shap.readthedocs.io/en/latest/>`_
     - >= 0.43.0
     - `Link <https://pypi.org/project/shap/>`_
   * - `ANCHOR <https://github.com/marcotcr/anchor>`_
     - >= 0.0.2.0
     - `Link <https://pypi.org/project/anchor/>`_


Make sure the packages are installed in the versions specified above. To check the 
version of a module in Python, you can use the version attribute of the module. The 
version of any Python module can be found by running the following code snippet. 
Replace module_name with the name of the module whose version you want to check.


.. code-block:: python

    import [packagename]
    print(module_name.__version__)

You should install all the python3 modules using the 

.. code-block:: python

    pip3 install [packagename]

or alternatively using

.. code-block:: python

    sudo apt-get install [packagename]

**Step 3**: How to use PyXML?

* Download the source code either with "Open with GitHub Desktop" or "Download zip".
* Once the .zip file is downloaded from the GitHub, extract the contents onto a desired folder.
* Open Spyder IDE, navigate to that folder in Spyder IDE and run the "pyxml.py" and "logo.png" file which is in the extracted folders directory. Note that PyXML is only test with the Spyder IDE. It may also work with other IDEs, but it has been tested with only Spyder.

Release Information
--------------------

* **Author**:
    Ali Degirmenci, MSc, PhD Ankara Yildirim Beyazit University, Turkey. 

    Email: alidegirmenci@aybu.edu.tr  

    GitHub: https://github.com/alidegirmenci 

* **Source Repository**:
    https://github.com/alidegirmenci/PyXML

* **Documentation**:
    https://pyxml.readthedocs.io/en/latest/

* **Citation**:
    (not published yet)


Feedback
--------

You can contact me at the following e-mail: (alidegirmenci@aybu.edu.tr).

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Import Data

   API Import Data <import_data>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Preprocessing

   API Preprocessing <preprocessing>
   Data Imputation <data_imputation>
   Train/Test Split <train_test_split>
   Data Scaling <data_scaling>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Dataset Property

   API Dataset Property <dataset_property>
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Training

   API Training <training>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Evaluate

   API Evaluate <evaluate>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: XML 

   API XML <explainable_machine_learning>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Credits

   API Credits <credits>



.. toctree::
   :maxdepth: 2
   :caption: Contents: 

