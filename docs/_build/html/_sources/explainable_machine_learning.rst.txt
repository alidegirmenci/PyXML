Explainable Machine Learning 
============================

The "Explainable Machine Learning" tab provides explanation of the machine 
learning model created in the "Training" tab. In this tab, different 
Explainable Machine Learning (XML) methods such as LIME, SHAP, and ANCHORS, 
which are widely adopted, can be used to describe the machine learning model. 
The XML methods in PyXML are post-hoc model-agnostic explanation methods, so 
they can be applied to any machine learning method. To activate one of these 
XML methods, users need to select the corresponding XML method via the radio 
buttons at the top of the window. The right side of the window is then changed, 
allowing the hyperparameters of the selected XML method to be set. Finally, the 
apply button at the end of each XML method gives the results of the explanation 
with the selected XML method. The graphical user interface also allows for 
method-specific hyperparameter adjustments in these methods. The default version 
of the "Explainable Machine Learning" tab is shown in Figure 42.

.. _fig42:

.. figure:: images/figure_42.png
   :alt: Explainable machine learning tab
   :align: center

   **Figure 42:** Explainable machine learning tab

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Explainable Machine Learning

   LIME <LIME>
   SHAP <SHAP>
   ANCHORS <ANCHORS>