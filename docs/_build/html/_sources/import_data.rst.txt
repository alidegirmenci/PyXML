Import Data 
===========

When the "pyxml.py" file is run on the Spyder IDE, the GUI shown in Figure 1 
opens.  The "Import Data" tab is the default tab of PyXML. "Browse" button 
allows selecting the dataset from the computer's memory.

.. _fig1:

.. figure:: images/figure_1.png
   :alt: GUI opening winodw

   **Figure 1:** GUI's default opening window

Clicking on the "Browse" button opens a file dialog allowing to navigate through 
the computer's memory and select the data set to load, as shown in Figure 2. 
Loading a sample data set from the computer is shown in Figure 2. Note: Please 
note that PyXML only supports “.csv” file formats; any other file formats are 
ignored. However, a few lines of code in the source code can be modified to 
allow different file types to be opened, such as xlsx. PyXML assumes that the 
dataset is organized in a numeric data type, with attributes at the beginning 
and the class label at the end.

.. _fig2:

.. figure:: images/figure_2.png
   :alt: load data

   **Figure 2:** Selecting and loading the dataset from the memory

Upon clinking “Open” in the file dialog loaded dataset is displayed in the 
"Import data" tab, as shown in Figure 3.  The data path of the loaded dataset 
can be seen next to the "Load Input Data" label. Additionally, the horizontal 
scroll bar can be used to scroll left/right and the vertical scroll bar can be 
used to scroll up and down to display features and instances that are not 
visible due to the size of the dataset.

.. _fig3:

.. figure:: images/figure_3.png
   :alt: Successful loading data

   **Figure 3:** Successful loading of the data set into the GUI

.. Here is a paragraph discussing some important points. As shown in Figure 1, the data trends are clearly visible. In Figure :ref:`fig2`, we observe different trends.

.. .. _fig1:

.. .. figure:: images/figure_1.png
..    :alt: Sample Figure 1

..    **Figure 1:** This is a caption for the first figure, explaining what it shows.

.. .. _fig2:

.. .. figure:: images/figure_2.png
..    :alt: Sample Figure 2

..    **Figure 2:** This is a caption for the second figure, explaining what it shows.

.. Further discussion can be placed here, continuing the text.