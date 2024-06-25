# Program Guide - AMS Test Kit Read Out and Evaluation

## General information
These Python scripts allow for reading out the .csv data files of measurements with the AMS OSRAM evaluation readout 
circuit. Note that this readout depends on a few naming conventions, so with varied file naming modification become necessary!
Same for the program structure - due to folder relative script links it's necessary to keep the folder structure of Python code.

The program structure is organized as follows: In the file read_MATRIX are the core functions for reading out 
measurements and converting them into plots. These functions may import functions from some other scripts in the same 
folder, e.g. the Otsu's algorithm from DataAnalysis.py.

All files with Console_ are used as application of the methods. Here the plots for the analysis of the first Cyrcè 
measurements are generated.

## Quick start guide
1. Import the folder with the Python code via git
2. Load all necessary packages from the requirements.txt (it is recommended using a virtual environment e.g. conda 
rather than working with your global Python installation)
3. Check for paths in the Python files: These should only appear in the consoles! Rename them with the paths on your PC,  
depending where you saved the measurement data, normalization files and where you want so save the plots to. (Copy the 
paths from your system and use pathlibs Path("string of path") for avoiding inter-system trouble with "/" or "\". 
Especially if you add code with folder relative paths!)
4. Run the code - and it should work.
5. If not: Recheck the steps, use the error message for troubleshooting or ask me (nico.brosda@rub.de)

## Plotting
For plotting I use the self written wrap around some matplotlib methods in "Plot_Methods". The settings are chosen for 
English language.

Important remark: The plotting uses LaTeX to compile the axis labels, etc.. This allows for using LaTeX syntax in the 
plots, as long you use a raw-string around it (r''). If some error with LaTeX occurs, this might be linked to this. To 
use this plotting, you must have a local LaTeX distribution installed on the system executing the code. It is possible 
to turn the LaTeX compiling off via the boolean variable "use_LaTeX" at the begin of plot_standards.py.
Note that you are also able to change the standard save_format of plots directly above with the boolean "save_format".
This comes in handy when using plt.contourf because in pdf format the files become to big.