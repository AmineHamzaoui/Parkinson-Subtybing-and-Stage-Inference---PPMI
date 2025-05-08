To get SuStaIn up and running first you need to install the package. I'm using Anaconda and had some conflicts with existing packagaes so I had to create a new environment. For me the whole set up process looked like this...
                                
Step 1: Open up a terminal window and create a new environment "sustain_env" in anaconda that uses python 3.7 and activate the environment ready to install pySuStaIn.
```console
conda create --name sustain_tutorial_env python=3.7
conda activate sustain_tutorial_env
```

You need to install conda, there was a packaging problem and its solved my doing these commands :
```
conda install -n base conda-libmamba-solver                                                
conda install -y ipython jupyter matplotlib statsmodels numpy pandas scipy seaborn pip --solver=libmamba  
conda config --set solver libmamba
conda update -n base -c defaults conda
```   
The problem was not being able to install or update conda-libmamba-solver and other must-matching libraries in the base which is the root Conda environment.
You just need to create your own environment instead of modifying the base.
Resources :
```
https://stackoverflow.com/questions/74781771/how-we-can-resolve-solving-environment-failed-with-initial-frozen-solve-retry
```

Step 2: Use the terminal to install necessary packages for running the notebook and pySuStaIn within the environment.
```console
conda install -y ipython jupyter matplotlib statsmodels numpy pandas scipy seaborn pip
pip install git+https://github.com/ucl-pond/pySuStaIn
```

You need to ensure the version of anaconda is matched with the version of python you have.


Step 3: Use the terminal to run the notebook from inside the environment.
```console
jupyter notebook
```

Once you've got your environment running the general workflow will be to open a terminal window and navigate to the directory with the notebook in, activate the envirnoment, open a jupyter notebook inside and use the notebook to run your analyses, then use the terminal deactivate the environment once you've finished running analyses.
```console
conda activate sustain_tutorial_env
jupyter notebook
conda deactivate
```
After that you need to understand Alexandra Young implementation of the model to be able to use it : it's really intresting pure numpy, bayesian equations and bunch of indexing and broadcasting!
