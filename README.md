# tcFNN analysis code for Temmar et al 20XX

## 1. Introduction
The following code was used to produce all the figures used in 'Artificial neural network for brain-machine interface consistently produces more naturalistic finger movements than linear methods' by Temmar et al., currently on biorxiv (DOI: 10.1101/2024.03.01.583000, [click here](https://www.biorxiv.org/content/10.1101/2024.03.01.583000v1). In order to properly run this code, you will need the dataset (which we provide in a preprocessed form on DeepBlue, the University of Michigan's Data repository), along with a couple of extras. The repo has versions of the figures as they were last produced uploaded - The code outputs .pdf files, and minor aesthetic changes are then applied in Adobe Illustrator.

## 2. Setup
### 1. Packages required
matplotlib, seaborn, pandas, numpy, scipy, pytorch

### 2. Model Output
After cloning the repo, please create and extra folders at the top level (in the same location as Analyses, Results, utils) called 'Models'. Inside of 'Models' create four folders 'context_offline', 'fits_offline', 'fits_online' and 'variance_offline'. 

### 3. Getting the data
Download the dataset from DeepBlue here: [[https://deepblue.lib.umich.edu/data/concern/data_sets/bg257g01k]]
Move Data.zip into the repo, and extract it into a folder titled 'Data' in this location. The repo should now have a Data folder along with Analyses, Results, and utils. This data is already preprocessed, so leave the preprocessing flags in the main script as False, or the code will not run (as there is no access to the raw data)

## 3. Running the Code
To run the script, run `generateFiguresAndAnalyses.py` in your preferred terminal or IDE. In this script, each section/figure can be turned off by setting the run_section flag above it to False. Additionally, since no models exist yet, please turn the train_rr and train_nn flags in each section to True, and keep the preprocess flag to False for all sections.

