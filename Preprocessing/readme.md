
# Preprocessing Instructions:

Before starting ensure that a python environment is created using the following packages:

setuptools numpy pandas random2 h5py argparse python-csv  rdkit-pypi xml-python glob2 tqdm openbabel
pybel mdms pillow jupyterlab torch-geometric torch-scatter torch-sparse

These packages will be used to run the preprocessing code.

After creating this python environment execute the following:
 
1. Copy the contents from the folder Roy/datamine-2022. Run the first 3 sections of prepare_data.ipynb on Anvil Jupyter notebook
2. Run the first 2 cells of section 4
3. Build a virtual singularity environment named "ingenii.sif" using the "scratch.def" file. Enter this environment using the command: "singularity shell ingenii.sif"
4. Run the third cell in the cluster by executing the command "sbatch myjobsubmissionfile.sh" in the terminal. Feel free to change the cluster parameters. This piece of code can also be run as is but will take a long time to complete.
5. Run the last two steps
