# Instructions

## Create a python environment using the following instructions

1. Instantiate a gpu machine and make sure you have conda for creating env<br>
$ module load anaconda

2. Create an env called `sgcnn`<br>
$ conda create -n 3dcnn python=3.9 ipykernel pandas h5py -y

3. activate that env<br>
$ conda activate sgcnn

4. Register ipython kernel for using in notebooks<br>
$ python -m ipykernel install --user --name sgcnn --display-name "Python (3dcnn)"

5. Install packages<br>
$ conda install os pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y<br>

## Run the Code

1. Create and enter a python environment named 3dcnn with all required packages
2. Run job.sh on the cluster. Feel free to change the cluster parameters
