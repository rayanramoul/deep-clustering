# Deep-Clustering-Cloud-Images
Applying deep clustering learning for the unsupervised classification of cloud images.

## Setup the project :
```bash
python3 -m venv /path/to/new/virtual/environment
source environment/bin/activate
pip install -r requirements.txt
pip install jupyter
ipython kernel install --name "local-venv" --user
```

## Train :
Modify the script train.sh with the different parameters you want.
```bash
./train.sh
```

## Use the model :
The notebook compare.ipynb shows how to use the model for clustering and a resulting confusion matrix.
visualize.ipynb shows how to get examples of images for each cluster from a dataset.

