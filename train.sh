TRAINDATA="${HOME}/DATASETS/CLASSIF_BIGGEST_SQUARE"
VALDATA="${HOME}/DATASETS/CLASSIF_BIGGEST_SQUARE"
PATHSAVE="${HOME}/Deep-Clustering-Cloud-Images/model/deepcluster.model"
KMEANSAVE="${HOME}/Deep-Clustering-Cloud-Images/model/kmeans.model"
PCASAVE="${HOME}/Deep-Clustering-Cloud-Images/model/pca.model"
LR=0.05
EPOCHS=100
PYTHON="python"


$PYTHON train.py --train_dataset $TRAINDATA --val_dataset $VALDATA --path_save $PATHSAVE --pca_save $PCASAVE --kmean_save $KMEANSAVE --lr $LR --epochs $EPOCHS
