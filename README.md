# Unet-Multiples-Suppression

## 1. Prepare for datasets
```
cd syn
run Data_Modeling_syn1.ipynb
run Data_Modeling_syn2.ipynb
run Data_Modeling_SEAM.ipynb


# modify config, such as network kind
cd configs
vim config.yaml


# Split the data according seismic shot
cd utils
python 1_3Dconvert2D.py


# Slice the data (256x256)
python 2_Getpatches.py


# you can view the data in 'view' folder
cd view


# and finally data put in 'data' folder
cd data


# run for test load data successfully
python build_data.py


```


## 2. training
```
python train.py
```


## 3. Testing
```
cd test
run jupyter notebook
```
