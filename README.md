# Unet-Multiples-Suppression

## 1. Build speed model synthesis record in 'syn' folder
```
cd syn
run Data_Modeling_syn1.ipynb
run Data_Modeling_syn2.ipynb
run Data_Modeling_SEAM.ipynb
# put result npz in 'dataset' folder
cd utils
# Split the data according seismic shot
python 3Dconvert2D.py

# Slice the data (256x256)
python Getpatches.py

# you can view the data in 'view' folder
cd view

# and finally data put in 'data' folder
cd data
# run for test load data successfully
python build_data.py
```

## 2. Modify parameters in train.py

```
vim train.py
# you can modify ..., finally train!
python train.py
```

## 3. Test data in 'test' folder
```
run jupyter notebook
```

