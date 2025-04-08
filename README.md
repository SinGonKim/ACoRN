# ACoRN
ACoRN: Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models


## Data Construction

```
python ./DataAugmentation/data_generation.py
python ./DataAugmentation/processing_documents.py
```

##
Otherwise, you can download constructed dataset by [here](https://drive.google.com/drive/folders/1DVvDaDNJRVeUTXnNEfnY0WM7WI-_QsHM?usp=sharing)


## Train
```
bash ./train/train.sh
```


## Inference
```
bash ./inference/inference.sh
```
