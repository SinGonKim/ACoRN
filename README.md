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

##
You can download trained model

Natural Questions: [ACoRN_Flan-t5-large-nq](https://huggingface.co/Alan96/ACoRN_Flan-t5-large-nq)

TriviaQA: [ACoRN_Flan-t5-large-triviaqa](https://huggingface.co/Alan96/ACoRN_Flan-t5-large-triviaQA)

PopQA: [ACoRN_Flan-t5-large-popqa](https://huggingface.co/Alan96/ACoRN_Flan-t5-large-popQA)

## Inference
```
bash ./inference/inference.sh
```
