# tf-listnet

This repository is for my [blog](https://medium.com/@kazk1018/tf-train-sequenceexample%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6listwise-learning-to-rank%E3%82%92%E3%82%84%E3%81%A3%E3%81%A6%E3%81%BF%E3%81%9F-fc5e3c9ec412).

## Requirements

  * python 3.7
  * pipenv

## Dataset

LETOR: Learning to Rank for Information Retrieval (Microsoft)

  * LETOR4.0/MQ2008
  * supervised ranking
  * Download from [link](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)
  * Dataset should be downloaded in `data/` at project root directory by default

  ```
  tf-listnet/
    data/
      MQ2008/
        Fold1/
        Fold2/
        ...
        S1.txt
        S2.txt
        ...
  ```

## Run

By default (only training model)

```
pipenv run datasets
pipenv run train
```

Training with evaluation on validation data

```
pipenv run datasets
pipenv run datasets --input data/MQ2008/Fold1/vali.txt --output data/validation.tfrecord
pipenv run train --validation data/validation.tfrecord
```
