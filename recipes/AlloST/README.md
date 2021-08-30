# AlloST
This folder contains recipes for AlloST 1 and 2

### How to run
0- Install extra dependencies
```
pip install -r extra_requirements.txt
```

1- Train a tokenizer. The tokenizer takes in input the training translations and determines the subword units that will be used for the ST task, the auxiliary MT task.

```
cd Tokenizer
python train.py hparams/train_bpe_1k.yaml
```

2- Train a Language model. The Language model takes the phone sequences as the input and produces the arpa language model.
```
cd LM
python train.py hparams/arpa.yaml
```

3- Train the speech translator
```
cd ST/transformer
python train.py hparams/transformer.yaml
```

# Performance summary
Results are reported in terms of sacrebleu.

| hyperparams file | dev   | dev2   | test   | ctc_weight | asr_weight | mt_weight | Model | GPUs               |
|:----------------:|:-----:| :-----:| :-----:| :--------: | :--------: | :-------: | :-------: | :----------------: |
| transformer.yaml | 40.67 | 41.51  | 40.30  | 0          | 0          | 0         | Not Avail. | 2xRTX 2080 Ti 11GB |
| transformer.yaml | 47.50 | 48.33  | 47.31  | 1          | 0.3        | 0         | [Model](https://drive.google.com/drive/folders/1wd4iWuFimZBanBDeZSPFjxM1m4LovXdb?usp=sharing) | 2xRTX 2080 Ti 11GB |
| transformer.yaml | 46.10 | 46.56  | 46.79  | 1          | 0.2        | 0.2       | Not Avail. | 2xRTX 2080 Ti 11GB |
| conformer.yaml   | 46.37 | 47.07  | 46.10  | 0          | 0          | 0         | Not Avail. | 2xRTX 2080 Ti 11GB |
| conformer.yaml   | 48.09 | 48.19  | 48.04  | 1          | 0.3        | 0         | [Model](https://drive.google.com/drive/folders/1hlMOy1yutwkcXgKIW7tMa5WEe1ixhLaU?usp=sharing) | 1xTesla A100 (works with 2xRTX 2080 Ti) |


# **Citing ALLOST**
Please, cite  if you use it for your research or business.