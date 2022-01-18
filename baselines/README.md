# SciGen Baselines

This repository contains the code for the baselines of the paper: "[SciGen: a Dataset for Reasoning-Aware Text Generation from Scientific Tables](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/149e9677a5989fd342ae44213df68868-Paper-round2.pdf)".



The baselines are taken from "[Investigating Pretrained Language Models for Graph-to-Text Generation](https://arxiv.org/pdf/2007.08426.pdf)", and they are implemented using the [HuggingFace](https://huggingface.co/) framework. Please, refer to their websites for further details on the installation and dependencies.

## Environments and Dependencies

- python 3.6
- transformers 2.10.0
- pytorch-lightning 0.8.1
- torch 1.5.1
- BLEURT 0.0.1
- bert-score 0.3.3
- sacrebleu 1.4.12
- moverscore 1.0.3


## Preprocessing

First, convert the json files of the SciGen dataset into the required format for BART and T5 models.

```
./convert_json_files.py -f <SciGen_json_folder> -s <corresponding_split_of_the_provided_json>
```

For instance, for converting the training split in the few-shot setting, run:
```
././convert_json_files.py -f ../dataset/train/few-shot/train.json -s train
```

## Finetuning

We use the following command for training the BART model:
```
python train_table2text_bart.py \
--data_dir=$DATA_DIRECTORY \
--model_name_or_path=bart-large \
--learning_rate=3e-5 \
--num_train_epochs 30 \
--train_batch_size=8 \
--eval_batch_size=4 \
--test_batch_size=4 \
--output_dir=$OUTPUT_DIR \
--n_gpu 1 \
--do_train \
--do_predict \
--early_stopping_patience 10 \
--max_source_length 384 \
--max_target_length 384 \

```


For training T5 models, we use the following command:
```
python train_table2text_t5.py \
--data_dir=$DATA_DIRECTORY \
--model_name_or_path=$MODEL_NAME \
--learning_rate=3e-5 \
--num_train_epochs 30 \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--test_batch_size=$BATCH_SIZE \
--output_dir=$OUTPUT_DIR \
--n_gpu 1 \
--do_train \
--do_predict \
--early_stopping_patience 10 \
--max_source_length 384 \
--max_target_length 384 \

```
We use MODEL_NAME=t5-base and BATCH_SIZE=8 for T5-base experiments, and MODEL_NAME=t5-large and BATCH_SIZE=2 for T5-large experiments.  

