import os

import torch
from torch.utils.data import Dataset
import re
from transformers.tokenization_utils import BatchEncoding
import os
from tqdm import tqdm
import sys
import logging
import random
import numpy as np
import sacrebleu as scb

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict

def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def convert_text(text):
    #return text
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text.lower()

def eval_mover_score(ref_file, pred_file):
    try:
        refs = get_lines(ref_file)
        sys = get_lines(pred_file)
        idf_dict_hyp = get_idf_dict(sys) 
        idf_dict_ref = get_idf_dict(refs) 

        scores = word_mover_score(refs, sys, idf_dict_ref, idf_dict_hyp, \
                          stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
        return round(np.mean(scores),3) , round(np.median(scores),3 )
    except Exception as e:
        print(e)
        return 0, 0

def get_lines(fil):
    lines = []
    with open(fil, 'r') as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
            else:
                lines.append('empty')
    return lines

def eval_sacre_bleu(ref_file, pred_file):
    try:
        refs = [get_lines(ref_file)]
        sys = get_lines(pred_file)
        bleu = scb.corpus_bleu(sys, refs)
        return bleu.score
    except:
        return 0

def eval_bleu(ref_file, pred_file):
    refs = [get_lines(ref_file)]
    sys = get_lines(pred_file)
    bleu = multi_list_bleu( refs, sys) 
    return bleu


def eval_bleu_sents(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/data/"

    cmd_string = "perl " + folder_data_before + "/multi-bleu.perl -lc " + ref_file + " < " \
                  + pred_file + " > " + pred_file.replace("txt", "bleu")

    os.system(cmd_string)

    bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0]

    return bleu_info

def eval_bertscore(ref_file, pred_file):

    cmd_string = "bert-score -c " + pred_file + " -r " \
                  + ref_file + " --lang en > " + pred_file.replace("notok", "bertscore")

    os.system(cmd_string)

    bertscore_info = open(pred_file.replace("notok", "bertscore"), 'r').readlines()[0].strip()

    return bertscore_info


def eval_meteor(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "java -jar " + folder_data_before + "/meteor-1.5.jar " + pred_file + " " \
                  + ref_file + " > " + pred_file.replace("txt", "meteor")

    os.system(cmd_string)

    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()

    return meteor_info

def eval_chrf(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "python " + folder_data_before + "/chrf++.py -H " + pred_file + " -R " \
                  + ref_file + " > " + pred_file.replace("txt", "chrf")

    os.system(cmd_string)

    chrf_info_1 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[1].strip()
    chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()

    return chrf_info_1 + " " + chrf_info_2

def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        for text in tqdm(f.readlines()):
            tokenized = tokenizer.batch_encode_plus(
                [text.strip() + ' </s>'], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors, # add_special_tokens=True,
                #[text.strip()], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors, # add_special_tokens=True,

            )
            examples.append(tokenized)
    return examples


def encode_file_bart(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        for text in tqdm(f.readlines()):
            tokenized = tokenizer.batch_encode_plus(
                [text.strip()], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors, #add_special_tokens=True,
            )
            examples.append(tokenized)
    return examples


def encode_file_table2text(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        for text in tqdm(f.readlines()):
            tokenized = tokenizer.batch_encode_plus(
                [text.strip() + ' </s>'], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples

def encode_file_sents_unsup_bart(tokenizer, data_path, max_source_length, max_target_length, pad_to_max_length=True, return_tensors="pt"):

    with open(data_path, "r") as f:
        texts = f.readlines()

        texts_source = []
        texts_target = []
        for idx_t, t in tqdm(enumerate(texts)):

            t_src = t.split()
            t_tgt = t.split()

            if len(t_tgt) < 6:
                continue

            noise_density = .15
            mean_noise_span_length = 3
            num_noise_tokens = round(len(t_tgt) * noise_density)

            try:

                count = 0
                while count < num_noise_tokens:
                    idx = np.random.randint(len(t_src))
                    if t_src[idx] == '<mask>':
                        continue
                    spam = np.random.poisson(mean_noise_span_length)
                    for i in range(spam):
                        t_src.pop(idx)
                        if idx >= len(t_src):
                            idx = idx - 1
                        count += 1
                    if (idx - 1 >= 0 and t_src[idx - 1] == '<mask>') or \
                            (idx + 1 < len(t_src) and t_src[idx + 1] == '<mask>'):
                        continue
                    if idx == len(t_src) - 1:
                        t_src.append('<mask>')
                    else:
                        t_src.insert(idx, '<mask>')

                texts_source.append(' '.join(t_src))
                texts_target.append(' '.join(t_tgt))

            except:
                continue


        examples_source = []
        examples_tgt = []

        for text in tqdm(texts_source):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_source_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples_source.append(tokenized)

        for text in tqdm(texts_target):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_target_length, pad_to_max_length=pad_to_max_length,
                return_tensors=return_tensors,
            )
            examples_tgt.append(tokenized)
    return examples_source, examples_tgt

def encode_file_sents_unsup_leonardo(tokenizer, data_path, max_source_length, max_target_length, pad_to_max_length=True, return_tensors="pt"):

    def get_extra_token(count):
        TOKEN_EXTRA_ID = '<extra_id_'
        token = TOKEN_EXTRA_ID + str(count) + '>'
        count += 1
        return token, count

    with open(data_path, "r") as f:
        texts = f.readlines()

        texts_source = []
        texts_target = []
        for t in tqdm(texts):
            count = 1

            t = t.split()

            text_source = []
            text_target = []

            noise_density = .15
            mean_noise_span_length = 3
            num_noise_tokens = round(len(t) * noise_density)
            num_noise_spans = round(
                num_noise_tokens / mean_noise_span_length)

            idxs = set()
            while len(idxs) < num_noise_tokens:
                idx = np.random.randint(len(t))

                idxs.add(idx)
                if idx + 1 < len(t):
                    idxs.add(idx + 1)

                if idx + 2 < len(t):
                    idxs.add(idx + 2)

                if len(idxs) >= num_noise_tokens:
                    break

            keep_source = True
            keep_target = True
            first_ = ''
            cont_source = 0
            cont_target = 0
            for idx, word in enumerate(t):

                if idx not in idxs:
                    keep_target = True
                    text_source.append(word)
                    cont_source += 1
                    if keep_source:
                        if first_ == '':
                            first_ = 'S'
                        if first_ == 'S':
                            token, count = get_extra_token(count)
                        text_target.append(token)
                        keep_source = False

                else:
                    keep_source = True
                    text_target.append(word)
                    cont_target += 1
                    if keep_target:
                        if first_ == '':
                            first_ = 'O'
                        if first_ == 'O':
                            token, count = get_extra_token(count)
                        text_source.append(token)
                        keep_target = False



            text_target.append('</s>')
            texts_source.append(' '.join(text_source))
            texts_target.append(' '.join(text_target))

        examples_source = []
        examples_tgt = []

        for text in tqdm(texts_source):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_source_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples_source.append(tokenized)

        for text in tqdm(texts_target):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_target_length, pad_to_max_length=pad_to_max_length,
                return_tensors=return_tensors,
            )
            examples_tgt.append(tokenized)
    return examples_source, examples_tgt


def encode_file_sents_unsup(tokenizer, data_path, max_source_length, max_target_length, pad_to_max_length=True, return_tensors="pt"):

    def get_extra_token(count):
        TOKEN_EXTRA_ID = '<extra_id_'
        token = TOKEN_EXTRA_ID + str(count) + '>'
        count += 1
        return token, count

    max_span_length = 5
    plm_probability = float(1/5.0)

    with open(data_path, "r") as f:
        texts = f.readlines()

        texts_source = []
        texts_target = []
        for t in tqdm(texts):
            count = 1

            if not t.strip():
                continue

            t = t.split()
            if len(t) < 6:
                continue

            text_source = []
            text_target = []

            max_len = len(t)
            cur_len = 0
            masked_spans = 0

            while cur_len < max_len and masked_spans < 100:
                # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                span_length = np.random.randint(1, max_span_length + 1)
                # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
                context_length = int(span_length / plm_probability)
                # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
                start_index = min(cur_len + np.random.randint(context_length - span_length + 1), max_len)
                for i in range(cur_len, start_index):
                    text_source.append(t[i])


                if start_index < max_len and start_index + span_length < max_len:
                    cur_len += context_length
                    token, count = get_extra_token(count)
                    text_source.append(token)
                    text_target.append(token)

                    for i in range(start_index + span_length , min(max_len, cur_len)):
                        text_source.append(t[i])
                
                    for i in range(start_index, start_index + span_length):
                        text_target.append(t[i])
                    masked_spans +=1
                else:
                    break

            for i in range(max(start_index,cur_len), max_len):
                text_source.append(t[i])


            text_target.append('</s>')
            if text_source and text_target:
                texts_source.append(' '.join(text_source))
                texts_target.append(' '.join(text_target))


        examples_source = []
        examples_tgt = []

        for text in tqdm(texts_source):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_source_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples_source.append(tokenized)

        for text in tqdm(texts_target):
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_target_length, pad_to_max_length=pad_to_max_length,
                return_tensors=return_tensors,
            )
            examples_tgt.append(tokenized)
    return examples_source, examples_tgt

def encode_file_sent_source(tokenizer, data_path, ids, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        texts = f.readlines()
        if len(ids) < len(texts):
            texts = [texts[i] for i in ids]
        for text in tqdm(texts):

            tokenized = tokenizer.batch_encode_plus(
                ['translate Graph to English: ' + text.strip() + ' </s>'], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples


def encode_file_sent_target(tokenizer, data_path, ids, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        texts = f.readlines()
        if len(ids) < len(texts):
            texts = [texts[i] for i in ids]
        for text in tqdm(texts):

            tokenized = tokenizer.batch_encode_plus(
                [text.strip() + ' </s>'], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples


def encode_file_text2graph(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        for text in tqdm(f.readlines()):
            tokenized = tokenizer.batch_encode_plus(
                ['translate English to Graph: ' + text.strip() + ' </s>'], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples

class AgendaGraph2textUnsupBARTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=768,
        max_target_length=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.source, self.target = encode_file_sents_unsup_bart(tokenizer, os.path.join(data_dir, type_path + ".target"), max_source_length,
                                                                 max_target_length)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}


    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}

class TextUnsupDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=768,
        max_target_length=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.source, self.target = encode_file_sents_unsup_leonardo(tokenizer, os.path.join(data_dir, type_path + ".target"), max_source_length,
                                                                 max_target_length)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}


    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}

class Table2textDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./data/sciLang/",
        type_path="train",
        max_source_length=768,
        max_target_length=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source = encode_file(tokenizer, os.path.join(data_dir, type_path + ".source"), max_source_length)
        self.target = encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}


    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}


class SentsDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=768,
        max_target_length=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        print('sentence files...')
        f = os.path.join(data_dir, "sentences.source")
        number_samples = 20000
        with open(f, "r") as f:
            texts = f.readlines()
        self.ids = random.sample(range(0, len(texts)), number_samples)
        self.source_sents = encode_file_sent_source(tokenizer, os.path.join(data_dir, "sentences.source"), self.ids,
                                          max_source_length)
        self.target_sents = encode_file_sent_target(tokenizer, os.path.join(data_dir, "sentences.target"), self.ids,
                                          max_target_length)

        self.source = self.source_sents
        self.target = self.target_sents

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    # def red(self):
    #     return BatchEncoding, (self.data,)
    #
    # BatchEncoding.__reduce__ = red

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}




class Table2textBARTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=768,
        max_target_length=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source = encode_file_bart(tokenizer, os.path.join(data_dir, type_path + ".source"), max_source_length)
        self.target = encode_file_bart(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}


    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}


