import json
import os
import sys

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd

from transformers import AutoTokenizer
from indobenchmark import IndoNLGTokenizer

from tqdm import tqdm

class Preprocessor(L.LightningDataModule):
    def __init__(self,
                 max_length,
                 batch_size,
                 lm_model = None):
        super(Preprocessor, self).__init__()
        self.indosum_dir = "datasets/indosum"
        if lm_model:
            self.tokenizer = AutoTokenizer.from_pretrained(lm_model)    
        else:
            self.tokenizer =  IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
        self.max_length = max_length
        self.batch_size = batch_size
        
    def join_paragraphs(self, paragraphs):
        # Join List of paragraphs to string paragraphs
        string_paragraph = ""
        for parag in paragraphs:
            for kal in parag:
                kalimat = " ".join(kal)
                string_paragraph += kalimat 
        
        return string_paragraph
    
    def join_summary(self, summaries):
        string_summary = ""
        for sumr in summaries:
            kal_sum = " ".join(sumr)
            string_summary += kal_sum
        
        return string_summary
            
    def load_data(self, flag):
        list_files = os.listdir(self.indosum_dir)
        datasets = []
        for fl in list_files:
            if flag in fl:
                with open(f"{self.indosum_dir}/{fl}", "r", encoding = "utf-8") as json_reader:
                    # load file jsonl (jsonl = kumpulan file json format di gabung jadi satu file)
                    
                    data_raw = json_reader.readlines()                   
                    # json_raw = [json.loads(jline) for jline in json_reader.readlines().rstrip().splitlines()]
                
                json_raw = []  
                for dd in data_raw:
                    
                    data_line = json.loads(dd)
                    paragraphs = self.join_paragraphs(data_line["paragraphs"])
                    summary = self.join_summary(data_line["summary"])
                    
                    data = {
                        "id": data_line["id"],
                        "paragraphs": paragraphs,
                        "summary": summary,
                    }
    
                    json_raw.append(data)
                
                datasets += json_raw
        
        return datasets
    
    def list2tensor(self, data, stage):
        x_ids, x_att, y_ids, y_att = [], [], [], []
        
        for d in tqdm(data):
            x_tok = self.tokenizer(
                d["paragraphs"],
                truncation = True,
                max_length = self.max_length,
                padding = "max_length"
            )
    
            y_tok = self.tokenizer(
                d["summary"],
                truncation = True,
                max_length = self.max_length,
                padding = "max_length"
            )
            
            x_ids.append(x_tok["input_ids"])
            x_att.append(x_tok["attention_mask"])
            
            y_ids.append(y_tok["input_ids"])
            y_att.append(y_tok["attention_mask"])

        x_ids = torch.tensor(x_ids)
        x_att = torch.tensor(x_att)
        y_ids = torch.tensor(y_ids)
        y_att = torch.tensor(y_att)
        
        tensor_data = TensorDataset(x_ids, x_att, y_ids, y_att)
        
        torch.save(tensor_data, f"datasets/preprocessed/indosum/{stage}.pt")
        
        return tensor_data
    
    def preprocessor(self):
        raw_train_data = self.load_data(flag = "train") 
        raw_val_data = self.load_data(flag = "dev") 
        raw_test_data = self.load_data(flag = "test")
        
        preprocessed_dir =  "datasets/preprocessed/indosum/"
        
        if os.path.exists(f"{preprocessed_dir}train.pt"):
            train_set = torch.load(f"{preprocessed_dir}train.pt")
        else:
            train_set = self.list2tensor(raw_train_data, stage = "train")
            
        if os.path.exists(f"{preprocessed_dir}val.pt"): 
            val_set = torch.load(f"{preprocessed_dir}val.pt")
        else:
            val_set = self.list2tensor(raw_val_data, stage = "val")
            
        if os.path.exists(f"{preprocessed_dir}test.pt"): 
            test_set = torch.load(f"{preprocessed_dir}test.pt")
        else:
            test_set = self.list2tensor(raw_test_data, stage = "test")
        
        return train_set, val_set, test_set
    
    def setup(self, stage = None):
        train_set, val_set, test_set = self.preprocessor()
        
        if stage == "fit":
            self.train_data = train_set
            self.val_data = val_set
        elif stage == "test":
            self.test_data = test_set
    
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 3
        )

    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 3
        )

    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 3
        )       
            
        
if __name__ == "__main__":
    pre = Preprocessor(max_length = 512,
                       batch_size = 10,
                       lm_model = "facebook/mbart-large-50",)
    # pre.setup(stage = "fit")
    # print(pre.train_dataloader())
    pre.preprocessor()