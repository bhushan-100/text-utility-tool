from matplotlib import rc
from pylab import rcParams
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from termcolor import colored
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import json
import os
import time
import glob
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from load_dataset import load_dataset
from preprocess import preprocess, split_data
from summarization_model import SummaryDataset, SummaryDataModule, SummaryModel


MODEL = T5ForConditionalGeneration.from_pretrained(
    't5-small', return_dict=True)
TOKENIZER = T5Tokenizer.from_pretrained('t5-small')
N_EPOCHS = 2
BATCH_SIZE = 4
DEVICE = "cuda:0"

articles, summaries, categories = load_dataset()
df = preprocess(articles, summaries, categories)
train_df, val_df, test_df = split_data(df)


text_token_counts, summary_token_counts = [], []
for _, row in train_df.iterrows():
    text_token_count = len(TOKENIZER.encode(row['articles']))
    text_token_counts.append(text_token_count)
    summary_token_count = len(TOKENIZER.encode(row['summaries']))
    summary_token_counts.append(summary_token_count)

data_module = SummaryDataModule(train_df, test_df, TOKENIZER, BATCH_SIZE)
data_module.setup()
next(iter(data_module.train_dataloader().dataset))

model = SummaryModel()

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
)

logger = TensorBoardLogger('lightning_logs', name="news-summary")

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
)

torch.cuda.empty_cache()
gc.collect()

trainer.fit(model, data_module)

trained_model = SummaryModel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)

trainer.save_checkpoint("text_wiz_checkpoint.ckpt")
