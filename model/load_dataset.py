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

articles_path = "./input/News Articles"
summaries_path = "./input/Summaries"
categories_list = ["politics", "sport", "tech", "entertainment", "business"]


def load_dataset(articles_path=articles_path, summaries_path=summaries_path, categories_list=categories_list, encoding="ISO-8859-1"):
    articles = []
    summaries = []
    categories = []
    for category in categories_list:
        article_paths = glob.glob(os.path.join(
            articles_path, category, '*.txt'), recursive=True)
        summary_paths = glob.glob(os.path.join(
            summaries_path, category, '*.txt'), recursive=True)
        print(
            f'found {len(article_paths)} file in articles {category} folder, {len(summary_paths)} file in summaries/{category}')
        if len(article_paths) != len(summary_paths):
            print('number of files is not equal')
            return
        for idx_file in range(len(article_paths)):
            categories.append(category)
            with open(article_paths[idx_file], mode='r', encoding=encoding) as file:
                articles.append(file.read())

            with open(summary_paths[idx_file], mode='r', encoding=encoding) as file:
                summaries.append(file.read())
    print(
        f'total {len(articles)} file in articles folders, {len(summaries)} file in summaries folders')
    return articles, summaries, categories
