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


MODEL = T5ForConditionalGeneration.from_pretrained(
    't5-small', return_dict=True)
TOKENIZER = T5Tokenizer.from_pretrained('t5-small')
N_EPOCHS = 2
BATCH_SIZE = 4
DEVICE = "cuda:0"

articles_path = "./input/News Articles"
summaries_path = "./input/Summaries"
categories_list = ["politics", "sport", "tech", "entertainment", "business"]
# categories_list = ["politics", "tech"]

torch.cuda.empty_cache()


def read_files_from_folders(articles_path, summaries_path, categories_list, encoding="ISO-8859-1"):
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


articles, summaries, categories = read_files_from_folders(
    articles_path, summaries_path, categories_list)

df = pd.DataFrame(
    {'articles': articles, 'summaries': summaries, 'categories': categories},)
df = df[['articles', 'summaries']]

# -- get length of each article and summary for analysis
df['articles_length'] = df['articles'].apply(lambda x: len(x.split()))
df['summaries_length'] = df['summaries'].apply(lambda x: len(x.split()))
# print(df)


df['articles'] = df['articles'].str.encode(
    'ascii', 'ignore').str.decode('ascii')
df['summaries'] = df['summaries'].str.encode(
    'ascii', 'ignore').str.decode('ascii')

df.dropna()

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# print(test_df)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


class SummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row['articles']
        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        summary_encoding = self.tokenizer(
            data_row['summaries'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        # to make sure we have correct labels for T5 text generation
        labels[labels == 0] = -100
        return dict(
            text=text,
            summary=data_row['summaries'],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )


class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = SummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = SummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )


text_token_counts, summary_token_counts = [], []
for _, row in train_df.iterrows():
    text_token_count = len(TOKENIZER.encode(row['articles']))
    text_token_counts.append(text_token_count)
    summary_token_count = len(TOKENIZER.encode(row['summaries']))
    summary_token_counts.append(summary_token_count)

data_module = SummaryDataModule(train_df, test_df, TOKENIZER, BATCH_SIZE)
data_module.setup()

next(iter(data_module.train_dataloader().dataset))


class SummaryModel(pl.LightningModule):
    def __init__(self):
        super(SummaryModel, self).__init__()
        self.model = MODEL

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             labels=labels, decoder_attention_mask=decoder_attention_mask)

        if labels is not None:
            outputs.loss = outputs.loss
        else:
            outputs.loss = None
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=labels_attention_mask,
                             labels=labels)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=labels_attention_mask,
                             labels=labels)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=labels_attention_mask,
                             labels=labels)

        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


model = SummaryModel()

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
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
torch.cuda.memory_summary(device=None, abbreviated=False)

trainer.fit(model, data_module)

trained_model = SummaryModel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
trained_model.freeze()


def summarizeText(text):
    text_encoding = TOKENIZER(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
        TOKENIZER.decode(gen_id, skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    return "".join(preds)


text = '''At least 12 people died and 19 others were rescued after the roof of an ancient well situated in Beleshwar Mahadev Jhulelal Temple in Madhya Pradesh's Indore collapsed during the Ram Navami festival, an official said.

A large number of people had gathered on the roof of the ancient bavdi
and it caved in as it was unable to bear the load, according to PTI report.

In a video shared by news agency ANI, police and locals were seen
trying to rescue the devotees trapped in the collapsed stepwell.
Chief minister Shivraj Singh Chouhan took cognizance of the
incident and instructed local authorities to speed up the rescue operation,
according to his office. He has also announced a compensation of
₹5 lakh each to the kin of the deceased, and ₹50,000 each to the injured.

“We are with the bereaved families in this hour of grief.
Proper arrangements have been made for the treatment of the injured,
the entire medical expenses will be borne by the state government,”
CM Chouhan said.

Meanwhile, Prime Minister Narendra Modi expressed anguish at the mishap.
"Extremely pained by the mishap in Indore. Spoke to CM Shivraj Chouhan Ji
and took an update on the situation. The state government is spearheading
rescue and relief work at a quick pace. My prayers with all those affected
and their families," he tweeted.'''

model_summary = summarizeText(text)

print(model_summary)
