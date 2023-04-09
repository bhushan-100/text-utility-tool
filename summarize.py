from transformers import (
    T5TokenizerFast as T5Tokenizer
)
from model.summarization_model import SummaryModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

TOKENIZER = T5Tokenizer.from_pretrained('t5-small')

trained_model = SummaryModel.load_from_checkpoint("text_wiz_checkpoint.ckpt")


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
