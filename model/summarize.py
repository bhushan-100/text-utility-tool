from transformers import (
    T5TokenizerFast as T5Tokenizer
)
from summarization_model import SummaryModel
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


text = '''At least twelve people died and nineteen others were rescued after the roof of an ancient well collapsed during the Ram Navami festival.

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
"Extremely pained by the mishap in Indore.", spoke to CM Shivraj Chouhan Ji
and took an update on the situation."The state government is spearheading
rescue and relief work at a quick pace. My prayers with all those affected
and their families," he tweeted.'''

model_summary = summarizeText(text)

print(model_summary)
