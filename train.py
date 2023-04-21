import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast,BertForMaskedLM, BertTokenizer, TrainingArguments, Trainer, logging
import os

#Inicial parameters
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
torch.backends.cuda.matmul.allow_tf32= True

# Directory settings
INPUT_DIR = '/home/oem/Desktop/ALEX/Project/data/data_1000.csv'
OUTPUT_DIR = '/home/oem/Desktop/ALEX/Project/data/trained.csv'

# Read data
df = pd.read_csv(INPUT_DIR)

# Convert 1d_seq column to uppercase
df['1d_seq'] = df['1d_seq'].str.upper()

# Model settings 
tokenizer_path = '/home/oem/prot_bert'
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to("cuda")

tokenizer.add_tokens("J")
model.resize_token_embeddings(len(tokenizer))


# Train arguments
logging.set_verbosity_error()
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    tf32= True,
    logging_steps= 1
)

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence= self.sequences[idx]
        label= self.labels[idx]
        sequence_spaces= ' '.join(list(sequence))
        label_spaces= ' '.join(list(label))
        encoded_inputs = self.tokenizer(sequence_spaces, padding= True,truncation=True, max_length=400, return_tensors='pt')
        encoded_labels = self.tokenizer(label_spaces, padding= True,truncation=True, max_length=400, return_tensors='pt')
        labels= [-100 if token == self.tokenizer.pad_token_id else encoded_labels['input_ids'][0][i] for i, token in enumerate(encoded_inputs['input_ids'][0])]
        print(f"Input sequence: {self.tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][0])}")
        print(f"Label sequence: {self.tokenizer.convert_ids_to_tokens(labels)}")
        return {
            'input_ids': encoded_inputs['input_ids'][0],
            'attention_mask': encoded_inputs['attention_mask'][0],
            'labels': torch.tensor(labels)
        }

# Create dataset from sequences and labels
dataset = CustomDataset(df['fasta_seq'].tolist(), df['1d_seq'].tolist(), tokenizer)


# Define trainer
trainer = Trainer(
    model=model,                       
    args=training_args,                  
    train_dataset=dataset,
    data_collator=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                                'attention_mask': torch.stack([item['attention_mask'] for item in data]),
                                'labels': torch.stack([item['labels'] for item in data])},
)
dataload = DataLoader(dataset, batch_size=4, shuffle=True,collate_fn= trainer.data_collator)

# Train model
trainer.train()

# Save model
model.save_pretrained('./model')
