from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('<|endoftext|>')

        for line in lines:
            if line.strip():
                tokenized_text = tokenizer.encode(line.strip(), add_special_tokens=True)
                self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add special tokens
special_tokens_dict = {'pad_token': '[PAD]', 'eos_token': '<|endoftext|>', 'bos_token': '<|startoftext|>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Prepare dataset
file_path = 'training_data.txt'
train_dataset = CustomDataset(file_path, tokenizer)

# Check if the dataset is loaded correctly
print(f"Dataset loaded with {len(train_dataset)} samples.")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
