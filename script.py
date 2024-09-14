import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd

s3_path = "s3://hugging-face-text-multi-class/training-data/newsCorpora.csv"
df = pd.read_csv(s3_path, sep='\t', names=["ID","TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"])

df_working = df_working[["TITLE", "CATEGORY"]]

category_dict = {
    'e' : 'Entertainment',
    'b' : 'Business',
    't' : 'Science',
    'm' : 'Health' 
}

def update_category(x):
    return category_dict[x]

df_working['CATEGORY'] = df_working['CATEGORY'].apply(lambda x: update_category(x))

encode_dictionary = {}

def encode_category(x):
    if x not in encode_dictionary.keys():
        encode_dictionary[x] = len(encode_dictionary)
    return encode_dictionary[x] 

df_working['ENCODE_CAT'] = df_working['CATEGORY'].apply(lambda x: encode_category(x))

df_working = df_working.reset_index(drop=True)
        

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class UpdatedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,   # add CLS and SEP
            max_length = self.max_len,
            padding = 'max_length',
            return_token_type_ids= True,
            truncation=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'targets' : torch.tensor(self.data.iloc[index, 2], dtype=torch.long)                     
        
        }
    
    def __len__(self):
        return self.len
    
    
TRAIN_SIZE = 0.8
train_dataset = df_working.sample(frac=TRAIN_SIZE, random_state=200)
test_dataset = df_working.drop(train_dataset.index).reset_index(drop=True)
train_dataset.reset_index(drop=True)

print("Dateset: {}".format(df_working.shape))
print("train_dataset: {}".format(train_dataset.shape))
print("test_dataset: {}".format(test_dataset.shape))

MAX_LENGTH = 512
TRAIN_BATCH = 4
VALIDATION_BATCH = 2

train_set = UpdatedDataset(train_dataset, tokenizer, MAX_LENGTH)
test_set = UpdatedDataset(test_dataset, tokenizer, MAX_LENGTH)
      
train_parameters = {
    'batch_size' : TRAIN_BATCH,
    'shuffle' : True,
    'num_workers' : 0      
}   

test_parameters = {
    'batch_size' : VALIDATION_BATCH,
    'shuffle' : True,
    'num_workers' : 0      
}      
  
train_dataloader = DataLoader(train_set , ** train_parameters)
test_dataloader = DataLoader(test_set , ** test_parameters)


class DistilBERT(torch.nn.Module):
    def __init__(self):
        super(DistilBERT, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased') #loads the model with using lowercases and trained on uncased text
        self.pre_classifier = torch.nn.Linear(768, 768)  #fully connected layer (dense), currently 768 d output
        self.dropout = torch.nn.Dropout(0.3)    #regularization  technique to prevent overfitting
        self.classifier = torch.nn.Linear(768,4)
    
    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids = input_ids, attention_mask=attention_mask)
        print(output_1.shape)
        hidden_state = output_1[0] 
        pooler = hidden_state[:, 0]   #(batch_size, sequence_lenght, hidden_seize)
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler) #, activation function to introduce non_linearity after dense layer
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
    
    
def accurancy(big_idx, targets):
    nof_correct = (big_idx == targets).sum().item()
    return nof_correct

def train(epoch, model, device, train_loader, optimizer, loss_function):
    train_loss = 0
    n_correct = 0
    train_step = 0
    train_example = 0
    model.train()
    
    for _,data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        outputs = model(ids, mask)
        
        loss = loss_function(outputs, targets)
        train_loss += loss.item()
        val, idx = torch.max(outputs.data, dim=1)
        n_correct += accurancy(idx, targets)
        
        train_step += 1  
        train_example += targets.size(0)
        
         
        optimizer.zero_grad()  # resets the gradient from previous step (dont accumualte)
        loss.backward()   # backpropagation, computes the gradients of the loss with respect to the model`s parameter
        optimizer.step() # update the model`s parameters
        
    epoch_loss = train_loss / train_step
    epoch_accuracy = (n_correct*100) / train_example
    print(f"Training Loss per Epoch: {epoch_loss}")
    print(f"Training Accuracy per Epoch: {epoch_accuracy}")
    
    return
    
    
def valid(epoch, model, test_loader,device, loss_function):
    model.eval()
    
    n_correct = 0
    test_loss = 0
    test_step=0
    test_example = 0
    
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['target'].to(device, dtype= torch.long)
            
            output = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            test_loss += loss.item()
            val, idx = torch.max(outputs.data, dim=1)
            n_correct += accurancy(idx, targets)
            
            test_step +=1
            test_example += targets.size(0)
            
    epoch_loss = test_loss / test_step
    epoch_accuracy = (n_correct*100)/test_example
    print(f"Validation loss per Epoch: {epoch_loss} at epoch: {epoch}")
    print(f"Validation accuracy per Epoch: {epoch_accuracy} at epoch: {epoch}")
    
    return

def main():
    print("starting......")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERT()
    model.to(device)
    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params = model.parameters(), lr =LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()
    
    # TRAINING LOOP
    
    EPOCHS = 2
    for epoch in range(EPOCHS):
        print(f"Starting epoch: {epoch}")
        train(epoch, model, device, train_dataloader, optimizer, loss_function)
        valid(epoch, model, test_dataloader, device, loss_function)
        
    output_directory = os.environ["SM_MODEL_DIR"] #SM designated output directory
    output_model_file = os.path.join(output_directory, 'pytoch_distilber.bin')
    output_vocab_file = os.path.join(output_directory, 'vocab_distilbert.bin')
    torch.save(model.state_dict(), output_model_file)
    tokenizer.save_vocabulary(output_model_file)
        
    
    

if __name__ == '__main__':
    main()
    
        
    
    
    
    
                             
        
    
    
        
        
        
    

            
            
        