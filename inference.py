import torch
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel

MAX_LEN = 512

class DistilBERT(torch.nn.Module):
    def __init__(self):
        super(DistilBERT, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
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
    
def model_function(model_directory):
    print("Loading model from model_directory")
    model = DistilBERT()
    model_state_dict = torch.load(os.path.join(model_directory, "pytoch_distilber.bin"), map_location = torch.device("cpu"))
    model.load_state_dict(model_state_dict)
    
    return model

def input_function(request_body, request_content_type):
    if request_content_type == 'application/json':
        input = json.loads(request_body)
        sentence = input['inputs']
        return sentence
    else:
        raise ValueError("Unsupported content type: {request_content_type}")
        
def predict_function(input, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(input, return_tensors="pt").to(device)
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(ids,mask)
        
    probs = torch.softmax(output, dim=1).cpu().numpy()
    class_names = ["Business", "Science", "Entertainment", "Health"]
    pred_class = probs.argmax(axis=1)[0]
    pred_label = class_names[pred_class]
    
    return {"pred label" : pred_label, "probs": probs.tolist()}

def output_function(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError("Unsupported accept type: {accept}")
    


             
             