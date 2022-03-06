import json
import numpy as np
from bot import stem,tokenize,bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from mode import NeuralNet
import  multiprocessing 

with open('intents.json','r') as f:
       intents=json.load(f)
all_words=[]
tags=[]
xy=[]
ignore_words=["!",",","?","."]
for intent in intents["intents"]:
     tag=intent['tag']
     tags.append(tag)
     for pattern in intent["patterns"]:
       w=tokenize(pattern)
       w=[stem(h)for h in w ]
       w=[stem(w) for w in w if(w not in ignore_words)]
       all_words.extend(w)
       xy.append((w,tag))

tags=sorted(set(tags))
all_words=sorted(set(all_words))

#delete pounctuation words from all_words
all_words=[stem(w) for w in all_words if(w not in ignore_words)]
x_train=[]
y_train=[]
for (pattern_sentece,tag)in xy:
     bag=bag_of_words(pattern_sentece,all_words)
     x_train.append(bag)
     label=tags.index(tag)
     y_train.append(label)
x_train=np.array(x_train)
y_train=np.array(y_train)
print("x_train=",str(x_train))
print("tags:"+str(tags)) 
print("deb")
print("all_words:"+str(all_words))


print("xy"+str(xy))
print(xy)

print("x_train",str(x_train))
print("y_train",str(y_train))
print("all_words:"+str(all_words))
print("fin")
class chatDataset(Dataset):
     def __init__ (self): 
             self.n_samples = len(x_train)
             self.x_data=x_train
             self.y_data=y_train

     def __getitem__(self,index):
             return self.x_data[index],self.y_data[index]
     
     def __len__(self):
             return self.n_samples

input_size=len(x_train[0])
learning_rate=0.001
num_epochs=1000
hidden_size=len(tags)
output_size=8
batch_size= 8
dataset=chatDataset()
train_Loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)
device=torch.device('cuda' if torch.cuda.is_available()  else 'cpu' )
model=NeuralNet(input_size,hidden_size,output_size).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for  epoch in range (num_epochs):
     for  (words ,labels) in train_Loader:
           print("bien recu")
           words=words.to(device)
           labels=labels.to(device)
            #forward  
           outputs=model(words)
           loss=criterion(outputs,labels)
            #backward and optimizer step 
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
     if (epoch+1)%100 ==0:
        print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}')
        
print(f'final loss,loss={loss.item():.4f}')







