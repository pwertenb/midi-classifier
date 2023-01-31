import midi
import config

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import torch.autograd as autograd

import os
import numpy as np

# train loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss      
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# test loop
def test_loop(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:      
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Epoch {epoch:4d}: Avg loss: {test_loss:8f} Accuracy: {(100*correct):5f}%")

class MidiLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, orig_y, answers, chord_notes):
        tally = 0
        ctx.inputs = inputs
        ctx.targets = targets
        ctx.orig_y = orig_y
        ctx.answers = answers
        ctx.chord_notes = chord_notes
        for i in range(len(inputs)):
            pred_chord = orig_y[torch.argmax(inputs[i])]
            target_chord = orig_y[torch.argmax(targets[i])]  
            
            if target_chord not in answers[chord_notes[pred_chord]]:
                tally += 0
            else:
                tally += 1
        result = torch.FloatTensor([1 - (tally / len(inputs[0]))])
        result.requires_grad_()
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        return ctx.inputs - ctx.targets, None, None, None, None

class MidiLoss(nn.Module):
    def __init__(self, orig_y):
        super(MidiLoss, self).__init__()
        self.chord_notes, self.answers = midi.get_base_data()
        self.orig_y = orig_y
        
    def forward(self, inputs, targets):
        return MidiLossFn.apply(inputs, targets, self.orig_y, self.answers, self.chord_notes)

class NeuralNetwork(nn.Module):
    def __init__(self, trained_data):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 96),
            nn.ReLU(),
            nn.Linear(96, 282),
        )
        self.chord_notes = trained_data.chord_notes
        self.answers = trained_data.answers
        self.orig_y = trained_data.orig_y
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        result = nn.functional.normalize(logits)
        return result
        
    def to_chord(self, y):
        return self.orig_y[torch.argmax(y)]

class MidiDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        if train:
            self.X, self.y = midi.create_train_data()
            self.chord_notes, self.answers = midi.get_base_data()
        else:
            self.X, self.y, self.chord_notes, self.answers = midi.create_test_data()

        self.orig_y = np.copy(self.y)
        self.le = OneHotEncoder()
        self.X = torch.from_numpy(self.X.astype('float32'))
        self.y = np.reshape(self.y, (len(self.y), 1))
        self.y = self.le.fit_transform(self.y)
        self.y = torch.from_numpy(self.y.toarray().astype('float32'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.flatten(self.X[idx]), torch.flatten(self.y[idx])
       

train_data = MidiDataset(train=True)
test_data = MidiDataset(train=False)
train_dataloader = DataLoader(train_data, shuffle=True)
test_dataloader = DataLoader(train_data, shuffle=True)
model = NeuralNetwork(train_data)
optimizer = torch.optim.SGD(model.parameters(), lr=config.LR)
loss_fn = MidiLoss(train_data.orig_y)

if not os.path.exists(config.MODEL_FILENAME):
    print('Creating and training model...')

    for t in range(config.MAX_EPOCHS):
        #print(f"Epoch {t+1}")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn, t+1)
    
    print("Saving model...")    
    torch.save(model, config.MODEL_FILENAME)
    
model = torch.load(config.MODEL_FILENAME)
    
eval_dataloader = train_dataloader
with torch.no_grad():
    tally = 0
    for X, y in eval_dataloader:      
        pred = model(X)
        pred_chord = model.to_chord(pred)
        target_chord = model.to_chord(y)   
        if pred_chord in model.answers[model.chord_notes[target_chord]]:
            tally += 1
        else:
            print('pred:',pred_chord,'targ:',target_chord)
            
    print('accuracy:', tally / len(eval_dataloader.dataset))

print("Done.")

