import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from joblib import cpu_count
import matplotlib.pyplot as plt

import model
import graph_funcs

# Set options
BATCH_SIZE = 32
HIDDEN_CHANNELS = 512
NUM_CONV_LAYERS = 3
EPOCHS = 7000
LR = 0.01
ID = random.randint(0,10000)

torch.manual_seed(25)
torch.set_num_threads(cpu_count())

df = pd.read_excel(f'../data/Round0_training.xlsx')
target = 'mean'

# Prepare data
for output in ['Rad1','Rad2','Rad3','Rad4','Rad5','Rad6']:
    df[output] = np.log(df[output])

dataset = graph_funcs.create_dataset(df)

for row,data in enumerate(dataset):
    if target == 'min':
        data.y = np.min(df.iloc[row,7:12])       
    if target == 'mean':
        data.y = np.mean(df.iloc[row,7:12])
    if target == 'max':
        data.y = np.max(df.iloc[row,7:12])       

# Split data
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=25)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)

# Define model
gnn_model = model.GCN(hidden_channels=HIDDEN_CHANNELS, num_conv_layers=NUM_CONV_LAYERS)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

# Train model
train_mse = []
test_mse = []

minimum_error = 0

for epoch in tqdm(range(1, EPOCHS)):

    model.train(gnn_model, train_loader, optimizer, criterion)

    epoch_train_mse = model.test(gnn_model,train_loader,criterion)
    train_mse.append(epoch_train_mse)
    
    epoch_test_mse = model.test(gnn_model,test_loader,criterion)
    test_mse.append(epoch_test_mse)

    if epoch_test_mse < minimum_error:
        tqdm.write(f'Saving model at epoch {epoch} with test MSE of {epoch_test_mse}')
        torch.save(gnn_model,f'../results/model.pt')
    
    minimum_error = np.min(test_mse)

print('=====================')
print(f'Run_ID {ID}')
print(f'BATCH_SIZE = {BATCH_SIZE}')
print(f'HIDDEN_CHANNELS = {HIDDEN_CHANNELS}')    
print(f'EPOCHS = {len(train_mse)}')
print(f'LR = {LR}')   
print(f'minimum TRAIN MSE of {min(train_mse):.2f}, RMSE of {np.sqrt(min(train_mse)):.2f}, at epoch {np.argmin(train_mse)}')
print(f'minimum TEST MSE of {min(test_mse):.2f}, RMSE of {np.sqrt(min(test_mse)):.2f}, at epoch {np.argmin(test_mse)}') 
print(gnn_model)

