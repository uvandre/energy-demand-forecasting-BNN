import re
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchbnn as bnn
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

prior_mu = 0
prior_sigma = 0.9

kl_weight = 0.1 

lr = 0.005
epochs = 5000
n_hidden_layers = 3
batch_size = 2400

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
    
        return data, labels

def MAPELoss(output, target):
    target = target.flatten(start_dim=1)
    return torch.mean(torch.abs((target - output) / target))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('DA_price_v_actual_load.csv', index_col=0)
df = df.drop(df[df['DA Price'] <= 0].index)
df = df[:-44]
train_len = int(len(df)*0.7)


train_df = df[:train_len-6]
train_df = train_df[['hour', 'day', 'month', 'DA Price', 'MW Load']]

test_df = df[train_len:-17]
test_df = test_df[['hour', 'day', 'month', 'DA Price', 'MW Load']]

x_test_arr = np.array([test_df[['hour', 'day', 'month','DA Price']]])
x_test = torch.tensor(x_test_arr).to(device='cpu')

y_test_arr = np.array([test_df[['MW Load']]])
y_test = torch.tensor(y_test_arr).to(device='cpu')

x_test = x_test.reshape((-1, 24, 4))
y_test = y_test.reshape((-1, 24, 1))

x_train_arr = np.array([train_df[['hour', 'day', 'month', 'DA Price']]])
x_train = torch.tensor(x_train_arr).to(device)

y_train_arr = np.array([train_df[['MW Load']]])
y_train = torch.tensor(y_train_arr).to(device)

x_train = x_train.reshape((-1, 24, 4))
y_train = y_train.reshape((-1, 24, 1))

train_dataset = CustomDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=1000),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=1000, out_features=100),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=100, out_features=24),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=24, out_features=1),
)

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=4),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=4),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=4),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=4),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=4),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4, out_features=4),
    nn.ReLU(),
    nn.Flatten(start_dim=1),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=4*24, out_features=50),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=50, out_features=24),
)

model = model.double()
model.to(device)

mse_loss = nn.MSELoss()

kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

optimizer = optim.Adam(model.parameters(), lr=lr)

def train(model, data_loader):
    for _ in tqdm(range(epochs)):
        for data, label in data_loader:
            pre = model(data)

            kl = kl_loss(model)
            mape = MAPELoss(pre, label)
            cost = mape
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    return mape, kl

if __name__ == '__main__':

    mse, kl = train(model, train_dataloader)    
    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))
    lr_name = re.sub(r'[\W\s]', ' ', f'{lr}')
    pm_name = re.sub(r'[\W\s]', ' ', f'{prior_mu}')
    ps_name = re.sub(r'[\W\s]', ' ', f'{prior_sigma}')
    kl_name = re.sub(r'[\W\s]', ' ', f'{kl_weight}')
    path = f"new/final/models_mape_kloss/test_{epochs}epochs_{lr}lr_{prior_mu}pm_{prior_sigma}ps_{kl_weight}kl_{batch_size}bs_{n_hidden_layers}hn.pth"
    torch.save(model, path)

    model.to(device='cpu')

    curr_run_data = {'MSE':[mse.item()],'epochs':[epochs], 'LR': [lr], 'Prior mu':[prior_mu], 'Prior sigma':[prior_sigma], 'KL weight': [kl_weight], 'Batch Size': [batch_size], 'hidden layers': [n_hidden_layers]}
    curr_run_data_df = pd.DataFrame.from_dict(curr_run_data)
    curr_run_data_df.to_csv('new/final/results_mape_kloss.csv', mode='a', index=True, header=False)
    
    for i in tqdm(range(0, len(x_test))):
        with torch.no_grad():
            x = x_test[i]
            x = x.reshape((-1, 24, 4))
            y_pred = model(x).flatten()
            x_vals = x[:,:,-1].flatten()
        plt.scatter(x_vals, y_pred, color='r', s=2)
        plt.scatter(x_vals, y_test[i], color='g', s=2)
        model.train()
    plt.title('Test set results')
    plt.savefig(f'new/final/fig_results_mape_kloss/test_{epochs}epochs_{lr}lr_{prior_mu}pm_{prior_sigma}ps_{kl_weight}kl_{batch_size}bs_{n_hidden_layers}hn.png')