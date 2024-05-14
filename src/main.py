# %%
# Imports

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

import dataloading
import modeling
import preprocessing

# %%
# Config

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
batch_size = config['batch_size']
n_epoch = config['n_epoch']
patience = config['patience']

# %%
# Preprocessing

print('Fetching data...')
data = preprocessing.fetch_data()
print('')

print('Preparing data...')
data, feats, target = preprocessing.prepare_data(data)
print('')

# %%
# Train-validation-test dataloaders

print('Training-validation-test split...')

n = len(data)
train_idc = range(n // 2)
val_idc = range(n // 2, 3 * n // 4)
test_idc = range(3 * n // 4, n)
data_train = data.iloc[train_idc]
data_val = data.iloc[val_idc]
data_test = data.iloc[test_idc]
dataset_train = dataloading.ElectricTimeSeries(data_train, feats, target)
dataset_val = dataloading.ElectricTimeSeries(data_val, feats, target)
dataset_test = dataloading.ElectricTimeSeries(data_test, feats, target)
print(f'Training:   {len(dataset_train)} items')
print(f'Validation: {len(dataset_val)} items')
print(f'Test:       {len(dataset_test)} items')
print('')

# Infer feature mean and std over training set
feat_mean_train, feat_std_train = dataset_train.infer_mean_and_std()

# Set feature mean and std for training and validation and test set
dataset_train.set_mean_and_std(feat_mean_train, feat_std_train)
dataset_val.set_mean_and_std(feat_mean_train, feat_std_train)
dataset_test.set_mean_and_std(feat_mean_train, feat_std_train)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, collate_fn=dataloading.collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=False, collate_fn=dataloading.collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                             shuffle=False, collate_fn=dataloading.collate_fn)

# %%
# TESTS

# for i, batch in enumerate(dataloader_test):
#     if batch is not None:
#         X, y = batch
#         print(i, X.size(), y.size())
#     else:
#         print('wooow')

# dataset_train.__getitem__(99)

# %%
# Model, loss function, optimizer

n_epoch = 30
patience = 3
output_size = 1

print('Model creation...')
model = modeling.Forecaster(len(feats), 256, 4, output_size).to('cuda')
print(f'{modeling.count_parameters(model)} trainable parameters')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
print('')

# %%
# TESTS

# X = torch.randn((3, 72, 10)).to('cuda')
# model.eval()
# model(X)

# %%
# Training

print('Training...')
model.train_many_epochs('cuda', dataloader_train, dataloader_val, loss_fn,
                        optimizer, n_epoch, patience)
print('')

plt.plot(model.checkpoint['train_loss_history'], label='Training')
plt.plot(model.checkpoint['val_loss_history'], label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../viz/training_history.png')
plt.close()

# %%
# Evaluation

print('Performance evaluation...')

model.load_state_dict(model.checkpoint['best_model_state_dict'])

print('Evaluation on training, validation and test sets...')
train_loss = model.evaluate('cuda', dataloader_train, loss_fn)
val_loss = model.evaluate('cuda', dataloader_val, loss_fn)
test_loss = model.evaluate('cuda', dataloader_test, loss_fn)

print(f'Training loss:   {train_loss:>8f}')
print(f'Validation loss: {val_loss:>8f}')
print(f'Test loss:       {test_loss:>8f}')

# Prediction
print('Prediction on test set...')
groundtruth, prediction = model.predict('cuda', dataloader_test)
groundtruth = groundtruth.cpu()
prediction = prediction.cpu()

# Groundtruth vs prediction plot on test set
min_plt = min(min(groundtruth), min(prediction))
max_plt = max(max(groundtruth), max(prediction))
plt.scatter(groundtruth, prediction)
plt.plot([min_plt, max_plt], [min_plt, max_plt], color='red')
plt.xlabel('Grountruth')
plt.ylabel('Prediction')
plt.xlim([min_plt, max_plt])
plt.ylim([min_plt, max_plt])
plt.savefig('../viz/groundtruth_vs_prediction.png')
plt.close()

# Performance on test set
r2 = r2_score(groundtruth, prediction)
nmae = torch.mean(torch.abs(groundtruth - prediction)) \
    / torch.mean(torch.abs(groundtruth))
print('R2 score:', r2)
print('NMAE:', nmae.item())

# %%