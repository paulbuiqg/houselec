# %%
# Imports

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

import dataloading
import modeling
import preprocessing

# %%
# Preprocessing

print('Fetching data...')
data = preprocessing.fetch_data()

print('Preparing data...')
data, feats, target = preprocessing.prepare_data(data)
data

# %%
# Train-validation-test dataloaders

batch_size = 64

n = len(data)
data_train = data.iloc[: int(n * .6)]
data_val = data.iloc[int(n * .6) : int(n * .8)]
data_test = data.iloc[int(n * .8):]
dataset_train = dataloading.ElectricTimeSeries(data_train, feats, target)
dataset_val = dataloading.ElectricTimeSeries(data_val, feats, target)
dataset_test = dataloading.ElectricTimeSeries(data_test, feats, target)
print(len(data_train), len(data_val), len(data_test))

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

model = modeling.Forecaster(len(feats), 256, 4, output_size).to('cuda')
print(f'{modeling.count_parameters(model)} trainable model parameters')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

# %%
# TESTS

# X = torch.randn((3, 72, 10)).to('cuda')
# model.eval()
# model(X)

# %%
# Training

model.train_many_epochs('cuda', dataloader_train, dataloader_val, loss_fn,
                        optimizer, n_epoch, patience)

checkpoint = torch.load('checkpoint.pt')

plt.plot(checkpoint['train_loss_history'], label='Training')
plt.plot(checkpoint['val_loss_history'], label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Prediction on test set

model = modeling.Forecaster(len(feats), 256, 4, output_size).to('cuda')
model.load_state_dict(checkpoint['best_model_state_dict'])
groundtruth, prediction = model.predict('cuda', dataloader_test)

# %%
# Performance assessment

groundtruth = groundtruth.cpu()
prediction = prediction.cpu()

min_plt = min(min(groundtruth), min(prediction))
max_plt = max(max(groundtruth), max(prediction))

plt.scatter(groundtruth, prediction)
plt.plot([min_plt, max_plt], [min_plt, max_plt], color='red')
plt.xlabel('Grountruth')
plt.ylabel('Prediction')
plt.xlim([min_plt, max_plt])
plt.ylim([min_plt, max_plt])
plt.show()

r2 = r2_score(groundtruth, prediction)
nmae = torch.mean(torch.abs(groundtruth - prediction)) \
    / torch.mean(torch.abs(groundtruth))

print('R2 score:', r2)
print('NMAE:', nmae.item())

# %%
