#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


# In[3]:


DATASET_PATH = r"archive\covid.csv"
from itertools import chain, combinations
RESULTS_FILE = "results\\results_{}_{}.csv"

final_outcomes = ['patient_type',
                 'intubed',
                 'pneumonia',
                 'icu',
                'date_died'
                 ]


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self, input_dim, output_dim=5):
        super(Classifier, self).__init__()
        # parameters
        self.input_dim = input_dim
        self.hidden_dim = 7
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        ones = torch.ones(self.output_dim)
        return x + ones


# In[5]:


class Classifier_no(nn.Module):

    def __init__(self):
        super(Classifier_no, self).__init__()
        # parameters
        self.fc1 = nn.Linear(9, 5)
        # self.fc2 = nn.Linear(7, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.sigmoid(x)
        ones = torch.ones(5)
        return x + ones


# In[6]:


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.labels = labels # torch.Tensor()# .to(dtype=torch.long) #.type(torch.LongTensor)
        self.features = features # torch.Tensor()# .to(dtype=torch.long) #.type(torch.LongTensor)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        feature = (self.features[index])

        # Load data and get label
        y = (self.labels[index])

        return torch.Tensor(feature), torch.Tensor(y)
  
  @staticmethod
  def collate_fn(batch):
      batch_features = []
      batch_y = []
      for feature, y in batch:
          batch_features.append(feature)
          batch_y.append(y)
        
      return (torch.stack(batch_features), 
              torch.stack(batch_y))


# In[7]:


def write_in_all_csvs(csv_writers: list, row):
    for writer in csv_writers:
        writer.writerow(row)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def calc_prediction_error(pred_vector, gt_vector):
    assert len(pred_vector) == len(gt_vector)
    error_vec = np.zeros(len(gt_vector))
    for indx in range(len(pred_vector)):
        error_vec[indx] = abs(pred_vector[indx] - gt_vector[indx])**2
    return error_vec


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# In[8]:


df = pd.read_csv(DATASET_PATH)
df = df.loc[df['covid_res'] == 1]  # take only positives
df = df.drop('id', axis=1)  # drop id

df.loc[df.date_died == "9999-99-99", 'date_died'] = 2
df.loc[df.date_died != 2, 'date_died'] = 1
df.loc[df.intubed == 97, 'intubed'] = 2
df.loc[df.icu == 97, 'icu'] = 2
df = df.loc[(df["age"] <= 40) & (df['age'] >= 18)]
df = df.loc[(df["patient_type"].isin([1,2])) &
            (df['intubed'].isin([1,2])) &
            (df['icu'].isin([1,2])) &
            (df['date_died'].isin([1,2])) &
            (df['pneumonia'].isin([1,2]))
            #(df['pregnant'].isin([1,2]))
            #(df['sex'] == 1)
            ]
df = df.drop('sex', axis=1)  # drop sex
df = df.drop('pregnancy', axis=1)  # drop sex
features = [#'pregnancy',
            'diabetes', 'copd', 'asthma',
            'inmsupr', 'hypertension',
            'cardiovascular', 'obesity', 'renal_chronic',
            'tobacco' #,
            # 'sex'
            ]

selected_feature = "asthma"



features.remove(selected_feature)
print(f"feature: {features}")
print(f"outcomes: {final_outcomes}")
df_yes_feature = df.loc[df[selected_feature] == 1]
df_no_feature =  df.loc[df[selected_feature] == 2]

print("making dataset")

batch_size = 64

features_yes = df_yes_feature[features].to_numpy() # torch.tensor(train_yes[features].to_numpy(), dtype=torch.long) #  train_yes[features].to_numpy() # (dtype=torch.long)
labels_yes = df_yes_feature[final_outcomes].to_numpy() # torch.tensor(train_yes[final_outcomes].to_numpy(), dtype=torch.long) # .to(dtype=torch.long)
yes_dataset = Dataset(features_yes, labels_yes)
yes_dataloader = DataLoader(dataset=yes_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=Dataset.collate_fn)
features_no = df_no_feature[features].to_numpy() # torch.tensor(train_yes[features].to_numpy(), dtype=torch.long) #  train_yes[features].to_numpy() # (dtype=torch.long)
labels_no = df_no_feature[final_outcomes].to_numpy() # torch.tensor(train_yes[final_outcomes].to_numpy(), dtype=torch.long) # .to(dtype=torch.long)
no_dataset = Dataset(features_no, labels_no)
no_dataloader = DataLoader(dataset=no_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=Dataset.collate_fn)

input_dim = len(features)


# In[9]:


import torch.optim as optim

# classifier for the yes: 
model_yes = Classifier(input_dim)

criterion_yes = nn.MSELoss()
optimizer_yes = optim.SGD(model_yes.parameters(), lr=0.0001, momentum=0.8)

total_examples = 0 
num_correct = 0 
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(yes_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer_yes.zero_grad()
        # forward + backward + optimize
        outputs = model_yes(inputs)
        loss = criterion_yes(outputs, labels)
        loss.backward()
        optimizer_yes.step()

        # print statistics
        running_loss += loss.item() # print every 2000 mini-batches
    print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # accuracy = 100 * num_correct / total_examples 
            # print(f"Current Accuracy is: {accuracy:.3f}")

# accuracy = 100 * num_correct / total_examples 
# print(f"Total Accuracy is: {accuracy:.3f}")
print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
print('Finished Training')


# In[10]:


model_no = Classifier(input_dim)

criterion_no = nn.MSELoss()
optimizer_no = optim.SGD(model_no.parameters(), lr=0.01, momentum=0.9)

for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(no_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer_no.zero_grad()

        # forward + backward + optimize
        outputs = model_no(inputs)
        loss = criterion_no(outputs, labels)
        loss.backward()
        optimizer_no.step()
        # print statistics
        running_loss += loss.item()  # print every 2000 mini-batches
    print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))

print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
print('Finished Training')


# In[86]:


def intify(x):
    x = x.detach().numpy()
    for i, item in np.ndenumerate(x):
        if item >= 1.5:
            x[i] = 2
        else:
            x[i] = 1
    return x


# In[11]:


num_features = len(features)
datasize = np.power(2, num_features)
dataset = np.zeros((datasize, num_features))
syn_data_no = {}
syn_data_yes = {}
ATEs = torch.zeros(5)
cnt = 0 
for i, subset in enumerate(powerset(features)):
    items = features.copy()
    items = np.array([1 if item in subset else 2 for item in items]).reshape(1, -1)
    dataset[i] = items
    input_feat = torch.tensor(items).float()
    outcome_no = model_no(input_feat)
    syn_data_no[i] = outcome_no
    outcome_yes = model_yes(input_feat)
    syn_data_yes[i] = outcome_yes
    ITE = outcome_yes - outcome_no
    ATEs += ITE.view(-1)
    cnt += 1
ATE = (ATEs.detach()).numpy()
ATE = np.divide(ATE, cnt)
print(ATE)    


# In[12]:


def calc_error(row, row_not):
    err = (row != row_not).sum()
    return err

def search_for_closest(my_feature, all_features, K):

    lst_errors = []
    for index, row_not_ft in enumerate(all_features):
        err = calc_error(my_feature, row_not_ft)
        if len(lst_errors) < K:
            lst_errors.append([err, index])
            lst_errors.sort(key=lambda x: x[0])
        else:
            if err >= lst_errors[-1][0]:
                continue
            lst_errors.pop(len(lst_errors)-1)
            lst_errors.append([err, index])
            lst_errors.sort(key=lambda x: x[0])
    return lst_errors
K = 15 
ATE  = np.zeros(5)
cnt = 0 
for i, subset in enumerate(dataset):
    my_closest = search_for_closest(subset, dataset, K)
    ITE_nei = np.zeros(5)
    for err, index in my_closest:
        ITE_nei += (syn_data_no[index].detach()).numpy().reshape(-1)
    ITE = np.divide(ITE_nei, K)
    ATE += syn_data_yes[i].detach().numpy().reshape(-1) - ITE
    cnt +=1 
ATE = np.divide(ATE, cnt)
print(ATE)


# In[182]:


TODO:
1) To feed both networks with all possibiliteis of feature input.
2) Create the synthetic data: 2*2^(num of features) outputs (one for pregnant and one for regular)
    shape of a synthetic datum: (9, 5, 1) 9 features; 5 outputs, 1 label
    Best way for the syntethic data is 2 dict where key is features and value is outputs
    pregnant dict : {feature : output}
    not pregnant dict : {feature : output}
3) Apply 1NN on the synthetic data:
    for each possibility of feature input f, we have m1(f) and m2(f).
    calculate the ITE: m1(f) - m2(f) (where m1 is the pregnant women model) 
    sum and avarage all the ITEs in order to get the ATE.
3) Compute ATE as before:
    for each possibility of feature input f, calculate the ITE by avaraging the KNN
    ITE = m1(f) - 1/K * Sum (m2(fi)) where fi are the KNN of f in the synthetic data.


# In[ ]:


goal_diff = np.zeros(len(final_outcomes))
cnt = 0
patient_type_results_file = open(RESULTS_FILE.format(selected_feature, "patient_type"), 'w', newline='')
intubed_results_file = open(RESULTS_FILE.format(selected_feature, "intubed"), 'w', newline='')
pneumonia_results_file = open(RESULTS_FILE.format(selected_feature, "pneumonia"), 'w', newline='')
icu_results_file = open(RESULTS_FILE.format(selected_feature, "icu"), 'w', newline='')
died_results_file = open(RESULTS_FILE.format(selected_feature, "died"), 'w', newline='')
patient_type_writer = csv.writer(patient_type_results_file)
intubed_writer = csv.writer(intubed_results_file)
pneumonia_writer = csv.writer(pneumonia_results_file)
icu_writer = csv.writer(icu_results_file)
died_writer = csv.writer(died_results_file)
all_writers = [patient_type_writer,
               intubed_writer,
               pneumonia_writer,
               icu_writer,
               died_writer,
               ]
write_in_all_csvs(all_writers, features + ["Smoker", "Non-smoker", "diff"])

for subset in powerset(features):
    items = features.copy()
    items = np.array([1 if item in subset else 2 for item in items]).reshape(1, -1)
    yes_result = logic_reg_yes.predict(items)[0]
    no_result = logic_reg_no.predict(items)[0]
    items = [True if i==1 else False for i in list(items[0])]
    patient_type_writer.writerow(items + [yes_result[0], no_result[0], yes_result[0]-no_result[0]])
    intubed_writer.writerow(items + [yes_result[1], no_result[1], yes_result[1]-no_result[1]])
    pneumonia_writer.writerow(items + [yes_result[2], no_result[2], yes_result[2]-no_result[2]])
    icu_writer.writerow(items + [yes_result[3], no_result[3], yes_result[3]-no_result[3]])
    died_writer.writerow(items + [yes_result[4], no_result[4], yes_result[4]-no_result[4]])

    goal_diff = np.add(np.subtract(yes_result, no_result), goal_diff)
    cnt += 1
goal_diff = np.divide(goal_diff, cnt)
print(f"---FINAL: { goal_diff}")


if __name__ == "__main__":
    main(final_outcomes)



