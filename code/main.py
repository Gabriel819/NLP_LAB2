import torch
import torch.nn as nn
import torch.optim as optim
from my_model import MyModel
from my_dataset import MyDataset
from util import get_ap_score
import torch.nn.functional as F
from util import load_data, tokenization, lemmatization, char_onehot, make_unique_char_dic
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

N_s = 20 # max length of sentence
N_w = 10 # max length of word
D_c = 100 # dimension of character vector

########## load data ##########
tr_sents, tr_labels = load_data(filepath='../data/sent_class.train.csv') # train: 4500
ts_sents, ts_labels = load_data(filepath='../data/sent_class.test.csv') # test: 500

# check the unique category value labels
category_list = list(set(tr_labels))
num_category = len(category_list)
# print(category_list)
# print(num_category)

########## Step 1. Tokenize the input sentence [1pt] ##########
# tokenization
tr_tokens = tokenization(tr_sents, N_s)
ts_tokens = tokenization(ts_sents, N_s)

########## Step 2. Lemmatize the tokenized words [1pt] ##########
# if tr_lemmas & ts_lemmas already exist
# if tr_lemmas & ts_lemmas doesn't exist, comment out this cell
# with open('./tr_lemmas.pickle','rb') as f:
#     tr_lemmas = pickle.load(f)
# with open('./ts_lemmas.pickle','rb') as f:
#     ts_lemmas = pickle.load(f)

# lemmatization
tr_lemmas = lemmatization(tr_tokens)
ts_lemmas = lemmatization(ts_tokens)

########## Step 3. Word Representation using Character Embedding [5pts] ##########
# extract unique character from tr_lemmas
unique_char_dict = make_unique_char_dic(tr_lemmas)
len_unique = len(unique_char_dict)

# the number of unique character in training dataset is 52, including padding 'P' and unknown 'U'
# print(unique_char_dict)
# print(len_unique)

# if tr_char_onehot & ts_char_onehot already exist
# if doens't exist, make this block as comment.
# with open('./tr_char_onehot.pickle','rb') as f:
#     tr_char_onehot = pickle.load(f)
# with open('./ts_char_onehot.pickle','rb') as f:
#     ts_char_onehot = pickle.load(f)

# character one-hot representation
tr_char_onehot = char_onehot(tr_lemmas, unique_char_dict, N_w)
ts_char_onehot = char_onehot(ts_lemmas, unique_char_dict, N_w)

# store tr_char_onehot and ts_char_onehot because making each char_onehot takes long
# with open('./tr_char_onehot.pickle','wb') as fw:
#     pickle.dump(tr_char_onehot, fw)
# with open('./ts_char_onehot.pickle','wb') as fw:
#     pickle.dump(ts_char_onehot, fw)

## Character Embedding Vector ##

# He initialized character embedding vector
char_embed_vec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(len_unique, D_c)))
# print(char_embed_vec)
# print(char_embed_vec.shape)

########## Step 4. Train your sentence classificaation model [3pts]
model = MyModel(char_embed_vec, N_w, D_c, num_category).to(device)

exp_num = 6 # set the number of this experiment
lr = 0.001
num_epochs = 20
batch_size = 32
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

### Prepare Dataset ###
my_dataset = MyDataset(tr_char_onehot, tr_labels)

train_size = int(0.9 * len(my_dataset)) # train dataset's size is 0.9 * total_labeled_dataset
valid_size = len(my_dataset) - train_size # valid dataset's size is 0.1 * total_labeled_dataset

# randomly choose data from total dataset to put in train_dataset or valid_dataset
train_dataset, valid_dataset = torch.utils.data.random_split(my_dataset, [train_size, valid_size])

# shuffle train dataset but not shuffle valid dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

# check the content of char_embed_vec before training
char_embed_vec = char_embed_vec.to(device)
# print(char_embed_vec)

############################### Real Training & Validation Epochs ###################################
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    model = model.to(device)

    ############### Training Phase #############
    for idx, (tr_data, tr_label) in enumerate(train_loader):
        # tr_data: (32, 20, 10, 52), tr_label: 32
        tr_data = tr_data.to(device)

        # tr_data is the multiplication of char_onehot and char_embed_vec
        # this is calculated in the training iteration to reflecting the changing char_embed_vec
        tr_data = torch.matmul(tr_data, char_embed_vec)
        tr_data = (tr_data.flatten(2, 3).unsqueeze(dim=1))  # tr_data: (32, 1, 20, 1000)

        tr_label = tr_label.to(device)

        model.train()
        optimizer.zero_grad()

        tr_output = model(tr_data)

        tr_loss = criterion(tr_output, tr_label)
        tr_loss.backward()
        optimizer.step()

        train_loss += tr_loss.item()
        train_acc += get_ap_score(torch.Tensor.cpu(F.one_hot(tr_label, num_classes=num_category)).detach().numpy(),
                                  torch.Tensor.cpu(tr_output).detach().numpy())

    train_num_samples = float(len(train_loader.dataset))
    tr_loss_ = train_loss / train_num_samples
    tr_acc_ = train_acc / train_num_samples

    train_loss_list.append(tr_loss_)
    train_acc_list.append(tr_acc_)

    ############### Evaluation Phase #############
    for idx, (val_data, val_label) in enumerate(valid_loader):
        val_data = val_data.to(device)
        val_data = torch.matmul(val_data, char_embed_vec)
        val_data = (val_data.flatten(2, 3).unsqueeze(dim=1)).to(device)  # val_data: (32, 1, 20, 1000)

        val_label = val_label.to(device)

        model.eval()

        vl_output = model(val_data)

        vl_loss = criterion(vl_output, val_label)

        val_loss += vl_loss.item()
        val_acc += get_ap_score(torch.Tensor.cpu(F.one_hot(val_label, num_classes=num_category)).detach().numpy(),
                                torch.Tensor.cpu(vl_output).detach().numpy())

    valid_num_samples = float(len(valid_loader.dataset))
    val_loss_ = val_loss / valid_num_samples
    val_acc_ = val_acc / valid_num_samples

    val_loss_list.append(val_loss_)
    val_acc_list.append(val_acc_)

    print('\nEpoch {}, train_loss: {:.4f}, train_acc:{:.3f}, valid_loss: {:.4f}, valid_acc:{:.3f}'.format(epoch, tr_loss_,
                                                                                                        tr_acc_,
                                                                                                        val_loss_,
                                                                                                        val_acc_))

    # if this epoch's model's validation accuracy is better than before, store the model parameter
    if val_acc_ > best_val_acc:
        best_val_acc = val_acc_
        torch.save(model.state_dict(), f'../LAB2_parameters/model_{exp_num}.pth')
        print(f'Epoch {epoch} model saved')

# check the content of char_embed_vec after training
# print(char_embed_vec) # can see character embedding vector has changed automatically

######## Visualize train, valid loss and accuracy #############
epoch_list = list(range(num_epochs))

plt.plot(epoch_list, train_loss_list, 'r', label='train loss')
plt.plot(epoch_list, val_loss_list, 'b', label='valid loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
# plt.show()
plt.savefig(f'../LAB2_submissions/exp{exp_num}_sch_bn_loss_graph.png') # store the loss figure in png file

plt.plot(epoch_list, train_acc_list, 'r', label='train acc')
plt.plot(epoch_list, val_acc_list, 'b', label='valid acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()
plt.savefig(f'../LAB2_submissions/exp{exp_num}_acc_graph.png') # store the accuracy figure in png file

########################## Submission ###############################
word_repr = torch.matmul(ts_char_onehot.to(device), char_embed_vec) # word_repr: (500, 20, 10, 100)
word_repr = (word_repr.flatten(2,3).unsqueeze(dim=1)) # word_repr: (500, 1, 20, 1000)

test_y = model(word_repr) # get the trained model's output of the testset

final_pred = test_y.argmax(dim=1) # get the biggest score of each row, which is the final predicted class category
# print(final_pred.shape)
# print(final_pred)

submission = ['id,pred\n']
f2 = '../data/sent_class.pred.csv'

with open(f2, 'rb') as f:
    file = f.read().decode('utf-8')
    content = file.split('\n')[:-1] # column name

    for idx, line in enumerate(content):
        if idx == 0: # first line is id, pred so just skip it
            continue
        tmp1 = line.split(',') # split the id and prediction result by ,
        res = final_pred[idx-1].item() # get the final prediction result of this id
        tmp2 = tmp1[0] + ',' + str(res) + '\n'
        submission.append(tmp2)

with open(f'../LAB2_submissions/20214047_lab2_submission{exp_num}.csv', 'w') as f:
    f.write(''.join(submission)) # store the submission file
