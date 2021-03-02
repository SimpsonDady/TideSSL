#!/usr/bin/env python
# coding: utf-8

# In[1]:


from library_data import *
from library_models import *


# In[2]:


import pandas as pd


# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
# In[4]:


# INITIALIZE PARAMETERS

class Args:
    def __init__(self):
        self.network = "lastfm"
        self.train_proportion = 0.8
        self.datapath = "data/%s.csv" % self.network
        self.model = "jodie"
        self.embedding_dim = 128
        self.epochs = 49
        self.state_change = False
        
args = Args()
'''

# In[ ]:


# INITIALIZE PARAMETERS 
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default="0", type=str, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=50, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 
args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "mooc":
    print("No interaction prediction for %s" % args.network)
    sys.exit(0)


# In[ ]:


output_fname = "results/interaction_prediction_%s.txt" % args.network
if os.path.exists(output_fname):
    f = open(output_fname, "r")
    search_string = 'Test performance of epoch %d' % args.epoch
    for l in f:
        l = l.strip()
        if search_string in l:
            print("Output file already has results of epoch %d" % args.epoch)
            sys.exit(0)
    f.close()


# In[5]:


tbatch_table = pd.read_csv("results/batches_lastfm1.txt", header=0)
tbatch_table


# In[20]:


user_sequence_id = tbatch_table["user_id"].to_numpy()
user_timediffs_sequence = tbatch_table["user_timediffs"].to_numpy()
user_previous_itemid_sequence = tbatch_table["previous_items"].to_numpy()
item_sequence_id = tbatch_table["item_id"].to_numpy()
item_timediffs_sequence = tbatch_table["item_timediffs"].to_numpy()
timestamp_sequence = tbatch_table["timestamp"].to_numpy()
feature_sequence = tbatch_table[tbatch_table.columns[9:]].to_numpy()


# In[6]:


num_interactions = len(tbatch_table["interactionids"])
num_users = len(set(tbatch_table["user_id"])) 
num_items = len(set(tbatch_table["item_id"])) + 1 # one extra item for "none-of-these"
num_features = len(tbatch_table.iloc[0]) - 9


# In[7]:


# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))


# In[8]:


timespan = tbatch_table["timestamp"].iloc[-1] - tbatch_table["timestamp"][0]
tbatch_timespan = timespan / 500 


# In[9]:


# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
crossEntropyLoss = nn.CrossEntropyLoss()
MSELoss = nn.MSELoss()
binaryCrossEntropyLoss = nn.BCELoss()


# In[10]:


# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


# In[11]:


# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)
if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')


# In[16]:


# SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx)


# In[17]:


# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
item_embeddings = item_embeddings.clone()
item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
item_embeddings_static = item_embeddings_static.clone()

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
user_embeddings_static = user_embeddings_static.clone()


# In[18]:


# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []


# In[21]:


tbatch_start_time = None
loss = 0
# FORWARD PASS
print("*** Making interaction predictions by forward pass (no t-batching) ***")
with trange(train_end_idx, test_end_idx) as progress_bar:
    for j in progress_bar:
        progress_bar.set_description('%dth interaction for validation and testing' % j)

        # LOAD INTERACTION J
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
        item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
        item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
        feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)

        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())
        
        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1) 
        
        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        if j < test_start_idx:
            validation_ranks.append(true_item_rank)
        else:
            test_ranks.append(true_item_rank)

        # UPDATE USER AND ITEM EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update') 

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

        # CALCULATE STATE CHANGE LOSS
        if args.state_change:
            loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss) 

        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_()


# In[22]:


# CALCULATE THE PERFORMANCE METRICS
performance_dict = dict()
ranks = validation_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
performance_dict['validation'] = [mrr, rec10]


# In[23]:


ranks = test_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
performance_dict['test'] = [mrr, rec10]


# In[24]:


# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
metrics = ['Mean Reciprocal Rank', 'Recall@10']


# In[ ]:


print('\n\n*** Validation performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Validation performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
    fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")


# In[ ]:


print('\n\n*** Test performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")


# In[ ]:


fw.flush()
fw.close()


# In[ ]:




