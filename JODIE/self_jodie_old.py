#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import time


# In[2]:


from library_data import *
import library_models as lib
from library_models import *


# In[3]:


import pandas as pd


# In[4]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[5]:


def softmax_cross_entropy(standard_tensor, self_tensor):
#     print(len(standard_tensor))
    cos = nn.CosineSimilarity()
    celoss = 0
    for i, st in enumerate(standard_tensor):
        denominator = molecular = 0
        for j, se in enumerate(self_tensor):
            if i == j:
                molecular = torch.exp(cos(st.reshape(1, -1), se.reshape(1, -1)) / 0.07)
#                 print(cos(st.reshape(1, -1), se.reshape(1, -1)).item(), end=' ')
#                 print("Hi", cos(st.reshape(1, -1), se.reshape(1, -1)).item(), torch.exp(cos(st.reshape(1, -1), se.reshape(1, -1))).item())
#                 continue
            denominator += torch.exp(cos(st.reshape(1, -1), se.reshape(1, -1)) / 0.07)
#             print(cos(st.reshape(1, -1), se.reshape(1, -1)).item(), end=' ')
#             print("Ho", cos(st.reshape(1, -1), se.reshape(1, -1)).item(), torch.exp(cos(st.reshape(1, -1), se.reshape(1, -1))).item())
        celoss += torch.log(molecular / denominator)[0]
#         print("\nlog(", molecular.item(), "/", denominator.item(), ") =", torch.log(molecular / denominator)[0].item())
#     print(celoss.item(), -(1 / len(standard_tensor)) * celoss.item())
#     print("=========================")
    if -(1 / len(standard_tensor)) * celoss.item() < 0:
        sys.exit()
    return -(1 / len(standard_tensor)) * celoss


# In[6]:


# INITIALIZE PARAMETERS
class Args:
    def __init__(self):
        self.network = "lastfm"
        self.train_proportion = 0.8
        self.datapath = "data/%s.csv" % self.network
        self.model = "jodie"
        self.embedding_dim = 128
        self.epochs = 50
        self.state_change = False
        
args = Args()


# In[9]:


tbatch_table = pd.read_csv("results/batches_lastfm1.txt", header=0)


# In[ ]:


num_interactions = len(tbatch_table["interactionids"])
num_users = len(set(tbatch_table["user_id"])) 
num_items = len(set(tbatch_table["item_id"])) + 1 # one extra item for "none-of-these"
num_features = len(tbatch_table.iloc[0]) - 9


# In[ ]:


# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))


# In[ ]:


timespan = tbatch_table["timestamp"].iloc[-1] - tbatch_table["timestamp"][0]
tbatch_timespan = timespan / 500 


# In[ ]:


# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
crossEntropyLoss = nn.CrossEntropyLoss()
MSELoss = nn.MSELoss()
binaryCrossEntropyLoss = nn.BCELoss()


# In[ ]:


# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding


# In[ ]:


user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

# In[ ]:


self_user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
self_item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding


# In[ ]:


# INITIALIZE MODEL
learning_rate = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


# In[ ]:


# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}


# In[ ]:


with trange(args.epochs) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        self_user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        self_item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        project_timediff = torch.Tensor([5]).cuda()

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count, self_total_interaction_count = 0, 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j) 
                current_batch_data = tbatch_table[tbatch_table["tbatch_id"] == j+1]
                drop_indices = np.random.choice(current_batch_data.index, int(len(current_batch_data)*0.3), replace=False)
                if len(current_batch_data) < 2:
                    continue
                
                batch_user = list(set(current_batch_data["user_id"]))
                batch_item = list(set(current_batch_data["item_id"]))
                
                user_embeddings_projections = Variable(torch.zeros(len(batch_user), args.embedding_dim).cuda())
                item_embeddings_projections = Variable(torch.zeros(len(batch_item), args.embedding_dim).cuda())
                self_user_embeddings_projections = Variable(torch.zeros(len(batch_user), args.embedding_dim).cuda())
                self_item_embeddings_projections = Variable(torch.zeros(len(batch_item), args.embedding_dim).cuda())

                # Standard Forward Propagations
                for i in range(len(current_batch_data)):
                    total_interaction_count += len(current_batch_data)

                    tbatch_userids = torch.LongTensor(current_batch_data["user_id"].values[i:i+1]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                    tbatch_itemids = torch.LongTensor(current_batch_data["item_id"].values[i:i+1]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_interactionids = torch.LongTensor(current_batch_data["interactionids"].values[i:i+1]).cuda()
                    feature_tensor = Variable(torch.Tensor(tbatch_table[tbatch_table.columns[9:]].values[i:i+1]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_timediffs_tensor = Variable(torch.Tensor(current_batch_data["user_timediffs"].values[i:i+1]).cuda().unsqueeze(1))
                    item_timediffs_tensor = Variable(torch.Tensor(current_batch_data["item_timediffs"].values[i:i+1]).cuda().unsqueeze(1))
                    tbatch_itemids_previous = torch.LongTensor(current_batch_data["previous_items"].values[i:i+1]).cuda()
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    
                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                    item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                    item_embeddings[tbatch_itemids,:] = item_embedding_output
                    user_embeddings[tbatch_userids,:] = user_embedding_output  
 
                    user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                    item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output
                    
                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                    loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                    loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                    
                    # Self-supervised Forward Propagations
                    if i not in drop_indices:
                        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                        self_item_embeddings[tbatch_itemids,:] = item_embedding_output
                        self_user_embeddings[tbatch_userids,:] = user_embedding_output  

                        self_user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                        self_item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                        loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                
                # PROJECT USER EMBEDDING TO CURRENT TIME
                for k in range(len(batch_user)):
                    user_embeddings_projections[k] = model.forward(user_embeddings[batch_user[k],:], None, timediffs=project_timediff, features=None, select='project')
                    self_user_embeddings_projections[k] = model.forward(self_user_embeddings[batch_user[k],:], None, timediffs=project_timediff, features=None, select='project')
                    
                # PROJECT ITEM EMBEDDING TO CURRENT TIME
                for k in range(len(batch_item)):
                    item_embeddings_projections[k] = model.forward(item_embeddings[batch_item[k],:], None, timediffs=project_timediff, features=None, select='project')
                    self_item_embeddings_projections[k] = model.forward(self_item_embeddings[batch_item[k],:], None, timediffs=project_timediff, features=None, select='project')

#                 print("loss", loss)
#                 print("user", softmax_cross_entropy(user_embeddings_projections, self_user_embeddings_projections.detach_()))
#                 print("item", softmax_cross_entropy(item_embeddings_projections, self_item_embeddings_projections.detach_()))
#                 print("-------------")
                loss += softmax_cross_entropy(user_embeddings_projections, self_user_embeddings_projections.detach_())
                loss += softmax_cross_entropy(item_embeddings_projections, self_item_embeddings_projections.detach_())
                
                # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # RESET LOSS FOR NEXT T-BATCH
                loss = 0
                item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()
                self_item_embeddings.detach_()
                self_user_embeddings.detach_()
                self_item_embeddings_timeseries.detach_() 
                self_user_embeddings_timeseries.detach_()

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)


# In[ ]:


# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




tbatch_id,user_id,item_id,timestamp,state_label,comma_separated_list_of_features
# In[ ]:


[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)


# In[ ]:




