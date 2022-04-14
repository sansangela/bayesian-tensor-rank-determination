import torch
import torch.nn as nn
import numpy as np
from tensor_layers.layers import TensorizedEmbedding, Q_TensorizedEmbedding

import warnings
warnings.filterwarnings("ignore")

learning_rate = 1e-1
# p_shapes = [200, 220, 250] # n = num_embeddings = voc_size
# q_shapes = [4, 4, 8]    # m = embedding_dim = emb_size
# max_rank = 16
p_shapes = [100, 51,55]
q_shapes = [2,4, 4]
max_rank = 10
# device = torch.device("cuda:1")
# torch.cuda.set_device(device)
device = 'cpu'
num_embeddings = np.prod(np.array(p_shapes))
embedding_dim = np.prod(np.array(q_shapes))

# EE = TensorizedEmbedding(
#     tensor_type='TensorTrainMatrix',
#     max_rank=max_rank,
#     shape=[p_shapes, q_shapes],
#     prior_type='log_uniform',
#     eta=None,
# )

Q_EE = Q_TensorizedEmbedding(
    tensor_type='TensorTrainMatrix',
    max_rank=16,
    shape=[p_shapes, q_shapes],
    prior_type='log_uniform',
    eta=None,
    bit_w = 4
)
Q_EE.to(device)

standard_emb = torch.nn.EmbeddingBag(num_embeddings=np.prod(p_shapes),
                                     embedding_dim=np.prod(q_shapes),
                                     include_last_offset=True,
                                     mode='sum')
standard_emb.to(device)

from tensor_layers.low_rank_tensors import TensorTrainMatrix

# ttm = TensorTrainMatrix(dims=[EE.shape[0], EE.shape[1]],
#                         max_rank=max_rank,
#                         prior_type='log_uniform',
#                         learned_scale=False,
#                         em_stepsize=1.0)
ttm = TensorTrainMatrix(dims=[Q_EE.shape[0], Q_EE.shape[1]],
                        max_rank=max_rank,
                        prior_type='log_uniform',
                        learned_scale=False,
                        em_stepsize=1.0)
ttm.to(device)
standard_emb.weight.data.copy_(ttm.get_full().detach()/100.0)

def get_batch():

    batch_size = np.prod(p_shapes)-1

    batch = torch.tensor(np.arange(0, batch_size), dtype=torch.int64)
    batch = batch.to(device)

    offsets_idx = list(range(batch_size))

    offsets = torch.tensor(offsets_idx + [batch_size], dtype=torch.int64)
    offsets = offsets.to(device)
    offset_without_last = offsets[:-1]

    return batch,offsets

batch,offsets = get_batch()


# out1 = EE.forward(batch)
# out2 = Q_EE.forward(batch)
# print(out2.shape)
# acc = torch.norm(out1-out2)/torch.norm(out2)
# print("diff: ", acc)


epochs = 100
learning_rate = 1e-1

def get_loss():

    batch,offset  = get_batch()
    batch = batch.to(device)    
    true_out = standard_emb.weight[batch]#(batch,offset)
    pred_out = Q_EE(batch)

    loss = torch.sum(torch.square(true_out-pred_out))

    return loss

def get_rmse():
    true_out = standard_emb.weight[batch]#(batch,offset)
    pred_out = Q_EE(batch)#(batch,offset)
    
    rmse = torch.sum(torch.square(true_out-pred_out))/torch.sum(torch.square(true_out))
    return rmse


from allennlp.training.optimizers import DenseSparseAdam
optimizer = DenseSparseAdam([(None, x) for x in Q_EE.parameters()],
                                        lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()

    loss = get_loss()
    kl_loss = Q_EE.tensor.get_kl_divergence_to_prior()

    warmup_clip = torch.clamp(torch.tensor((epoch-50)/200),torch.tensor(0.0),torch.tensor(1.0))
    loss += warmup_clip * kl_loss

    print("epoch {}: loss={}, rmse={}".format(epoch,loss,get_rmse()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    Q_EE.tensor.update_rank_parameters()