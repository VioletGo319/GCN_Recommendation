import numpy as np
user = ('u1', 'u2')
item = ('i1', 'i2', 'i3', 'i4')

edges = [('u1', 'i1'), ('u1', 'i3'), ('u2', 'i2'), ('u2', 'i3'), ('u2', 'i4')]

# actual data

nodes = user + item

udx_user = len(user)
udx_item = udx_user+1
n = len(nodes)


A = np.zeros((n, n))
for i, j in edges:
    u_idx = nodes.index(i)
    v_idx = nodes.index(j)
    A[u_idx,v_idx] = 1
    A[v_idx,u_idx] = 1


degrees = np.sum(A, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

A_hat = D_inv_sqrt @ A @ D_inv_sqrt

# define initial embeddings
embedding_dim = 10

E0 = np.random.rand(n, embedding_dim) # this part (GEE instead of random init)
E1 = A_hat @ E0 
E2 = A_hat @ E1
E3 = A_hat @ E2

final_embeddings = (E0+ E1 + E2 + E3) / (4) # simple average (weighted) (new type of pooling)


# loss calculation (sampling ) - how exactly does this work? this part ()
def bpr_loss(u, pos, neg, embeddings, nodes):
    u_idx = nodes.index(u)
    pos_idx = nodes.index(pos)
    neg_idx = nodes.index(neg)
    
    u_emb = embeddings[u_idx]
    pos_emb = embeddings[pos_idx]
    neg_emb = embeddings[neg_idx]
    
    pos_score = np.dot(u_emb, pos_emb)
    neg_score = np.dot(u_emb, neg_emb)
    
    loss = -np.log(1 / (1 + np.exp(neg_score - pos_score)))
    return loss


triplets = [('u1', 'i1', 'i2'), ('u2', 'i2', 'i1')]
losses = [bpr_loss(u, p, n, final_embeddings, nodes) for (u, p, n) in triplets]
total_loss = np.sum(losses)

print(f"Total Loss: {total_loss}")


# ver 1 - use same conditions to compare new init - compare performance
# ver 2 - use same conditions to compare new init, loss - comare performance
# ver 3 - use same conditions to compare new init, loss, pooling - compare performance


    