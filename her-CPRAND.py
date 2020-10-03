pip install -U tensorly
import numpy as np
import tensorly as tl

# stopping criteria 
def err(tensor,weight,factors): 
  t_tilde=tl.kruskal_to_tensor((weight,factors)) # transform tensor decomposition (kruskal tensor) to tensor
  return(tl.norm(tensor-t_tilde))
  
# ALS method to compute tensor decomposition
def als(tensor,rank,it_max=100,tol=1e-5):
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor
  factors=[] # list of factor matrices
  # Initializtion of factor matrices by left singular vectors
  for mode in range(N):
    unfolded=tl.unfold(tensor, mode)
    if rank<=tl.shape(tensor)[mode] : 
      u,s,v=tl.partial_svd(unfolded,n_eigenvecs=rank) # first rank eigenvectors/values (ascendent)
    else : 
      u,s,v=tl.partial_svd(unfolded,n_eigenvecs=N) 
      u=np.append(u,np.random.random((np.shape(u)[0],rank-N)),axis=1)  # singular matrix ? 
    factors+=[u]
  weights,factors=tl.kruskal_tensor.kruskal_normalise((None,factors)) # normalise factor matrices
  it=0
  while ((err(tensor,weights,factors)/norm_tensor)>tol and it<it_max): 
    for n in range(N):
      V=np.ones((rank,rank))
      for i in range(len(factors)):
        if i != n : V=V*tl.dot(tl.transpose(factors[i]),factors[i])
      W=tl.kruskal_tensor.unfolding_dot_khatri_rao(tensor, (None,factors), n) # do I need to reverse factors ?
      factors[n]= tl.transpose(tl.solve(tl.transpose(V),tl.transpose(W)))
    weights,factors=tl.kruskal_tensor.kruskal_normalise((None,factors))
    it=it+1
  return(weights,factors)
