import numpy as np
import random
import torch
class SumTree:
    '''
    Binary tree where the leaves store the priorities of the experiences and the internal nodes
        stores the cumulative priorities of their children. 
    '''
    write = 0
    def __init__(self, state_dim,action_dim,max_size):
        self.max_size = max_size
        self.tree = np.zeros(2*max_size -1)
        self.s = np.zeros((max_size,state_dim),dtype=np.float32)
        self.a = np.zeros((max_size,action_dim),dtype=np.int64)
        self.r = np.zeros((max_size,1),dtype=np.float32)
        self.s_prime = np.zeros((max_size,state_dim),dtype=np.float32)
        self.terminated = np.zeros((max_size,1),dtype=np.int64)
        self.n_entries = 0
    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx-1)//2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left,s)
        else:
            return self._retrieve(right, s-self.tree[left])
    def total(self):
        return self.tree[0]
    
    # store priority and sample
    def add(self, p, s,a,r,s_prime,terminated):
        idx = self.write + self.max_size-1
        self.s[self.write] = s
        self.a[self.write] = a
        self.r[self.write] = r
        self.s_prime[self.write] = s_prime
        self.terminated[self.write] = terminated
        self.update(idx,p)

        self.write += 1
        if self.write >= self.max_size:
            self.write = 0
        if self.n_entries < self.max_size:
            self.n_entries += 1
    
    # update priority
    def update(self, idx, p):
        change = p-self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0,s)
        dataIdx = idx - self.max_size + 1
        return idx,self.tree[idx],(self.s[dataIdx],
                self.a[dataIdx],
                self.r[dataIdx],
                self.s_prime[dataIdx],
                self.terminated[dataIdx])
    
class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    def __init__(self, state_dim,action_dim,max_size):
        self.tree = SumTree(state_dim, action_dim,max_size)
        self.max_size = max_size
    def _get_priority(self, error):
        return (np.abs(error)+self.e)**self.a
    def update(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, *sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim,max_size=int(1e5)):
        self.s = np.zeros((max_size,state_dim),dtype=np.float32)
        self.a = np.zeros((max_size,action_dim),dtype=np.int64)
        self.r = np.zeros((max_size,1),dtype=np.float32)
        self.s_prime = np.zeros((max_size,state_dim),dtype=np.float32)
        self.terminated = np.zeros((max_size,1),dtype=np.float32)
        #self.prev_state_type = np.zeros((max_size,1),dtype=np.int64)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self,s,a,r,s_prime,terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated
        #self.prev_state_type[self.ptr] = prev_state_type

        self.ptr = (self.ptr+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
            #torch.FloatTensor(self.prev_state_type[ind])
        )