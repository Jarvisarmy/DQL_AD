import torch
from PER import Memory
import torch.nn as nn
import numpy as np
from PER import SumTree, Memory, ReplayBuffer
import torch.nn.functional as F

# -------------------------------------------------------------------------------
# action_value
class ActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes,device,activation=torch.nn.ReLU,verbose=False):
        super(ActionValue, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.activation = activation

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim
        self.network=nn.Sequential(*layers)
        self.last_layer = nn.Linear(hidden_sizes[-1],action_dim,bias=False)
        self.tail = torch.nn.Softmax(dim=1)
        if verbose:
            print(self)
    def forward(self, x):
        x = self.network(x)
        x = self.last_layer(x)
        x = self.tail(x)
        return x


    def extract_last_hidden(self,x,train=False):
        if train:
        #x = torch.from_numpy(x).float().to(self.device)
            x = x.float().to(self.device)
            x = self.network(x)
            return x
        else:
            with torch.no_grad():
                x = x.float().to(self.device)
                x = self.network(x)
                return x

class DQN:
    def __init__(
        self,
        args,
        device,
    ):
        self.use_prioritized = args.use_prioritized
        self.lr = args.lr
        self.epsilon = args.epsilon,
        self.epsilon_min = args.epsilon_min
        self.action_dim = args.action_dim
        self.epsilon = args.epsilon
        self.epsilon_decay = (args.epsilon-args.epsilon_min)/1e4
        self.gamma = args.gamma
        self.batch_size = args.dqn_batch_size
        self.warmup_steps = args.warmup_steps
        self.target_update_interval = args.target_update_interval
        self.device = device
        self.network = ActionValue(args.state_dim, args.action_dim,args.hidden_sizes,device)
        self.target_network = ActionValue(args.state_dim, args.action_dim,args.hidden_sizes,device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(),args.lr)
        self.buffer_size = args.buffer_size
        #self.buffer = ReplayBuffer(state_dim, 1, buffer_size)
        if self.use_prioritized:
            self.buffer = Memory(args.state_dim,1,args.buffer_size)
        else:
            self.buffer = ReplayBuffer(args.state_dim,1,args.buffer_size)
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        
        self.alpha = 0.0
        self.alpha_max = 1.0
        self.alpha_step = (self.alpha_max-self.alpha)/1e4


    @torch.no_grad()
    def act(self, x, training=True):
        # take as input one state of np.ndarray and ouptut actions by following epsilon-greedy policy
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a

    def learn(self):
        self.network.train()
        if self.use_prioritized:
            batch, idxs,is_weight, = self.buffer.sample(self.batch_size)
            s,a,r,s_prime,terminated = [torch.FloatTensor(np.array(column)).to(self.device) for column in zip(*batch)]
            is_weight = torch.FloatTensor(is_weight).to(self.device)
            next_q = self.target_network(s_prime).detach()
            td_target = r + (1. - terminated)* self.gamma * next_q.max(dim=1,keepdim=True).values
            td_error = self.network(s).gather(1,a.long())-td_target
            loss = (td_error.pow(2)*is_weight).mean()
            for idx, error in zip(idxs, td_error.abs().reshape(-1).detach().cpu().numpy().tolist()):
                self.buffer.update_priorities(idx,error)
        else:
            s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
            next_q = self.target_network(s_prime).detach()
            td_target = r + (1. - terminated)* self.gamma * next_q.max(dim=1,keepdim=True).values
            loss = F.mse_loss(self.network(s).gather(1,a.long()),td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
    def process(self, transition):
        result = {}
        self.total_steps +=1
        if self.use_prioritized:
            self.buffer.update(1.0,transition)
        else:
            self.buffer.update(*transition)
        if self.total_steps > self.warmup_steps:
            result = self.learn()
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_decay)
        self.alpha = min(self.alpha_max,self.alpha+self.alpha_step)
        return result



            
