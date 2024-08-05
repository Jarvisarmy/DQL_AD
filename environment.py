from utils import StateType
from gym import spaces
import torch
from DQN import ActionValue
from data import featuresExtractor
import faiss
import random 
import numpy as np
from utils import NumpyDataset
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest

class Environment:
    def __init__(self, normal_files,
                 target_abnormal_files,
                 target_abnormal_masks,
                 args,
                 device,
                 eval=False):
        self.use_intrinsic = args.use_intrinsic
        self.use_copypaste = args.use_copypaste
        self.use_faiss = args.use_faiss

        self.normal_files = normal_files
        self.target_abnormal_files = target_abnormal_files
        self.target_abnormal_masks = target_abnormal_masks
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.action_space = spaces.Discrete(args.action_dim)
        self.prob = args.prob
        self.state = None
        self.state_type = None # either target or unknown
        self.state_idx = None
        self.max_samples = 40
        
        self.total_steps = 0
        self.current_steps = 0
        self.device = device
        self.eval = eval

        if self.use_intrinsic:
            print("fitting iForest")
            #self.iForest_update_interval=int(2e3)
            self.rng = np.random.RandomState(0)
            self.iForest_max_samples = args.iForest_max_samples
            self.iForest_batch_size = 1024
            self.iForest = IsolationForest(max_samples=self.iForest_max_samples, random_state= self.rng)
        
        

        self.epsilon_min = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = (self.epsilon-self.epsilon_min)/1e4
        
        
        self.normal_size = len(normal_files)
        self.target_size = len(target_abnormal_files)
        
        self.action_value = ActionValue(args.state_dim, args.action_dim,args.hidden_sizes,device).to(device)
    
        self.featureExtractor = featuresExtractor(normal_files,args,device)
        
        all_target_anomalies = []
        if self.use_faiss:
            self.anomalous_index = faiss.IndexFlatL2(self.state_dim)
            for file, mask in zip(self.target_abnormal_files, self.target_abnormal_masks):
                _,target_anomalies,target_masks = self.featureExtractor.get_features([file],[mask])
                target_anomalies = target_anomalies.reshape(-1,target_anomalies.shape[-1])
                target_masks = target_masks.reshape(-1)
                target_masks = target_masks > 0
                target_anomalies = target_anomalies[target_masks == 1]
                all_target_anomalies.append(target_anomalies)
            all_target_anomalies = np.concatenate(all_target_anomalies,axis=0)
            idxs = random.sample(list(range(all_target_anomalies.shape[0])),min(1000,all_target_anomalies.shape[0]))
            self.anomalous_index.add(all_target_anomalies[idxs])
            print("Is this index trained? ", self.anomalous_index.is_trained) 
            print("Total vectors in the index: ", self.anomalous_index.ntotal)
        
        #self.target_size = self.target_anomalies.shape[0]
        self.subsamples = None
        self.ano_samples = None
        self.ano_masks = None
    
    def update_subsamples(self,model):
        # update normal subsamples
        files = random.sample(self.normal_files,self.max_samples)
        sub_samples = []
        for file in files:
            _,tmp,_ = self.featureExtractor.get_features([file],None) # [B,H,W,dim]
            sub_samples.append(tmp)
        sub_samples = np.concatenate(sub_samples, axis=0)
        sub_samples = sub_samples.reshape(-1,sub_samples.shape[-1])
        idxs = random.sample(list(range(sub_samples.shape[0])),int(1e5))
        sub_samples = sub_samples[idxs]

        

        # update ano_samples and nor_samples
        idx = random.sample(list(range(self.target_size)),1)
        file = [self.target_abnormal_files[i] for i in idx] 
        mask = [self.target_abnormal_masks[i] for i in idx]
        _,samples, masks = self.featureExtractor.get_features(file,mask)
        if self.use_copypaste:
            _,aug_samples, aug_masks = self.featureExtractor.get_anomalous_features(file,mask)
            samples = np.concatenate((samples,aug_samples),axis=0)
            masks = np.concatenate((masks,aug_masks),axis=0)
        samples = samples.reshape(-1,samples.shape[-1])
        masks = masks.reshape(-1)
        self.ano_samples = samples[masks>0]
        
        #self.ano_masks = masks[masks]
        nor_samples = samples[masks == 0]
        idxs = random.sample(list(range(nor_samples.shape[0])),int(1e4))
        nor_samples = nor_samples[idxs]
        if self.use_faiss:
            #new added
            #self.anomalous_index = faiss.IndexFlatL2(self.state_dim)
            #self.anomalous_index.add(self.ano_samples)


            distances, I = self.anomalous_index.search(nor_samples,3)
            mean_distances = np.mean(distances, axis=1)
            idx = np.argsort(mean_distances)[:int(1e2)]
            nor_samples = nor_samples[idx]
            
        self.subsamples = np.concatenate((sub_samples, nor_samples),axis=0)

        # update IsolationForest
        if self.use_intrinsic:
            self.action_value.load_state_dict(model.state_dict())
            dataset = NumpyDataset(self.subsamples)
            dataloader = DataLoader(dataset,batch_size=self.iForest_batch_size)
            embeddings = []
            with torch.no_grad():
                for batch in dataloader:
                    embedding = self.action_value.extract_last_hidden(batch).cpu().numpy()
                    embeddings.extend(embedding)
            self.iForest.fit(embeddings)

    def reset(self):
        _,sub_samples,_ = self.featureExtractor.get_features(random.sample(self.normal_files,1),None) # [B,H,W,dim]
        sub_samples = sub_samples.reshape(-1,sub_samples.shape[-1])
        idx = random.sample(list(range(sub_samples.shape[0])),1)[0]
        state = sub_samples[idx]
        self.state = state
        self.state_type = StateType.UNKNOWN
        #self.total_steps = 0
        self.current_steps = 0
        return state
    def anomalies_sampling(self,model,action):
        #if random.random() < 0.8:
        idxs = random.sample(list(range(self.ano_samples.shape[0])),1)[0]
        samples = self.ano_samples[idxs]
            #masks = self.ano_masks[idxs]
        return samples, StateType.TARGET
        #else:
        #    idxs = random.sample(list(range(self.nor_samples.shape[0])),1000)
        #    samples = self.nor_samples[idxs]
        #    distances,I = self.anomalous_index.search(samples,5)
        #    mean_distances = np.mean(distances,axis=1)
        #    idx = np.argsort(mean_distances)[0]
        #    state = samples[idx]
        #    return state,0
        #embeddings = model.extract_last_hidden(samples).cpu().numpy()
        #embed = model.extract_last_hidden(torch.from_numpy(np.expand_dims(self.state,axis=0))).squeeze(0).cpu().numpy()
        #distances = np.array([np.linalg.norm(embed-n_embed) for n_embed in embeddings])
        #distances,I = self.anomalous_index.search(samples,5)
        #mean_distances = np.mean(distances,axis=1)
        #top_m_indices = np.argsort(mean_distances)[:20]
        #idx= np.random.choice(top_m_indices,1)[0]
        #state = samples[idx]
        #state_label = masks[idx]
        #return state, state_label
    def unknown_sampling(self,model,action):
        #_,sub_samples,_ = self.featureExtractor.get_features(random.sample(self.normal_files,1),None) # [B,H,W,dim]
        #sub_samples = sub_samples.reshape(-1,sub_samples.shape[-1])
        #print(self.subsamples.shape)
        #print(self.nor_samples.shape)
        #cat_samples = np.concatenate((self.subsamples, self.nor_samples),axis=0)
        idxs = random.sample(list(range(self.subsamples.shape[0])),1000)
        sub_samples = self.subsamples[idxs]
        samples = torch.from_numpy(sub_samples).to(self.device)
        embeddings = model.extract_last_hidden(samples).cpu().numpy()
        embed = model.extract_last_hidden(torch.from_numpy(np.expand_dims(self.state,axis=0))).squeeze(0).cpu().numpy()
        distances = np.array([np.linalg.norm(embed-n_embed) for n_embed in embeddings])
        #if np.random.rand() > self.epsilon:
        if action == 0:
            idx = np.argmax(distances)
            state= samples[idx]
        else:
            idx = np.argmin(distances)
            state = samples[idx]
        #else:
        #    idx = np.random.randint(0, 1000)
        #    state = samples[idx]
        return state.cpu().numpy()
    def step(self, model, action):
        self.action_value.eval()
        self.total_steps += 1
        self.current_steps += 1
        
        if not self.eval:
            if self.total_steps % int(5e2) == 0:
                print('update subsamples')
                self.update_subsamples(model)
        
        if random.random() < self.prob:
            next_state,next_state_type = self.anomalies_sampling(model,action)
        else:
            next_state = self.unknown_sampling(model,action)
            next_state_type = StateType.UNKNOWN
        if self.use_intrinsic:
            embed = self.action_value.extract_last_hidden(torch.from_numpy(np.expand_dims(self.state,axis=0))).squeeze(0).cpu().numpy()
            intrinsic_reward = self.iForest.decision_function(np.expand_dims(embed,axis=0)).squeeze()
            intrinsic_reward = 1-(intrinsic_reward+1)/2
        else:
            intrinsic_reward = 0
        if self.state_type==StateType.TARGET:
            if action == 1:
                external_reward = 1
            else:
                external_reward = -1
        elif self.state_type==StateType.UNKNOWN:
            if action == 1:
                external_reward = -2
            else:
                external_reward = 0
        else:
            print('wrong combination')

        

        prev_state_type = self.state_type
        self.state = next_state
        self.state_type = next_state_type
        self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_decay)
        if self.current_steps % 2e3 == 0:
            done = True
        else:
            done = False
        return next_state, external_reward+intrinsic_reward, done,done,{'prev_state_type':prev_state_type==StateType.TARGET}
    '''
    def copy_from_env(self,env):
        self.subsamples = env.subsamples
        self.update_theta(env.action_value)
        self.iForest = copy.deepcopy(env.iForest)
    '''
