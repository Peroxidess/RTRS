import copy
import pandas as pd
import torch
import torch as T
import torch.nn.functional as F
import numpy as np
from model.TD3.networks import ActorNetwork, CriticNetwork, ClassifyNetwork
from model.TD3.buffer import ReplayBuffer
import torch.utils.data as Data
from sklearn.metrics import roc_curve

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, alpha, beta, gama, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, name_RL='ADRL', gamma=0.99, tau=0.005, lmbda =2.5, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=1000000,
                 batch_size=256, seed=2022):
        self.seed = seed
        T.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.gamma = gamma
        self.tau = tau
        self.lmbda = lmbda
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir
        self.name_RL = name_RL
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim).to(self.device)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim).to(self.device)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim).to(self.device)
        self.classifier = ClassifyNetwork(gama=gama, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim).to(self.device)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim).to(self.device)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim).to(self.device)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim).to(self.device)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def memory_init(self, max_action):
        self.memory.memory_init()

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        state = T.reshape(state, shape=(-1, observation.shape[1]))
        rewards_tensor = T.zeros(size=(observation.shape[0], 1), dtype=T.float).to(device)
        action = self.actor.forward(state)

        if train:
            # exploration noise
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            action = action + noise
        self.actor.train()

        self.classifier.eval()
        state_grad = T.tensor([observation], dtype=T.float).to(device)
        state_grad = T.reshape(state_grad, shape=(-1, observation.shape[1]))

        actions_tensor_grad = self.target_actor.forward(state_grad)
        observation[:, -1] = actions_tensor_grad.detach().numpy().reshape(-1, )
        state_grad_ = T.tensor(observation, dtype=T.float, requires_grad=True).to(device)
        state_grad_.retain_grad()
        y_grad = self.classifier.forward(state_grad_)[0]
        y_grad[:, 0].backward(T.ones((rewards_tensor.shape[0],)).to(device), retain_graph=True)
        state_grad = copy.deepcopy(state_grad_.grad)
        y_grad[:, 1].backward(T.ones((rewards_tensor.shape[0],)).to(device), retain_graph=True)
        act_bias = state_grad.detach().numpy()[:, -1]

        if 'LC' in self.name_RL:
            action_emseble = action.squeeze().detach().cpu().numpy() - 1e-1 * act_bias
        else:
            action_emseble = action.squeeze().detach().cpu().numpy()
        return action_emseble

    def learn_class(self, X, Y, val_x, val_y):
        np.random.seed(self.seed)
        X_new = pd.DataFrame([])

        id_unique = X['病例编号'].unique()
        for id_ in id_unique:
            data_single = X[X['病例编号'] == id_]
            label_single = Y.loc[data_single.index]
        # lv1
            data_tmp = data_single.iloc[[0]]
            data_tmp.loc[:, 'action'] = data_single['action'].sum()
            data_tmp.loc[:, 'label_cumulative'] = label_single.iloc[0].values[0]
            X_new = pd.concat([X_new, data_tmp])

        val_x.drop(columns=val_x.filter(regex=r'次序|编号').columns, inplace=True)

        Y_ = Y.loc[X_new.index]
        index_raw = Y_.index
        X_ = X_new.drop(columns=X_new.filter(regex=r'次序|编号|label_cumulative').columns)
        data_labeled = MyDataset(X_, Y_)
        train_data_loader = Data.DataLoader(data_labeled, batch_size=256, worker_init_fn=np.random.seed(self.seed), shuffle=True)
        loss_CE = self.classifier.loss

        for iter_count in range(250):
            total_loss_ = 0.
            for train_data_batch, train_y_batch in train_data_loader:
                train_data_batch = train_data_batch.to(self.device)
                train_y_batch = train_y_batch.to(self.device)
                self.classifier.train()
                y, dist = self.classifier.forward(train_data_batch)
                loss_c = loss_CE(y, train_y_batch)
                self.classifier.optimizer.zero_grad()
                loss_c.backward()
                self.classifier.optimizer.step()
                total_loss_ += (loss_c.detach().cpu().numpy() / len(train_data_batch))
            self.classifier.eval()
            x_tensor = T.tensor(X_.values, dtype=T.float).to(self.device)
            pred_y_, dist_ = self.classifier.forward(x_tensor)
            pred_y_ = pred_y_.detach().cpu().numpy()
            pred_y_0 = pred_y_[:, 0]
            pred_y_0_df = pd.DataFrame(pred_y_0, index=Y_.index)
            pred_y_0_raw = pred_y_0_df.loc[index_raw]
            fpr, tpr, thresholds = roc_curve(Y_.loc[index_raw], pred_y_0_raw)
        pass

    def learn(self):
        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        reward_pred = self.predict_y(states, actions)
        if 'ADRL' in self.name_RL:
            rewards[rewards == 1] = 1
            rewards[rewards == 0] = -0.5
            rewards += 1e-3 * (reward_pred.reshape(-1, 2)[:, 0] * 2 - 1)
        rewards_tensor_raw = T.tensor(rewards, dtype=T.float).to(device)
        # rewards += (abs(reward_pred[:, 0] - reward_pred[:, 1]) / (reward_pred[:, 0] + reward_pred[:, 1]) - 0.5) * 4
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        # states_tensor_grad = T.tensor(states, dtype=T.float, requires_grad=True).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float, requires_grad=True).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(device)
            state_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(device)
            # smooth noise
            next_actions_tensor_nor = (next_actions_tensor - next_actions_tensor.mean()) / next_actions_tensor.std()
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            state_noise = T.clamp(state_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = next_actions_tensor + action_noise
            next_states_tensor = next_states_tensor + state_noise
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            critic_val = T.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val

        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)
        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return True, critic_loss.detach().cpu().numpy(), np.nan

        new_actions_tensor = self.actor.forward(states_tensor)
        new_actions_tensor_nor = (new_actions_tensor - new_actions_tensor.mean()) / new_actions_tensor.std()
        q1_a = self.critic1.forward(states_tensor, new_actions_tensor)
        q1_label = self.critic1.forward(states_tensor, actions_tensor)
        q_diff = np.array((q1_a - q1_label).detach().cpu())
        action_diff = np.array((new_actions_tensor - actions_tensor).detach().cpu())

        lmbda1 = self.lmbda / q1_a.abs().mean().detach().cpu()
        loss_1 = - T.mean(q1_a)
        loss_2 = F.mse_loss(new_actions_tensor * rewards_tensor_raw.view(-1, 1), actions_tensor * rewards_tensor_raw.view(-1, 1))

        if 'ADRL' in self.name_RL:
            actor_loss = 1 * loss_1 * lmbda1 + 6 * loss_2
        else:
            actor_loss = loss_1 * lmbda1 + 1e-1 * loss_2
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_network_parameters()
        return True, critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def predict_q(self, observation, action):
        self.actor.eval()
        self.critic1.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        action_tensor = T.tensor([action], dtype=T.float).to(device)
        state = T.reshape(state, shape=(-1, observation.shape[1]))
        action_tensor = T.reshape(action_tensor, shape=(-1, action.shape[1]))
        q = self.critic1.forward(state, action_tensor)
        self.actor.train()
        self.critic1.train()
        q_numpy = q.squeeze().detach().cpu().numpy()
        return q_numpy

    def predict_y(self, observation, action):
        self.classifier.eval()
        states = T.tensor([observation], dtype=T.float).to(device)
        action_tensor = T.tensor([action], dtype=T.float).to(device)
        states = T.reshape(states, shape=(-1, observation.shape[1]))
        y = self.classifier.forward(states)
        y_np = y[0].squeeze().detach().cpu().numpy()
        return y_np

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Saving critic1 network successfully!')
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Saving target critic1 network successfully!')
        self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Saving critic2 network successfully!')
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Saving target critic2 network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        print('Loading critic2 network successfully!')
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))
        print('Loading target critic2 network successfully!')


class MyDataset(Data.Dataset):
    def __init__(self,
                 data,
                 label=None,
                 random_seed=0):
        super(MyDataset, self).__init__()
        self.rnd = np.random.RandomState(random_seed)
        data = data.astype('float32')
        label = label.astype('float32')

        list_data = []
        if label is not None:
            for index_, values_ in data.iterrows():
                # y = torch.LongTensor([label.loc[index_].astype('int64')]).squeeze()
                y = torch.tensor(label.loc[index_])
                x = data.loc[index_].values
                list_data.append((x, y))
        else:
            for index_, values_ in data.iterrows():
                x = data.loc[index_].values
                list_data.append((x))

        self.shape = x.shape
        self.data = list_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
