import torch
from transformers import DecisionTransformerModel, DecisionTransformerConfig
from environment import Env
from DataFeature import DataFeature
import time
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import torch.optim as optim
from ReplayBuffer import ReplayBuffer
from torchviz import make_dot
import os
import matplotlib.pyplot as plt


# 對於離散行動使用交叉熵損失，對於連續行動使用均方誤差
class RL_Train():
    def __init__(self, symbols: list) -> None:
        self.symbols = list(set(symbols))  # 避免重複
        self.hyperparameters()
        self.env = self.prepare_env()  # 需要環境
        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.model = self.prepare_model()  # 需要模型
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learn_rate)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.model)

    def prepare_model(self):
        config = DecisionTransformerConfig(
            state_dim=self.state_dim,         # 狀態維度
            act_dim=self.act_dim,            # 動作維度
            max_ep_len=self.max_ep_len,      # 最大時間步長
            hidden_size=256,                 # 隱藏層大小
            n_layer=4,                       # Transformer層數
            n_head=1,                        # 注意力頭數
            n_inner=256,                     # 前饋層大小
            activation_function='relu',      # 激活函數
            n_positions=1024,                # 序列長度
            resid_pdrop=0.1,                 # 殘差層丟棄概率
            attn_pdrop=0.1,                  # 注意力層丟棄概率
            output_hidden_states=True,       # 是否輸出隱藏狀態
            max_length=100                 # 模型最多可以查看往前幾步
        )

        return DecisionTransformerModel(config=config).to(self.device)

    def prepare_env(self):
        data = DataFeature().get_train_net_work_data_by_path(self.symbols)
        return Env(bars_count=1, prices=data, random_ofs_on_reset=True)

    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        states = states.reshape(1, -1, self.model.config.state_dim)
        actions = actions.reshape(1, -1, self.model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # 不指定第三個維度：默認選擇所有的狀態維度數據。
        if self.model.config.max_length is not None:
            cut_states = states[:, -self.model.config.max_length:]
            cut_actions = actions[:, -self.model.config.max_length:]
            cut_returns_to_go = returns_to_go[:, -
                                              self.model.config.max_length:]
            cut_timesteps = timesteps[:, -self.model.config.max_length:]

            # 處理變長度序列：在自然語言處理（NLP）任務中，不同序列的長度通常不同。通過使用 attention_mask，我們可以在同一批次中處理不同長度的序列，而不會對模型的輸出產生影響。
            # 0代表模型不需要注意, 1代表需要注意
            attention_mask = torch.cat(
                [torch.zeros(self.model.config.max_length - cut_states.shape[1]),
                 torch.ones(cut_states.shape[1])]
            )

            attention_mask = attention_mask.to(
                dtype=torch.long, device=cut_states.device).reshape(1, -1)

            cut_states = torch.cat([
                torch.zeros(
                    (cut_states.shape[0],
                     self.model.config.max_length - cut_states.shape[1],
                     self.model.config.state_dim),
                    device=cut_states.device),
                cut_states
            ], dim=1)

            cut_actions = torch.cat(
                [
                    torch.zeros(
                        (cut_actions.shape[0], self.model.config.max_length -
                         cut_actions.shape[1], self.model.config.act_dim),
                        device=cut_actions.device,
                    ),
                    cut_actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            cut_returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (cut_returns_to_go.shape[0], self.model.config.max_length -
                         cut_returns_to_go.shape[1], 1),
                        device=cut_returns_to_go.device,
                    ),
                    cut_returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            cut_timesteps = torch.cat(
                [
                    torch.zeros(
                        (cut_timesteps.shape[0], self.model.config.max_length - cut_timesteps.shape[1]), device=cut_timesteps.device
                    ),
                    cut_timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

        else:
            attention_mask = None

        with torch.no_grad():
            _, action_preds, _ = self.model(
                states=cut_states,
                actions=cut_actions,
                rewards=rewards,
                returns_to_go=cut_returns_to_go,
                timesteps=cut_timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )

        return action_preds[0, -1], cut_states, cut_actions, cut_returns_to_go, attention_mask, cut_timesteps

    def train(self):
        print("開始訓練")
        ep = 0
        if self.load_checkpoint(if_use=False):
            ep = self.start_episode
            print(f"Resuming from episode {ep}")

        while True:
            begintime = time.time()
            ep += 1
            episode_return, episode_length = 0, 0

            target_return = torch.tensor(
                self.TARGET_RETURN, device=self.device).reshape(1, 1)
            state = self.env.reset()

            states = torch.from_numpy(state).to(self.device)
            actions = torch.zeros((0, self.act_dim), device=self.device)
            rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

            timesteps = torch.tensor(
                0, device=self.device, dtype=torch.long).reshape(1, 1)

            # 改用max_ep_len來控制整個遊戲的長度 ，原本是放在環境裡面的
            for t in range(self.max_ep_len):
                actions = torch.cat([actions, torch.zeros(
                    (1, self.act_dim), device=self.device)], dim=0)
                rewards = torch.cat(
                    [rewards, torch.zeros(1, device=self.device)])

                action, cut_states, cut_actions, cut_returns_to_go, attention_mask, cut_timesteps = self.get_action(
                    states,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                print(action)
                actions[-1] = action
                action = action.detach().cpu().numpy()
                state, reward, done, _ = self.env.step(action)

                # # 添加到replay buffer
                self.replay_buffer.add(
                    action, cut_states, cut_actions, cut_returns_to_go, attention_mask, cut_timesteps)

                cur_state = torch.from_numpy(state).to(
                    self.device).reshape(1, self.state_dim)
                states = torch.cat([states, cur_state], dim=0)
                rewards[-1] = torch.from_numpy(reward).to(self.device)


                pred_return = target_return[0, -1].cpu() + reward * self.gamma
                
                target_return = torch.cat(
                    [target_return, pred_return.to(self.device).reshape(1, 1)], dim=1)                
                
                
                timesteps = torch.cat([timesteps, torch.ones(
                    (1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

                episode_return += reward
                episode_length += 1

                if done:
                    break

                if len(self.replay_buffer) == self.batch_size:
                    # 更新模型
                    self.optimizer.zero_grad()
                    loss = self.update_model(ep)
                    loss.backward()
                    self.optimizer.step()
                    self.replay_buffer.clear()

            print(f"Episode {ep}: return {episode_return}, length {episode_length},this use time:{time.time() - begintime}")
            # 記錄並繪製訓練過程中的指標
            # self.log_metrics(ep, episode_return, episode_length, target_return)


            # Save checkpoint every 10 episodes
            if ep % 20 == 0:
                self.save_checkpoint(ep)

    def log_metrics(self, episode, episode_return, episode_length, target_return):
        # 記錄指標
        self.returns_to_go_history.append(target_return.mean().item())
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

        # 繪製指標圖
        if episode % 10 == 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 3, 1)
            plt.plot(self.episode_returns)
            plt.title('Episode Returns')

            plt.subplot(1, 3, 2)
            plt.plot(self.episode_lengths)
            plt.title('Episode Lengths')

            plt.subplot(1, 3, 3)
            plt.plot(self.returns_to_go_history)
            plt.title('Returns to Go')

            plt.show()
            
    def load_checkpoint(self, if_use):
        if if_use == False:
            return False

        save_dir = 'save'
        if not os.path.exists(save_dir):
            print("No checkpoint directory found.")
            return False

        checkpoint_files = [f for f in os.listdir(
            save_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        if not checkpoint_files:
            print("No checkpoint files found.")
            return False

        latest_checkpoint = max(
            checkpoint_files, key=lambda f: os.path.getctime(os.path.join(save_dir, f)))
        checkpoint_path = os.path.join(save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        self.start_episode = checkpoint['episode']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        return True

    def save_checkpoint(self, episode):
        os.makedirs('save', exist_ok=True)
        checkpoint_path = f"save\\checkpoint_{episode}.pt"
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Checkpoint saved at episode {episode}")

    def update_model(self, ep):
        # 從replay buffer中取樣
        action, states, actions, returns_to_go, attention_masks, timesteps = self.replay_buffer.sample(
            self.batch_size)

        _, action_preds, _ = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_masks,
            return_dict=False,
        )

        loss = self.calculate_loss(action_preds[:, -1, :], action)
        return loss

    def calculate_loss(self, action_preds, actions):
        # 使用MSE損失計算
        actions = actions.reshape(-1, 1)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(action_preds, actions)
        return loss

    def make_count_map(self, loss):
        # 假設你有一個損失張量 loss
        dot = make_dot(loss, params=dict(list(self.model.named_parameters())))
        dot.format = 'png'
        dot.render('computed_graph')

    def hyperparameters(self):
        self.replay_buffer_size = 100000
        self.batch_size = 64
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_ep_len = 1000
        self.learn_rate = 1e-4
        self.TARGET_RETURN = 0
        self.gamma = 0.99

RL_Train(symbols=['BTCUSDT']).train()


# model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert")
# model = model.to(self.device)
