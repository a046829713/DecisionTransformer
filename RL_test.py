import torch
print("開始")
from transformers import DecisionTransformerModel, DecisionTransformerConfig
print("開始")
from environment import Env
print("開始")
from DataFeature import DataFeature
print("開始")
import time


class RL_Inference:
    def __init__(self, symbols: list) -> None:
        self.symbols = list(set(symbols))  # 避免重複
        self.hyperparameters()
        # 準備裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_ep_len = 1000
        self.env = self.prepare_env()  # 需要環境
        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.model = self.load_model()  # 加載模型

    def hyperparameters(self):
        self.TARGET_RETURN = 0

    def prepare_env(self):
        data = DataFeature().get_train_net_work_data_by_path(self.symbols)
        return Env(bars_count=1, prices=data, random_ofs_on_reset=False)

    def load_model(self):
        config = DecisionTransformerConfig(
            state_dim=self.state_dim,         # 狀態維度
            act_dim=self.act_dim,            # 動作維度
            max_ep_len=self.max_ep_len,      # 最大時間步長
            hidden_size=1024,                 # 隱藏層大小
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
        model = DecisionTransformerModel(config=config).to(self.device)
        checkpoint = torch.load(r"save\checkpoint_40.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 設置模型為評估模式
        return model

    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.model.config.state_dim)
        actions = actions.reshape(1, -1, self.model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        

        if self.model.config.max_length is not None:
            states = states[:, -self.model.config.max_length :]
            actions = actions[:, -self.model.config.max_length :]
            returns_to_go = returns_to_go[:, -self.model.config.max_length :]
            timesteps = timesteps[:, -self.model.config.max_length :]
            
            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self.model.config.max_length - states.shape[1]), torch.ones(states.shape[1])]
            )
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (states.shape[0], self.model.config.max_length - states.shape[1], self.model.config.state_dim),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            
            actions = torch.cat(
                [
                    torch.zeros(
                        (actions.shape[0], self.model.config.max_length - actions.shape[1], self.model.config.act_dim),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (returns_to_go.shape[0], self.model.config.max_length - returns_to_go.shape[1], 1),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.model.config.max_length - timesteps.shape[1]), device=timesteps.device
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None


        with torch.no_grad():
            _, action_preds, _ = self.model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )

        return action_preds[0, -1]

    def run(self):
        print("開始進入")
        target_return = torch.tensor(self.TARGET_RETURN, device=self.device).reshape(1, 1)
        state = self.env.reset()

        states = torch.from_numpy(state).to(self.device)
        actions = torch.zeros((0, self.act_dim), device=self.device)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
        episode_return, episode_length = 0, 0        
        done = False
        while not done:
            # 查看已分配的記憶體
            
            actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])


            action = self.get_action(
                states,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )


            actions[-1] = action
            action = action.detach().cpu().numpy()
            state, reward, done, _ = self.env.step(action)
            print(episode_length,reward)
            cur_state = torch.from_numpy(state).to(self.device).reshape(1, self.state_dim)
            states = torch.cat([states, cur_state], dim=0)

            rewards[-1] = torch.from_numpy(reward).to(self.device)
            pred_return = target_return[0, -1].cpu() + reward
            target_return = torch.cat([target_return, pred_return.to(self.device).reshape(1, 1)], dim=1)
            
            timesteps = torch.cat([timesteps, torch.tensor(((episode_length + 1) % 1000), device=self.device, dtype=torch.long).reshape(1, 1)], dim=1)

            episode_return += reward
            episode_length += 1
            

            if episode_length % 1000 == 0:
                actions = self.cut_offdata(actions,dim = 0)
                states = self.cut_offdata(states,dim = 0)
                timesteps = self.cut_offdata(timesteps,dim = 1)
                rewards = self.cut_offdata(rewards, dim=0)
                target_return = self.cut_offdata(target_return, dim=1)

            


        print(f"Episode finished: return {episode_return}, length {episode_length}")
    
            
    def cut_offdata(self, data, dim):
        if data.dim() == 1:  # 如果是一維張量
            return data[-self.model.config.max_length * 2:]
        elif data.dim() == 2:  # 如果是二維張量
            if dim == 0:
                return data[-self.model.config.max_length * 2:, :]
            elif dim == 1:
                return data[:, -self.model.config.max_length * 2:]
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(data.dim()))
    

    def memory_allocated(self):
        # 查看已分配的記憶體
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2} MB")

if __name__ == "__main__":
    print("開始")
    agent = RL_Inference(['BTCUSDT'])
    print("開始1")
    agent.run()
