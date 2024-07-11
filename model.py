from transformers.models.decision_transformer import DecisionTransformerPreTrainedModel
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput,DecisionTransformerGPT2Model
from torch import nn
import torch
from typing import Optional, Tuple, Union


class DecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """
        覆寫原本DecisionTransformerModel的架構

    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 注意：此GPT2Model與默認的Huggingface版本唯一的區別在於移除了位置嵌入（因為我們會自行添加）        
        self.encoder = DecisionTransformerGPT2Model(config)

        # 定義嵌入層
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)  # 嵌入時間步長
        self.embed_return = torch.nn.Linear(1, config.hidden_size)  # 嵌入回報        
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)  # 嵌入狀態        
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)  # 嵌入動作
        self.embed_ln = nn.LayerNorm(config.hidden_size)  # 層歸一化

        # 我們不預測狀態或回報（根據論文）
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)  # 預測狀態
        
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )  # 預測動作
        
        
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)  # 預測回報

        # 初始化權重並進行最終處理
        self.post_init()

    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DecisionTransformerModel
        >>> import torch

        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```"""


        # 模型在前向傳播過程中輸出注意力權重
        # self.config.output_attentions 預設是False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        

        # False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 獲取批次大小和序列長度 #1,1000
        batch_size, seq_length = states.shape[0], states.shape[1]


        # 如果沒有提供attention_mask，則默認全部可關注
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)


        # 使用不同的頭嵌入每種模態        
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # 時間嵌入類似於位置嵌入
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings


        # 將序列變成 (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)
        # (1,3000,128)

        # 將attention mask也堆疊以匹配輸入
        # torch.Size([1, 3000])
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        device = stacked_inputs.device

        # 將嵌入的輸入（不是NLP中的詞索引）餵入模型
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 重塑x，使第二維度對應於原始的回報（0），狀態（1）或動作（2）
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # 獲取預測結果
        return_preds = self.predict_return(x[:, 2])  # 預測下一個回報
        state_preds = self.predict_state(x[:, 2])  # 預測下一個狀態
        action_preds = self.predict_action(x[:, 1])  # 預測下一個動作
        if not return_dict:
            return (state_preds, action_preds, return_preds)

        # 返回結果
        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )