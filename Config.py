
from transformers import DecisionTransformerConfig
from transformers.testing_utils import require_torch, slow, torch_device

class DecisionTransformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        act_dim=6,
        state_dim=17,
        hidden_size=23,
        max_length=11,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        states = floats_tensor((self.batch_size, self.seq_length, self.state_dim))
        actions = floats_tensor((self.batch_size, self.seq_length, self.act_dim))
        rewards = floats_tensor((self.batch_size, self.seq_length, 1))
        returns_to_go = floats_tensor((self.batch_size, self.seq_length, 1))
        timesteps = ids_tensor((self.batch_size, self.seq_length), vocab_size=1000)
        attention_mask = random_attention_mask((self.batch_size, self.seq_length))

        config = self.get_config()

        return (
            config,
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask,
        )

    def get_config(self):
        return DecisionTransformerConfig(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
        )

    def create_and_check_model(
        self,
        config,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask,
    ):
        model = DecisionTransformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)

        self.parent.assertEqual(result.state_preds.shape, states.shape)
        self.parent.assertEqual(result.action_preds.shape, actions.shape)
        self.parent.assertEqual(result.return_preds.shape, returns_to_go.shape)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.seq_length * 3, self.hidden_size)
        )  # seq length *3 as there are 3 modelities: states, returns and actions

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict