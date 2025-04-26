#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,hidden_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.hidden_size = hidden_size
        "*** YOUR CODE HERE ***"
        self.gru = nn.GRU(
            input_size   = state_size,
            hidden_size  = hidden_size,
            num_layers   = 2,
            batch_first  = True,   # expects input of shape (batch, seq, feature)
        )

        # final fully-connected to map from GRU hidden to action_size Q-values
        self.fc = nn.Linear(hidden_size, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        # add a time-dimension of length 1 so GRU can process it
        # becomes (batch, seq_len=1, state_size)
        x = state.unsqueeze(1)

        # GRU returns:
        #   out:   (batch, seq_len, hidden_size)
        #   h_n:   (num_layers, batch, hidden_size)
        out, h_n = self.gru(x)

        # we only care about the last time‐step’s output:
        # out[:, -1, :] has shape (batch, hidden_size)
        last_out = out[:, -1, :]

        # pass through ReLU then final linear layer
        x = F.relu(last_out)
        action_values = self.fc(x)

        return action_values