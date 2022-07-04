import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceFilter(nn.Module):
    def __init__(self, feature_dim, emb_dim, bi_lstm, fc_dim):
        super(VoiceFilter, self).__init__()
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.bi_lstm = bi_lstm
        self.fc_dim = fc_dim

        self.lstm_1 = nn.LSTM(
                        self.feature_dim + self.emb_dim,
                        self.feature_dim // 2 if self.bi_lstm else self.feature_dim,
                        batch_first=True,
                        bidirectional=self.bi_lstm
                    )
        self.lstm_2 = nn.LSTM(
                        self.feature_dim + self.emb_dim,
                        self.feature_dim // 2 if self.bi_lstm else self.feature_dim,
                        batch_first=True,
                        bidirectional=self.bi_lstm
                    )
        self.lstm_3 = nn.LSTM(
                        self.feature_dim + self.emb_dim,
                        self.feature_dim // 2 if self.bi_lstm else self.feature_dim,
                        batch_first=True,
                        bidirectional=self.bi_lstm
                    )

        self.fc1 = nn.Linear(self.feature_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.feature_dim)

    def forward(self, x, dvec):
        # dvec: [B, 1, emb_dim]
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2) # [B, T, feature + emb_dim]
        x, _ = self.lstm_1(x) # [B, T, 2*lstm_dim]
        x = torch.cat((x, dvec), dim=2)
        x, _ = self.lstm_2(x) # [B, T, 2*lstm_dim]
        x = torch.cat((x, dvec), dim=2)
        x, _ = self.lstm_3(x) # [B, T, 2*lstm_dim]
        
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        
        return x