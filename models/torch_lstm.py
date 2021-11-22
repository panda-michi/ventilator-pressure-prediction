import torch
import torch.nn as nn

# reffred from https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
class simpleLSTM(nn.Module):
    def __init__(self, input_dim, lstm_dim, dense_dim, logit_dim, num_classes):
        super().__init__()
        
        self.fc_in0 = nn.Linear(input_dim, dense_dim // 2)
        self.fc_in1 = nn.Linear(dense_dim // 2, dense_dim)

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.fc_out0 = nn.Linear(lstm_dim * 2, logit_dim) 
        self.fc_out1 = nn.Linear(logit_dim, num_classes) 
    
    def forward(self, x):
        x = F.relu(self.fc_in0(x))
        x = F.relu(self.fc_in1(x))

        x, _ = self.lstm(x)

        x = F.relu(self.fc_out0(x))
        x = self.fc_out1(x)

        return x 

# implementation of https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/278964
class embedLSTM(nn.Module):
    def __init__(self, in_dim, sq_dim, hidden, dense):
        super().__init__()
        self.seq_emb = nn.Sequential(
            nn.Linear(in_dim, sq_dim),
            nn.LayerNorm(sq_dim),
        )
        
        self.lstm = nn.LSTM(sq_dim, hidden, batch_first=True, bidirectional=True, dropout=0.0, num_layers=4)

        self.head = nn.Sequential(
            nn.Linear(2*hidden, 2*hidden),
            nn.LayerNorm(2*hidden),
            nn.ReLU(),
        )

        self.pressure_in  = nn.Linear(2*hidden, 1)
        self.pressure_out = nn.Linear(2*hidden, 1)        
       
        # init LSTM
        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f'init {m}')
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        batch_size = len(x)
        x = self.seq_emb(x)
        
        x, _ = self.lstm(x, None)
        x = self.head(x)

        pressure_in  = self.pressure_in(x).reshape(batch_size,80)
        pressure_out = self.pressure_out(x).reshape(batch_size,80)

        return pressure_in, pressure_out

class dualDeepLSTM(nn.Module):
    def __init__(self,in_dim, hidden, dense):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2*hidden, hidden//2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden, hidden//4, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2*(hidden//2 + hidden//4), hidden//8, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(hidden//4, dense),
            nn.SELU(),
        )
        self.pressure_in  = nn.Linear(dense, 1)
        self.pressure_out = nn.Linear(dense, 1)

        #reffered: https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        batch_size = len(x)

        x, _ = self.lstm1(x)
        x1, _ = self.lstm2(x)
        x2, _ = self.lstm3(x1)
        x = torch.cat((x1, x2), 2)
        x, _ = self.lstm4(x)

        x = self.head(x)

        pressure_in  = self.pressure_in(x).reshape(batch_size,80)
        pressure_out = self.pressure_out(x).reshape(batch_size,80)

        return pressure_in, pressure_out

class TsLSTM(nn.Module):
    def __init__(self, in_dim, sq_dim, hidden0, hidden1, hidden2, hidden3, dense):
        super().__init__()
        self.r_embed = nn.Embedding(4, 2)
        self.c_embed = nn.Embedding(4, 2)

        self.seq_embed = nn.Sequential(
            Rearrange('b l d -> b d l'),
            nn.Conv1d(2+in_dim, sq_dim, kernel_size=5, padding=2, stride=1),
            Rearrange('b d l -> b l d'),
            nn.LayerNorm(sq_dim),
            nn.SiLU(),
            nn.Dropout(0.),)

        self.pos_encoder = PositionalEncoding(d_model=sq_dim, dropout=0.2)

        encoder_layers = nn.TransformerEncoderLayer(d_model=sq_dim, nhead=8, dim_feedforward=2048, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.lstm1 = nn.LSTM(sq_dim, hidden0, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2*hidden0, hidden1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2*hidden1, hidden2, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2*hidden2, hidden3, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(2*hidden3, dense),
            nn.SiLU(),
        )
        self.pressure_in  = nn.Linear(dense, 1)
        self.pressure_out = nn.Linear(dense, 1)

        initrange = 0.1
        self.r_embed.weight.data.uniform_(-initrange, initrange)
        self.c_embed.weight.data.uniform_(-initrange, initrange)

        #reffered: https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                #print(name,m)
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        batch_size = x.shape[0]

        r_embed = self.r_embed(x[:,:,0].long()).view(batch_size, 80, -1)
        c_embed = self.c_embed(x[:,:,1].long()).view(batch_size, 80, -1)
        seq_x = torch.cat((r_embed, c_embed, x[:, :, 2:]), 2)

        x = self.seq_embed(seq_x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.head(x)

        pressure_in  = self.pressure_in(x).reshape(batch_size,80)
        #pressure_out = self.pressure_out(x).reshape(batch_size,80)
        #return pressure_in, pressure_out

        return pressure_in