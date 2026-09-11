import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.RevIN import RevIN




class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        #self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.seg_len = configs.patch_len
        self.seg_num_x = self.seq_len // self.seg_len
        # building model
        #self.drop = nn.Dropout(self.dropout)
        self.rnn1 = nn.GRU(input_size= self.seg_len , hidden_size= self.seq_len, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        
        self.temp_linear= nn.Sequential(
            nn.Linear(self.seq_len ,self.pred_len),
        )
        self.layernorm1 = nn.LayerNorm(self.seq_len)
        self.layernorm2 = nn.LayerNorm(self.d_model )
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.pred_len * self.seg_num_x,self.d_model)
        )
        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.pred_len)
        )
        self.rev = RevIN(configs.c_in)
    
    def norm(self,x):
       
        x = (x - x.mean()) / x.std()
        return x
    
    def encoder(self, x_enc):
    
        batch_size, seq_len, channel_size= x_enc.size()
        x = x_enc.permute(0,2,1)
        x_seg = x.reshape(-1, self.seg_num_x, self.seg_len)
        hm,_ = self.rnn1(x_seg) 
        hsp = self.temp_linear(hm)
        hsp = hsp.reshape(batch_size,channel_size,-1)
        hp = self.mlp(hsp)
        hp = self.layernorm2(hp)
        y = self.predict(hp).view(-1,channel_size,self.pred_len)
        
        dec_out = y.permute(0,2,1)
      
        return dec_out #+ seq_last

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_enc = self.rev(x_enc, 'norm')
            dec_out = self.forecast(x_enc)
            dec_out = self.rev(dec_out, 'denorm')
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D