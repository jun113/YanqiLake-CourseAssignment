import os
import time
import torch
from torch import nn
from torch.autograd import Variable

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, vocab_size)


    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            h_0 = Variable(h_0)
        else:
            h_0 = hidden

        embeds = self.embeddings(input)
        output, hidden = self.gru(embeds, h_0)

        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden

def train(model, data, ix2word, word2ix, lr, batch_size, num_epochs, device):

    dataloader=torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)

    model.to(device)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 定义训练过程
    loss_list=[]
    for epoch in range(1, num_epochs+1):
        start=time.time()
        count, loss_temp=0, 0
        for batch_idx, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            input, target = data[:-1, :], data[1:, :]
            output, _ = model(input)
            loss = criterion(output, target.view(-1))

            if (batch_idx==0) or ((batch_idx+1)%50 == 0):
                print('Train Epoch: %d [%d/%d (%.1f%%)]\tLoss: %.6f\t elapse: %.3f sec' %( \
                    epoch, batch_idx * len(data[1]), len(dataloader.dataset), \
                    100. * batch_idx / len(dataloader), loss.item(), time.time()-start))

            loss_temp+=loss.item()
            count+=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss_temp/count)

    return model, loss_list

