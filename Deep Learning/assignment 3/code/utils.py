import os, time, torch, numpy as np
import os.path as path
from torch.autograd import Variable
from matplotlib import pyplot as plt

def load_dataset(file):
    dataset=np.load(file, allow_pickle=True)
    data=dataset['data']
    ix2word=dataset['ix2word'].item()
    word2ix=dataset['word2ix'].item()

    return data,ix2word,word2ix

def mkdir(name):
    '''
    创建以当前时间为文件名的文件夹，并返回该文件夹地址
    '''
    PATH=path.abspath(path.join(path.dirname("__file__"),path.pardir))+'\\result'
    if not path.exists(PATH):
        print('creating folder...')
        os.mkdir(PATH)
        print(PATH)
        PATH+='\\'+name
        os.mkdir(PATH)
        print(PATH)
        print('Done.')
    else:
        PATH+='\\'+name
        if not path.exists(PATH):
            print('creating folder...')
            os.mkdir(PATH)
            print(PATH)
            print('Done.')
        else:
            print('the folder has been created.')
            print(PATH)
    return PATH

def draw_fig(data_list, save_path):
    plt.title('Loss Curve')
    plt.plot(range(1,len(data_list)+1),data_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.savefig(save_path+'\\loss-curve.png')
    plt.close()

def generate(model, start_words, ix2word, word2ix, \
            max_gen_len, device):

    model.to(device)
    
    # 读取唐诗的第一句
    results = list(start_words)
    start_word_len = len(start_words)
    
    # 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    # 生成唐诗
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        # 读取第一句
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 生成后面的句子
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        # 结束标志
        if w == '<EOP>':
            del results[-1]
            break
            
    string=''.join(results).replace('。','。\n')
    return string

def gen_acrostic(model, start_words, ix2word, word2ix, \
                max_gen_len,device):

    model.to(device)
    # 读取唐诗的“头”
    results = []
    start_word_len = len(start_words)
    
    # 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(device)
    hidden = None

    index = 0            # 指示已生成了多少句
    pre_word = '<START>' # 上一个词

    # 生成藏头诗
    for _ in range(max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        # 如果遇到标志一句的结尾，喂入下一个“头”
        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果生成的诗已经包含全部“头”，则结束
            if index == start_word_len:
                break
            # 把“头”作为输入喂入模型
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
                
        # 否则，把上一次预测作为下一个词输入
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    string=''.join(results).replace('。','。\n')
        
    return string


    
