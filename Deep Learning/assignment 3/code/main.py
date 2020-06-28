import torch
from model import PoetryModel, train
from utils import load_dataset, generate, gen_acrostic, mkdir, draw_fig

class config(object):
    dataset_file='tang.npz'
    lr=1e-3
    batch_size=128
    num_epochs=50
    start_words_1='机器学习'
    start_words_2='秋水共长天一色'
    max_gen_len = 125
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_or_load=False
    num_layers=3
    save_path=mkdir('20200617_1827')

if __name__=='__main__':
    #load config
    opt=config()
    #load dataset
    data, ix2word, word2ix=load_dataset(opt.dataset_file)

    #init model
    model=PoetryModel(len(word2ix), 128, 256, num_layers=opt.num_layers)
    #training
    if opt.train_or_load:
        model, loss_list=train(model=model, data=data, ix2word=ix2word, word2ix=word2ix, \
                lr=opt.lr, batch_size=opt.batch_size, num_epochs=opt.num_epochs, \
                device=opt.device)

        #save model
        torch.save(model.state_dict(), opt.save_path+'\\model.pt')
        #draw loss-curve
        draw_fig(loss_list, opt.save_path)
    else:
        #load model
        model.load_state_dict(torch.load(opt.save_path+'\\model.pt'))
    

    #generate acrostic
    acrostic=gen_acrostic(model=model, start_words=opt.start_words_1, \
                        ix2word=ix2word, word2ix=word2ix, \
                        max_gen_len=opt.max_gen_len, device=opt.device)
    #generate poetry
    Poetry=generate(model=model, start_words=opt.start_words_2, \
                    ix2word=ix2word, word2ix=word2ix, \
                    max_gen_len=opt.max_gen_len, device=opt.device)
    
    print(Poetry)
    print()
    print(acrostic)
