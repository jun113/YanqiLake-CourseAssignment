
import mxnet as mx
from mxnet import nd
import os
import os.path as path
import time
import random
def mkdir():
    '''
    创建以当前时间为文件名的文件夹，并返回该文件夹地址
    '''
    current_time=time.strftime("%Y-%m-%d %H%M%S", time.localtime())
    PATH=path.abspath(path.join(path.dirname("__file__"),path.pardir))+'\\picture'
    if not path.exists(PATH):
        print('creating folder...')
        os.mkdir(PATH)
        print(PATH)
        PATH+='\\'+current_time
        os.mkdir(PATH)
        print(PATH)
        print('Done.')
    else:
        PATH+='\\'+current_time
        if not path.exists(PATH):
            print('creating folder...')
            os.mkdir(PATH)
            print(PATH)
            print('Done.')
        else:
            print('the folder has been created.')
            print(PATH)
    return PATH

def randomcolor():
    '''
    生成随机颜色
    '''
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def try_all_gpus():
    """Return all available GPUs, or [mx.cpu()] if there is no GPU."""
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx