from mxnet.gluon import data as gdata
import mxnet.gluon.data.vision.transforms as img_transforms
import os.path as path

def load_dataset(dataset_name,data_type):
    '''
    加载本地数据集
    number: cat vs dog: 2000:2000
    train set: cat vs dog: 1500:1500
    validation set: cat vs dog: 250:250
    test set: cat vs dog: 250:250
    '''
    PATH=path.abspath(path.join(path.dirname("__file__"),dataset_name))
    file_name=data_type + ' set'
    imgs=gdata.vision.ImageFolderDataset(path.join(PATH,file_name),flag=1)
    return imgs
    
    
def load_img_batch(dataset,batch_size,type):
    if type == 'train':
        transform=img_transforms.Compose([\
            #随机对图像裁剪出面积为原图像面积的0.08~1倍
            #高/宽：3/4 ~ 4/3，最后高度与宽度都缩放到224像素
            img_transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0,4.0/3.0)),\
            #随机左右翻转
            img_transforms.RandomFlipLeftRight(),\
            #随机变化亮度、对比度、饱和度
            img_transforms.RandomColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),\
            #随机噪声
            img_transforms.RandomLighting(0.1),\
            img_transforms.ToTensor(),\
            # 对图像的每个通道做标准化
            img_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        return gdata.DataLoader(dataset.transform_first(transform),batch_size=batch_size,shuffle=True,last_batch='keep')
    elif type == 'test':
        transform=img_transforms.Compose([\
            img_transforms.Resize(256),\
            img_transforms.CenterCrop(224),\
            img_transforms.ToTensor(),\
            img_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        return gdata.DataLoader(dataset.transform_first(transform),batch_size=batch_size,shuffle=False,last_batch='keep')
