import os
from skimage import io,transform
from paddle.io import Dataset
img_dim = 256

PATH = '/home/aistudio/data/data111879/images256x256/00000/'

class MyDataSet(Dataset):
    """
    数据集定义
    """
    def __init__(self,path=PATH):
        """
        构造函数
        """
        super(MyDataSet, self).__init__()
        self.dir = path
        self.datalist = os.listdir(PATH)
        self.image_size = (img_dim,img_dim)
    
    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):
        img = self._load_img(self.dir + self.datalist[idx])
        return img, 0

    # 返回整个数据集的总数
    def __len__(self):
        return len(self.datalist)
    
    def _load_img(self, path):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        try:
            img = io.imread(path)
            img = transform.resize(img,self.image_size)
            img = img.transpose()
            img = img.astype('float32')
        except Exception as e:
                print(e)
        return img