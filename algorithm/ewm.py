from prettytable import PrettyTable, SINGLE_BORDER
import numpy as np


class Entropy_Weight_Method:
    """Entropy weight method, 熵权法

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        指标矩阵

    data_type : ndarray of shape (n_features,)
        指示向量, 指示各列指标数据是正向指标或负向指标, 1表示正向指标,2表示负向指标, 例如[1,1,2,1]

    scale_min : float, optional, default=0.0001
        归一化的区间端点, 即归一化时将数据缩放到(scale_min, scale_max)的范围内, 默认应设置为(0,1)

    scale_max : float, optional, default=0.9999
        归一化的区间端点, 即归一化时将数据缩放到(scale_min, scale_max)的范围内, 默认应设置为(0,1)

    display : bool, optional, default=True
        是否打印指标权重输出表格

    Returns
    ----------
    y_norm : ndarray of shape (n_samples, n_features)
        归一化后的数据
    score : ndarray of shape (n_features, )
        综合加权评分
    weight : ndarray of shape (n_features,)
        各指标权重
    """

    def __init__(self, data, data_type, scale_min=0, scale_max=1, display=True, indexCol=None):
        # 检测输入数据是否为numpy数组
        if not isinstance(data, np.ndarray):
            raise TypeError("指标矩阵必须为numpy.ndarray类型")
        # 检测输入数据是否为二维数组
        if len(data.shape) != 2:
            raise ValueError("指标矩阵必须为二维数组")
        self.data = data
        self.data_type = data_type
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.n, self.m = self.data.shape
        self.display = display
        self.indexCol = indexCol

    def transform(self):
        # MinMaxNormalize 归一化
        y_norm = np.zeros((self.n, self.m))
        x_min, x_max = np.min(self.data, axis=0), np.max(self.data, axis=0)
        for i in range(self.m):
            if self.data_type[i] == 1:  # 正向指标归一化
                for j in range(self.m):
                    y_norm[:, j] = ((self.scale_max - self.scale_min) * (self.data[:, j] - x_min[j]) / (
                            x_max[j] - x_min[j]) + self.scale_min).flatten()
            elif self.data_type[i] == 2:  # 负向指标归一化
                for j in range(self.m):
                    y_norm[:, j] = ((self.scale_max - self.scale_min) * (x_max[j] - self.data[:, j]) / (
                            x_max[j] - x_min[j]) + self.scale_min).flatten()
        return y_norm

    def fit(self):
        # EWM熵权法
        y_norm = self.transform()
        # 计算第m项指标下第m个样本值占该指标的比重:比重P(i,j)
        P = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                P[i, j] = y_norm[i, j] / np.sum(y_norm[:, j])
        # 第j个指标的熵值e(j)
        e = np.zeros((1, self.m))
        # 其中k = 1/ln(n)
        k = 1 / np.log10(self.n)
        for j in range(self.m):
            e[0, j] = -k * np.sum(P[:, j] * np.log10(P[:, j]))
        # 计算信息熵冗余度
        d = np.ones_like(e) - e
        # 计算各项指标的权重
        weight = (d / np.sum(d)).flatten()
        # 计算该样本的综合加权评分
        score = np.sum(weight * y_norm, axis=1)
        # 输出结果
        if self.display:
            print_tb = PrettyTable()
            print_tb.add_column("index", np.arange(self.m) if self.indexCol is None else self.indexCol)
            print_tb.add_column("Index weight", weight)
            print_tb.align = "l"
            print_tb.set_style(SINGLE_BORDER)
            print(print_tb)
        return y_norm, score, weight
