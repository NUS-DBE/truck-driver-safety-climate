import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 原始的采样点
x_original = np.linspace(0, 1, 16)
y_original = np.random.rand(16)  # 这里用随机数代替原始的y值

# 生成目标的50个采样点
x_target = np.linspace(0, 1, 50)

# 使用线性插值将16个采样点插值到50个采样点
interpolator = interp1d(x_original, y_original, kind='linear')
y_target = interpolator(x_target)



def piecewise_function_1(x,split):
    return np.where(x < split, 0,1/(1-split))

def piecewise_function_2(x,split):
    return np.where(x < split, 1/split,0)

def distribution_value(x):


    x=(x - np.min(x)) / (np.max(x) - np.min(x))

    x_values = np.linspace(0, 1, len(x))


    # 生成目标的50个采样点
    x_target = np.linspace(0, 1, 50)
    # 使用线性插值将16个采样点插值到50个采样点
    interpolator = interp1d(x_values, x, kind='linear')
    y_target = interpolator(x_target)



    area = np.trapz(y_target, x=x_target)
    # 归一化处理
    normalized_y_values = y_target / area  #prepare normlized_pdp
    print(np.trapz(normalized_y_values,x=x_target))

    epsilon = 1e-10
    normalized_y_values += epsilon
    kl=10000
    kl_data=0

    for i in x_target[2:-2]:
        for piecewise_function in [piecewise_function_1,piecewise_function_2]:
            y_values_1 = piecewise_function(x_target,i).astype(np.float64)
            y_values_1 += epsilon

            # kl_divergence = np.sum(y_values_1 * np.log(y_values_1 / normalized_y_values))
            kl_divergence = np.mean((y_values_1 - normalized_y_values)**2)
            if kl>kl_divergence:
                kl=kl_divergence.copy()
                kl_data=y_values_1.copy()


    return x_target,normalized_y_values,kl,kl_data
