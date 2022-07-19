# vloong-First-submit
本文件用于说明该工程中各个文件的使用顺序，由于原始数据较大，还有一些数据精简处理相关操作，均说明如下：
请按照下列顺序执行文件：
1.  LoadData.m
    这是一个matlab文件，是在原文作者给的数据处理程序的基础上修改的，用以将数据加载到Matlab工作区。这里没有附上原始数据，在使用该文件时，请先将原始数据放入data文件夹下，或修改该文件中的数据文件路径。
2.  data_pick.m
    这是一个matlab文件，用以将原始数据中不重要的、未用到的数据剔除，并生成一个mat格式的精简数据simplified2.mat。
3.  data_load.py
    用以读取simlified2.mat并存储成npy格式，方便后面的调用。
4.  Elanet.py
    基于弹性网络得到的结果，主要是在比赛主办方给的参考代码基础上修改的，能达到约13%的平均相对误差。
5.  withCNN.py
    使用卷积神经网络得到的结果，能达到约9%的平均相对误差。
![image](https://user-images.githubusercontent.com/37606459/179770375-1646288e-9ad7-41cd-a2e0-25828b4fd76a.png)
![CNN对比图](https://user-images.githubusercontent.com/37606459/179770524-279255a8-9a42-41d0-a209-47852a19cbf6.png)

所用的Python第三方库有：
Pytorch, Numpy, Matplotlib, Sklearn, h5py. Scipy, Pandas
