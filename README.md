# Neuron_distance_v3d
A cpp implementation of neuron_distance plugin in Vaa3d software 
https://github.com/vaa3d

引用这些Metric的论文：如
Li, Q., & Shen, L. (2019). 3D neuron reconstruction in tangled neuronal image with deep networks. IEEE transactions on medical imaging, 39(2), 425-435.



使用方式: neuron_dist <gt_swc_file> <test_swc_file>

输出结果: ESA12, ESA21, ESA_mean, DSA, PDS

ESA12: Entire Structure Average 1 to 2
ESA21: Entire Structure Average 2 to 1
ESA_mean: Mean of above
DSA: Different Structure Average
PDS: Percentage of Different Structures

此代码计算结果与Vaa3d中neuron_dist插件计算结果前3位小数均一致，方便用于批量计算。


