API 参考
=========

BaoIAD 基于 MMEngine 的 Registry 机制组织模型、数据集、转换、指标、Hook 和可视化组件。公开 API 以源码和英文 API 索引为准：

- `英文 API 索引 <https://baoiad.readthedocs.io/en/latest/api.html>`_
- `baoiad/registry.py <https://github.com/Baosight-xVue/BaoIAD/blob/master/baoiad/registry.py>`_
- `baoiad/method_inventory.py <https://github.com/Baosight-xVue/BaoIAD/blob/master/baoiad/method_inventory.py>`_

主要命名空间
------------

``baoiad.models``
   检测器、骨干网络、Neck、Head 与损失函数。

``baoiad.datasets``
   数据集适配器、数据转换与采样器。数据集本身不随仓分发。

``baoiad.evaluation``
   图像级和像素级异常检测指标。

``baoiad.registry``
   BaoIAD scope 下的 Registry 定义。

``baoiad.structures``
   异常检测数据样本结构。

导入这些命名空间可能需要与所选方法对应的可选依赖。如遇到条件导入或外部资产要求，请查阅方法 README 和方法状态清单。
