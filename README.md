# 修改的内容
新增__init__.py
seele\gaussian_renderer\__init__.py
```
#from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except:
    pass
```
seele\scene\gaussian_model.py是重要，是调用此框架和主要部分
```
#新的导包
import time
import torch
import numpy as np
from submodules.seele.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from submodules.seele.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from submodules.seele.utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from submodules.seele.utils.graphics_utils import BasicPointCloud
from submodules.seele.utils.general_utils import strip_symmetric, build_scaling_rotation
from torch.utils.hooks import unserializable_hook
```

```
#新的异步加载函数
    def preload_next(self, next_cid, lod_dict, lod_use_str_list,length_list):
        self.next_cid = next_cid

        if next_cid == self.current_cid:
            self.next_gaussians = self.current_gaussians
            return

        if isinstance(next_cid, list):

            with torch.cuda.stream(self.load_stream):
                t1 = time.time()
                # 按索引提GS
            all_index=[]
            if isinstance(lod_use_str_list, list):
                for i, lod_use_str in zip(next_cid,lod_use_str_list):
                    if i == 0:
                        all_index.append(lod_dict[lod_use_str][0:length_list[lod_use_str][0]])
                    else:
                        all_index.append(lod_dict[lod_use_str][sum(length_list[lod_use_str][:i]):sum(length_list[lod_use_str][:i+1])])
            else:#lod_use_str_list是一个字符串，例如‘lod0’
                for i in next_cid:
                    lod_use_str = lod_use_str_list
                    if i == 0:
                        all_index.append(lod_dict[lod_use_str][0:length_list[lod_use_str][0]])
                    else:
                        all_index.append(lod_dict[lod_use_str][sum(length_list[lod_use_str][:i]):sum(length_list[lod_use_str][:i+1])])

            index=torch.cat(all_index,dim=0)
            # 关键点
            expanded_index = index.unsqueeze(1).expand(-1, self.cluster_gaussians.size(1))
            result = torch.gather(self.cluster_gaussians, 0, expanded_index)
            cat_means, cat_scales, cat_quats, cat_colors, cat_opacities = result[:, 0:3], result[:, 3:6], result[:, 6:10], result[:,10:13], result[:, 13]
            cat_colors=cat_colors.unsqueeze(1)

            # print('拼接后数据量',cat_means.shape[0])#, cat_scales.shape, cat_quats.shape, cat_colors.shape, cat_opacities.shape)
            # print("torch.cat use time----------------:",time.time()-t1)
            cat_tensor = (cat_means, cat_scales, cat_quats, cat_colors, cat_opacities)
            self.next_gaussians = tuple(map(
                lambda tensor: tensor.to(self.device, non_blocking=True),
                cat_tensor
            ))
```