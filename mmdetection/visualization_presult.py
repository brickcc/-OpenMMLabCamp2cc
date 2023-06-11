# 预测结果数据可视化

import os
import matplotlib.pyplot as plt
from PIL import Image

#%matplotlib inline

plt.figure(figsize=(20, 20))

# 你如果重新跑，这个时间戳是不一样的，需要自己修改
root_path='/nvme0/chengcan/openmmlab/mmdetection/work_dirs/balloon_rtmdet/20230611_191752/results/'
image_paths= [filename for filename in os.listdir(root_path)][2:6]

for i,filename in enumerate(image_paths):
    name = os.path.splitext(filename)[0]

    image = Image.open(root_path+filename).convert("RGB")
  
    plt.subplot(4, 1, i+1)
    plt.imshow(image)
    plt.title(f"{filename}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()