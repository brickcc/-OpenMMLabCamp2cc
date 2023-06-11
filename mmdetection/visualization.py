# 数据集可视化

import os
import matplotlib.pyplot as plt
from PIL import Image

#%matplotlib inline  # 输出窗口显示图形--IPython/Jupyter Notebook中的魔术命令
#%config InlineBackend.figure_format = 'retina'  # 设置高分辨率--同上

original_images = []
images = []
texts = []
plt.figure(figsize=(16, 5))

image_paths= [filename for filename in os.listdir('train')][:8]

for i,filename in enumerate(image_paths):
    name = os.path.splitext(filename)[0]

    image = Image.open('train/'+filename).convert("RGB")
  
    plt.subplot(2, 4, i+1)
    plt.imshow(image)
    plt.title(f"{filename}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()