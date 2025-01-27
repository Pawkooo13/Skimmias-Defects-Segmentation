# %%
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np
import cv2

# %%
data_folder_path = os.path.join(os.getcwd()[:-9], 'data')
data_folder_path

# %%
images_path = os.path.join(data_folder_path, 'images')
masks_path = os.path.join(data_folder_path, 'masks')

# %%
random_image = random.choice(os.listdir(images_path))
random_image

image_path = os.path.join(images_path, random_image)
mask_path = os.path.join(masks_path, random_image[:-4] + '-mask.png')

image = Image.open(image_path)
mask = Image.open(mask_path)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(image)
ax2.imshow(mask)

ax1.axis('off')
ax2.axis('off')

fig.show()

# %%
mask = np.asarray(mask)
image = np.asarray(image)

np.unique(mask.reshape(-1, 3))

# %%
crop_image = np.where(mask == 255, image, 0)
crop_image.shape

plt.imshow(crop_image)

# %%
plt.imshow(np.where((mask>0) & (mask<255), image, 0))

# %%
images = os.listdir(images_path) 
masks = os.listdir(masks_path)

# %% [markdown]
# POGRYZIONE FRAGMENTY KWIATOW

# %%
bitten_chunks = []

for i in range(len(images)):
    image_path = os.path.join(images_path, images[i])
    mask_path = os.path.join(masks_path, images[i][:-4] + '-mask.png')

    image = np.asarray(Image.open(image_path))
    mask = np.asarray(Image.open(mask_path))

    bitten_chunks.append(np.where(mask==255, image, 0))

bitten_chunks = np.array(bitten_chunks)



# %%
bitten_chunks.shape

# %% [markdown]
# RGB

# %%
rgbs = bitten_chunks.reshape(366*512*512, 3)

# %%
r_channel = rgbs[:,0]

plt.hist(r_channel[r_channel != 0])

# %%
g_channel = rgbs[:,1]

plt.hist(g_channel[g_channel != 0])

# %%
b_channel = rgbs[:,2]

plt.hist(b_channel[b_channel != 0])

# %% [markdown]
# HSV

# %%
hsv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in bitten_chunks])
hsv = hsv.reshape(366*512*512, 3)

# %%
hsv.shape

# %%
h_channel = hsv[:,0] #od 0 do 180 max
plt.hist(h_channel[h_channel != 0])

# %%
s_channel = hsv[:,1] 
plt.hist(s_channel[s_channel != 0])

# %%
v_channel = hsv[:,2]
plt.hist(v_channel[v_channel != 0])

# %% [markdown]
# SPALONE FRAGMENTY KWIATOW

# %%
burned_chunks = []

for i in range(len(images)):
    image_path = os.path.join(images_path, images[i])
    mask_path = os.path.join(masks_path, images[i][:-4] + '-mask.png')

    image = np.asarray(Image.open(image_path))
    mask = np.asarray(Image.open(mask_path))

    burned_chunks.append(np.where((mask > 0) & (mask < 255), image, 0))

burned_chunks = np.array(burned_chunks)

# %%
burned_chunks.shape

# %%
rgbs = burned_chunks.reshape(366*512*512, 3)

# %%
r_channel = rgbs[:,0]

plt.hist(r_channel[r_channel != 0])

# %%
g_channel = rgbs[:,1]

plt.hist(g_channel[g_channel != 0])

# %%
b_channel = rgbs[:,2]

plt.hist(b_channel[b_channel != 0])

# %% [markdown]
# HSV

# %%
hsv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in burned_chunks])
hsv = hsv.reshape(366*512*512, 3)

# %%
h_channel = hsv[:, 0]

plt.hist(h_channel[h_channel != 0])

# %%
s_channel = hsv[:, 1]

plt.hist(s_channel[s_channel != 0])

# %%
v_channel = hsv[:, 2]

plt.hist(v_channel[v_channel != 0])

# %%



