import os
folder_path = 'E:/aafa/unet/d/label/'
for filename in os.listdir(folder_path):
    if filename.endswith('_mask.tif'): 
        new_name = filename.replace('_mask', '')
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
