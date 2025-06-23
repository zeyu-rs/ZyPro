"""
Author: Zeyu Xu
Date Created: September 4, 2023
Last Modified: September 4, 2023
Description: Organize the wildebeest dataset
"""

import os

base_dir = 'D:/***'

years = ['2009', '2010', '2013', '2015', '2018', '2020']
target_image_folder='D:/***/'
target_mask_folder='D:/***/'

for year_str in years:

    image_folder = os.path.join(base_dir, year_str, '3_Train_test', 'train', 'image')
    mask_folder = os.path.join(base_dir, year_str, '3_Train_test', 'train', 'mask')


    for filename in os.listdir(image_folder):
        new_name = os.path.splitext(filename)[0] + "_" + year_str + os.path.splitext(filename)[1]
        os.rename(os.path.join(image_folder, filename), os.path.join(image_folder, new_name))
        os.rename(os.path.join(image_folder, new_name), os.path.join(target_image_folder, new_name)) #move

    for filename in os.listdir(mask_folder):
        new_name = os.path.splitext(filename)[0] + "_" + year_str + os.path.splitext(filename)[1]
        os.rename(os.path.join(mask_folder, filename), os.path.join(mask_folder, new_name))
        os.rename(os.path.join(mask_folder, new_name), os.path.join(target_mask_folder, new_name))
