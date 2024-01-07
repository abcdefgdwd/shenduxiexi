#街景字符数据集下载运行这个文件
#地址：https://tianchi.aliyun.com/competition/entrance/531795/information

import pandas as pd
import os
import requests
import zipfile
import shutil
links = pd.read_csv('./content/mchar_data_list_0515.csv')
dir_name = 'NDataset'
mypath = './content/'
if not os.path.exists(mypath + dir_name):
    os.mkdir(mypath + dir_name)
for i,link in enumerate(links['link'] ):
    file_name = links['file'][i]
    print(file_name, '\t', link)
    file_name = mypath + dir_name + '/' + file_name
    if not os.path.exists(file_name):
        response = requests.get(link, stream=True)
        with open( file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
zip_list = ['mchar_train', 'mchar_test_a', 'mchar_val']
for little_zip in zip_list: # 卖萌可耻
    if not os.path.exists(mypath + dir_name + '/' + little_zip):
        zip_file = zipfile.ZipFile(mypath + dir_name + '/' + little_zip + '.zip', 'r')
        zip_file.extractall(path = mypath + dir_name )
if os.path.exists(mypath + dir_name + '/' + '__MACOSX'):
    shutil.rmtree(mypath + dir_name + '/' + '__MACOSX')
