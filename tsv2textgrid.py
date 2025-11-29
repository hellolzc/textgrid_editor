#!/usr/bin/env python
#_*_encoding:utf-8_*_
# hellolzc 20251030
import os
import sys
import re
import pandas as pd

sys.path.append('./src')
from textgrid_editor import TextGrid
# 查找in目录下TSV文件并转换成TextGrid文件

IN_DIR = './tests/in'
OUT_DIR = './tests/out'

# %%
def read_tsv(filepath, split='\t'):
    df = pd.read_csv(filepath, encoding='utf-8', quoting=3, sep=split)  #  QUOTE_NONE (3)
    return df

def save_tsv(filepath, df, split='\t'):
    df.to_csv(filepath, sep=split, header=True, index=False, encoding='utf-8', quoting=3)  # csv.QUOTE_NONE (3)




def convert_file(old_path, out_dir):
    (head, tail) = os.path.split(old_path)
    name, ext = os.path.splitext(tail)
    #print("%s %s"%(name, ext))
    if ext == '.tsv':
        print("process name: %s"%name, end='\t')
        out_file = os.path.join(out_dir, name + '.TextGrid')
        try:
            tsv_df = read_tsv(old_path)
            if 'speaker' in tsv_df.columns:
                tsv_df['tier'] = tsv_df['speaker']
            else:
                tsv_df['tier'] = 'Text'
            tsv_df['tmin'] = tsv_df.start/1000
            tsv_df['tmax'] = tsv_df.end/1000
            tg = TextGrid.create_from_table(tsv_df)
            tg.write_textgrid_file(out_file, check=False)
        except Exception as err:
            print(err)
            raise err
    return None

if __name__ == '__main__':
    # main
    # in_dir 一级目录
    print("TSV转TextGrid工具v1.0")
    in_dir = IN_DIR
    out_dir = OUT_DIR
    args = sys.argv
    if len(args)==3:
        in_dir = args[1]
        out_dir = args[2]
    print('\n输入目录:', in_dir)
    print('输出目录:', out_dir)


    dir_list = os.listdir(in_dir)
    print('\n找到的文件:', dir_list)

    print('\n开始处理:')
    for one_path in dir_list:
        full_in_dir_path = os.path.join(in_dir, one_path)
        if os.path.isfile(full_in_dir_path):
            convert_file(full_in_dir_path, out_dir)

    input('\n处理完成，请按Enter键退出')
    exit()
