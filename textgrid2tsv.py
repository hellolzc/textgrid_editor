#!/usr/bin/env python
#_*_encoding:utf-8_*_
# hellolzc 2018-2025
import os
import sys
import re
import pandas as pd

sys.path.append('./src')
from textgrid_editor import TextGrid

# 查找in目录下TextGrid文件并转换成TSV文件

IN_DIR = './tests/in'
OUT_DIR = './tests/out'

# %%
def read_tsv(filepath, split='\t'):
    df = pd.read_csv(filepath, encoding='utf-8', quoting=3, sep=split)  #  QUOTE_NONE (3)
    return df

def save_tsv(filepath, df, split='\t'):
    df.to_csv(filepath, sep=split, header=True, index=False, encoding='utf-8', quoting=3)  # csv.QUOTE_NONE (3)

def convert_file(old_path, out_dir, tier_indexes=None):
    (head, tail) = os.path.split(old_path)
    name, ext = os.path.splitext(tail)

    if ext == '.TextGrid':
        print("process name: %s"%name, end='\t')
        out_name = os.path.join(out_dir, name + '.tsv')
        try:
            tg = TextGrid.read_textgrid_file(old_path)
            tsv_df = tg.down_to_table(tier_indexes=tier_indexes)
            tsv_df['start'] = (tsv_df.tmin*1000).round().astype('Int64')
            tsv_df['end'] = (tsv_df.tmax*1000).round().astype('Int64')
            tsv_df['speaker'] = tsv_df.tier
            tsv_df['dur'] = tsv_df.end - tsv_df.start
            tsv_df = tsv_df[['start', 'end', 'dur', 'speaker', 'text']]
            save_tsv(out_name, tsv_df)
        except Exception as err:
            print(err)
            raise err
    return None

if __name__ == '__main__':
    # main
    # in_dir 一级目录
    print("TextGrid转换工具v1.0")
    in_dir = IN_DIR
    out_dir = OUT_DIR
    args = sys.argv
    if len(args)==3:
        in_dir = args[1]
        out_dir = args[2]
    print('\n输入目录:', in_dir)
    print('输出目录:', out_dir)


    dir_list = os.listdir(in_dir)
    dir_list.sort()
    print('\n找到的文件:', dir_list)

    print('\n开始处理:')
    for one_path in dir_list:
        full_in_dir_path = os.path.join(in_dir, one_path)
        if os.path.isfile(full_in_dir_path):
            convert_file(full_in_dir_path, out_dir)

    input('\n处理完成，请按Enter键退出')
    exit()
