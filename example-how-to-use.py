#!/usr/bin/env python
#_*_encoding:utf-8_*_
# hellolzc 2025
import os
import sys
import pandas as pd
sys.path.append('./src')
from textgrid_editor import TextGrid, IntervalTier



def example_create_and_save():
    print("=== 示例 1: DataFrame -> TextGrid ===")
    
    # 1. 准备数据
    # TextGrid.create_from_table 需要 DataFrame 包含以下列:
    # 'tmin' (开始时间), 'tmax' (结束时间), 'text' (标注内容), 'tier' (层级名称)
    data = [
        # words 层
        {'tmin': 0.0, 'tmax': 0.6, 'text': 'hello', 'tier': 'words'},
        {'tmin': 0.6, 'tmax': 1.2, 'text': 'world', 'tier': 'words'},
        # phones 层 (音素)
        {'tmin': 0.0, 'tmax': 0.2, 'text': 'h', 'tier': 'phones'},
        {'tmin': 0.2, 'tmax': 0.4, 'text': 'e', 'tier': 'phones'},
        {'tmin': 0.4, 'tmax': 0.6, 'text': 'l', 'tier': 'phones'},
        {'tmin': 0.6, 'tmax': 1.2, 'text': 'w', 'tier': 'phones'}
    ]
    
    df = pd.DataFrame(data)
    print("原始 DataFrame:")
    print(df)

    # 2. 转换为 TextGrid 对象
    # create_from_table 会自动根据 'tier' 列拆分成不同的 IntervalTier
    # 并调用 fill_interval_df_gaps 自动填充中间的空隙（如果有的话）
    tg = TextGrid.create_from_table(df)

    # 3. 保存文件
    output_filename = 'example_output_1.TextGrid'
    tg.write_textgrid_file(output_filename)
    print(f"\n成功保存文件到: {output_filename}")
    
    # 验证：打印一下生成的 TextGrid 信息
    print(f"TextGrid 总时长: {tg.xmin} - {tg.xmax}")
    print(f"包含的层级: {[t.name for t in tg.tiers]}")



def example_process_and_merge():
    print("\n=== 示例 2: 读取 -> Resize -> Concat -> 编辑 -> 保存 ===")

    # 1. 读取第一个 TextGrid (复用示例 1 生成的文件)
    filename1 = 'example_output_1.TextGrid'
    try:
        tg1 = TextGrid.read_textgrid_file(filename1)
    except FileNotFoundError:
        print(f"错误：请先运行示例 1 生成 {filename1}")
        return

    # 2. 创建第二个 TextGrid (为了演示，这里手动创建一个简单的)
    # 假设这是另一段录音 "Part Two", 时长 1.0s
    data2 = [{'tmin': 0.0, 'tmax': 1.0, 'text': 'part_two', 'tier': 'words'},
             {'tmin': 0.0, 'tmax': 1.0, 'text': 'p2', 'tier': 'phones'}]
    tg2 = TextGrid.create_from_table(pd.DataFrame(data2))

    print(f"原始 TG1 时长: {tg1.xmax}, 原始 TG2 时长: {tg2.xmax}")

    # 3. Resize (截取)
    # 从 TG1 截取前 0.5秒
    tg1_cut = tg1.resize_textgrid(start_time=0.0, stop_time=0.5)
    # 从 TG2 截取前 0.8秒
    tg2_cut = tg2.resize_textgrid(start_time=0.0, stop_time=0.8)

    print(f"截取后 TG1 时长: {tg1_cut.xmax}, 截取后 TG2 时长: {tg2_cut.xmax}")

    # 4. 拼接 (Concat)
    tg_merged = tg1_cut.concat_a_textgrid(tg2_cut)
    
    current_total_duration = tg_merged.xmax
    print(f"拼接后总时长: {current_total_duration} (预期: 0.5 + 0.8 = 1.3)")

    # 5. 添加一个新的 Tier 指示来源
    # 我们需要创建一个 IntervalTier，包含两个 interval
    # 第一个 interval: 0.0 - 0.5 (来源: File1)
    # 第二个 interval: 0.5 - 1.3 (来源: File2)
    
    meta_data = [
        {'xmin': 0.0, 'xmax': tg1_cut.xmax, 'text': 'Source_File_1'},
        {'xmin': tg1_cut.xmax, 'xmax': current_total_duration, 'text': 'Source_File_2'}
    ]
    meta_df = pd.DataFrame(meta_data)
    
    # 使用提供的 IntervalTier 类
    # 注意：IntervalTier 初始化需要一个包含 xmin, xmax, text 的 DataFrame
    new_tier = IntervalTier(name='Metadata', interval_df=meta_df)
    
    # 将新 tier 添加到 TextGrid
    tg_merged.add_tier(new_tier)

    # 6. 保存最终结果
    output_filename = 'example_merged_edited.TextGrid'
    tg_merged.write_textgrid_file(output_filename)
    print(f"编辑后的 TextGrid 已保存至: {output_filename}")
    
    # 打印最终层级结构供检查
    print("最终层级列表:")
    for i, tier in enumerate(tg_merged.tiers):
        print(f"  Tier {i+1}: {tier.name} (Items: {tier.size})")

    print("Table:")
    print(tg_merged.down_to_table())


if __name__ == "__main__":
    # 示例 1：使用 Pandas DataFrame 创建 TextGrid 并保存
    # 这个示例展示了如何利用 `create_from_table` 接口，将结构化的表格数据转换为 Praat 可读的 `.TextGrid` 文件。
    example_create_and_save()
    # 示例 2：读取、Resize、拼接并添加新 Tier.
    # 这个示例模拟了一个音频剪辑场景：我们需要从两个不同的 TextGrid 文件中截取片段，将它们拼在一起，并添加一个新的层级来标记每一段的来源。
    example_process_and_merge()

""" 执行输出结果：
=== 示例 1: DataFrame -> TextGrid ===
原始 DataFrame:
   tmin  tmax   text    tier
0   0.0   0.6  hello   words
1   0.6   1.2  world   words
2   0.0   0.2      h  phones
3   0.2   0.4      e  phones
4   0.4   0.6      l  phones
5   0.6   1.2      w  phones

成功保存文件到: example_output_1.TextGrid
TextGrid 总时长: 0.0 - 1.2
包含的层级: ['words', 'phones']

=== 示例 2: 读取 -> Resize -> Concat -> 编辑 -> 保存 ===
原始 TG1 时长: 1.2, 原始 TG2 时长: 1.0
截取后 TG1 时长: 0.5, 截取后 TG2 时长: 0.8
拼接后总时长: 1.3 (预期: 0.5 + 0.8 = 1.3)
编辑后的 TextGrid 已保存至: example_merged_edited.TextGrid
最终层级列表:
  Tier 1: words (Items: 2)
  Tier 2: phones (Items: 4)
  Tier 3: Metadata (Items: 2)
Table:
   tmin  tmax           text      tier
0   0.0   0.5          hello     words
1   0.0   0.2              h    phones
2   0.0   0.5  Source_File_1  Metadata
3   0.2   0.4              e    phones
4   0.4   0.5              l    phones
5   0.5   1.3       part_two     words
6   0.5   1.3             p2    phones
7   0.5   1.3  Source_File_2  Metadata
"""