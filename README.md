# textgrid_editor

`textgrid_editor` is yet another python module for modifying with Praat TextGrid files. Also provides APIs for editing TextGrids and converting between TextGrids and pandas DataFrames.

## Features:

1. Regex Parsing: It uses regular expressions (re) to identify the specific structure of a Praat file (headers, item classes etc.).

2. Pandas Integration: It stores the annotation data in a Pandas DataFrame. This makes it very powerful for analyzing, filtering, or modifying large amounts of phonetic data.

3. Gap Filling: It includes a utility function fill_interval_df_gaps to "repair" TextGrids by filling in silent/empty spaces between defined intervals, ensuring the timeline is continuous.

4. Serialization: It can convert the Python object back into the text string format required by Praat files. Dict and DataFrame are also supported.

5. Support TextGrid editing, that is, cropping/extending and concatenating TextGrids.

## TODO:

[ ] 将转换脚本作为打包的一部分提供
[ ] 支持处理非标准的TextGrid文件, 并修复存在的错误
[ ] 支持short format TextGrid读写

## How to install:

Install this package via `pip`, like so:

```bash
pip install textgrid_editor
```

You also can put this code in your working directory or in your `$PYTHONPATH`.

