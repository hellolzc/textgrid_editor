#!/usr/bin/env python
#_*_encoding:utf-8_*_
# hellolzc 2018-2025
'''A tool to process Praat TextGrids.

1. Read and parse the TextGrid format files used by the Praat.
2. Converting between TextGrids and pandas DataFrames.
3. Resize a TextGrid; Concat two textgrid.
4. TODO:支持处理非标准的TextGrid文件, 并修复存在的错误

'''

from typing import List
import re
import copy
import pandas as pd

# 定义文件头格式和intervals格式
intRegex = r'([0-9]+)'
floatRegex = r'([-+]?[0-9]*\.?[0-9]+)'

fileHeadRegex = r'''File type = \"ooTextFile\"\s*
Object class = \"TextGrid\"\s*
\s*
\s*xmin = ''' + floatRegex + r'''\s*
\s*xmax = ''' + floatRegex + r'''\s*
tiers\? <exists>\s*
size =\s+''' + intRegex

tierHeadRegex = r'''item \['''+ intRegex + r'''\]:\s*
\s*class = "(.*?)"\s*
\s*name = "(.*?)"\s*
\s*xmin = ''' + floatRegex + r'''\s*
\s*xmax = ''' + floatRegex + r'''\s*
\s*(?:intervals|points): size = ''' + intRegex

#intervals \[([0-9]+)\]:\s*\n\s*xmin = ([-+]?[0-9]*\.?[0-9]+)\s*\n\s*xmax = ([-+]?[0-9]*\.?[0-9]+)\s*\n\s*text = "((?:[^"]|"")*)"
intervalsRegex = r'''intervals \['''+ intRegex + r'''\]:\s*
\s*xmin = ''' + floatRegex + r'''\s*
\s*xmax = '''+ floatRegex + r'''\s*
\s*text = "((?:[^"]|"")*)"'''

# points \[([0-9]+)\]:\s*\n\s*number = ([-+]?[0-9]*\.?[0-9]+)\s*\n\s*mark = "((?:[^"]|"")*)"
pointsRegex = r'''points \['''+ intRegex + r'''\]:\s*
\s*number = ''' + floatRegex + r'''\s*
\s*mark = "((?:[^"]|"")*)"'''



LOG_LEVEL = 2
LEVEL_DICT = {
    'DEBUG':0,
    'INFO':1,
    'WARNING':2,
    'ERROR':3
}

def _log(level, string):
    level = level.upper()
    level_num = LEVEL_DICT[level]
    if level_num >= LOG_LEVEL:
        print(level + ':' + string)



class IntervalTier(object):
    """
    Represents Praat IntervalTiers.

    This class store all data in self.data, which is a dict.
    Intervals are store in panda DataFrames.
    {
        'class': "IntervalTier"
        'name' : string
        'xmin': float
        'xmax': float
        'size': int
        'intervals' : panda DataFrame: 3 columns : xmin, xmax, text
    }

    """

    def __init__(self, name='', interval_df=None):
        """Construct IntervalTier with name and interval_df
        interval_df: panda DataFrame: 3 columns : xmin, xmax, text
        """
        self.name = name
        self.intervals = interval_df
        if interval_df is None:
            self.intervals = pd.DataFrame(columns=['xmin', 'xmax', 'text'])

    def to_dict(self):
        self.data = {
                'class': "IntervalTier",
                'name' : self.name,
                'xmin': self.xmin,
                'xmax': self.xmax,
                'size': self.size,
                'intervals' : self.intervals
        }

    def __copy__(self):
        return self.__init__(name=self.name, interval_df=self.intervals.copy())

    @property
    def xmin(self):
        if len(self.intervals) == 0:
            return None
        return self.intervals.iloc[0]['xmin']
    
    @property
    def xmax(self):
        if len(self.intervals) == 0:
            return None
        return self.intervals.iloc[-1]['xmax']

    @property
    def size(self):
        return len(self.intervals)

    @staticmethod
    def parse_tier(ftext):
        """ parse text, return a dict.
            Parameter:
                ftext: string
            Return:
                tier: a object of IntervalTier:
                    {
                        'class': "IntervalTier"
                        'name' : string
                        'xmin': float
                        'xmax': float
                        'size': int
                        'intervals' : panda DataFrame: 3 columns : xmin, xmax, text
                    }
        """
        match = re.search(tierHeadRegex, ftext)
        if match is None:
            _log('ERROR', "Tier Head Not Found! ")
            raise Exception("No Tier Head")

        class_name = match.group(2)
        assert class_name == "IntervalTier", "Unsupported tier type! :%s"%class_name
        name = match.group(3)
        xmin = float(match.group(4))
        xmax = float(match.group(5))
        size = int(match.group(6))

        itemDict = {
                    'class': "IntervalTier",
                    'name' : name,
                    'xmin': xmin,
                    'xmax': xmax,
                    'size': size,
                    'intervals' : None
        }

        interval_df = pd.DataFrame(columns=['xmin', 'xmax', 'text'])

        if size > 0:
            all_matchs = re.findall(intervalsRegex, ftext) # , flags=re.DOTALL|re.MULTILINE
            #print(all_matchs)
            assert all_matchs , "intervals not found."
            for row_id, match_tuple in enumerate(all_matchs):
                #print(match_tuple)
                no = match_tuple[0]
                xmin = float(match_tuple[1])
                xmax = float(match_tuple[2])
                text = match_tuple[3]
                if xmax < xmin:
                    _log('Error', "intervals time error, xmin %f > xmax %f"%(xmin, xmax))
                    raise Exception("File Not Read")
                interval_df.loc[row_id] = [xmin, xmax, text]
                if (row_id > 0):
                    last_xmax = interval_df.loc[row_id-1, 'xmax'] 
                    if last_xmax > xmin:
                        _log('Error', "intervals time error, last xmax %f > this xmin %f"%(last_xmax, xmin))
                        raise Exception("File Not Read")
            itemDict['intervals'] = interval_df
            if row_id+1 != size:
                _log('Warning', "intervals number error, '%s' should have %d interval, get %d interval"%(name, size, row_id+1))

        tier = IntervalTier(name, interval_df)
        if tier.xmin != itemDict['xmin']:
            _log('Error', "intervals time error, first xmin %f != tier.xmin %f"%(tier.xmin, itemDict['xmin']))
        if tier.xmax != itemDict['xmax']:
            _log('Error', "intervals time error, last xmax %f != tier.xmax %f"%(tier.xmax, itemDict['xmax']))

        return tier

    def to_tier_ftext(self):
        "convert a tier to text, return ftext"
        ftext = ''
        ftext += '''        class = "IntervalTier"
        name = "%s"
        xmin = %g
        xmax = %g
        intervals: size = %d
'''%(self.name, self.xmin, self.xmax, self.size)

        df = self.intervals
        for indx in range(self.size):
            ftext += '''        intervals [%d]:
            xmin = %g
            xmax = %g
            text = "%s"
'''%(indx+1, df.at[indx, 'xmin'], df.at[indx, 'xmax'], df.at[indx, 'text'])

        return ftext

    @staticmethod
    def fill_interval_df_gaps(df, tier_xmin=None, tier_xmax=None, item_name=''):
        """ Repair the TextGrid interval tier. Fill in the missing Intervals.
        Possible missing parts include missing headers, missing tails, and missing parts in the middle.
        """
        # 添加缺失的interval
        # print(df)
        df2 = df.copy()
        total_rows = len(df)
        total_rows2 = total_rows
        indx = 0
        indx2 = 0
        if tier_xmin is None:
            tier_xmin = df.iloc[0]['xmin']
        if tier_xmax is None:
            tier_xmax = df.iloc[-1]['xmax']
        last_interval_xmax = tier_xmin
        while indx < total_rows:
            interval_xmin = df.loc[indx, 'xmin']
            interval_xmax = df.loc[indx, 'xmax']
            interval_text = df.loc[indx, 'text']
            if interval_xmin > last_interval_xmax:
                print("Info", "'%s' %f-%f: insert a empty interval." % (item_name, last_interval_xmax, interval_xmin))
                df2.loc[indx2, :] = [last_interval_xmax, interval_xmin, '']
                indx2 += 1
                total_rows2 += 1
            df2.loc[indx2, :] = [interval_xmin, interval_xmax, interval_text]
            last_interval_xmax = interval_xmax
            indx += 1
            indx2 += 1
        # 尾部缺失 单独处理
        if tier_xmax is not None and (tier_xmax > last_interval_xmax):
            print("Info", "'%s' %f-%f: insert a empty interval." % (item_name, last_interval_xmax, tier_xmax))
            df2.loc[indx2, :] = [last_interval_xmax, tier_xmax, '']
            indx2 += 1
            total_rows2 += 1
        # print(df2)
        return df2


    def to_table_df(self):
        mask = self.intervals.text != ''
        table_df = pd.DataFrame(data={
                    'tmin': self.intervals.loc[mask, 'xmin'],
                    'tmax': self.intervals.loc[mask, 'xmax'],
                    'text': self.intervals.loc[mask, 'text']
                })
        return table_df, self.name, self.xmin, self.xmax, self.size

    @staticmethod
    def from_table_df(table_df, name='', xmin=None, xmax=None):
        interval_df = pd.DataFrame(data={
            'xmin': table_df['tmin'],
            'xmax': table_df['tmax'],
            'text': table_df['text']
        }).reset_index(drop=True)
        interval_df = IntervalTier.fill_interval_df_gaps(interval_df, tier_xmin=xmin, tier_xmax=xmax)
        return IntervalTier(name, interval_df)


    def resize_tier(self, start_time=0.0, stop_time=None, append=False, auto_extend=False):
        ''' Resize the content of a tier, the end time must be specified.
        (1) If stop_time is shorter than a tier's end time, the content of that tier will be truncated. 
        (2) If stop_time exceeds the tier end, the tier end is used by default.
        When append=True, empty intervals are appended; when append=False, tier duration will be based on the interval duration.
        If both append=True and auto_extend=True, the last interval is extended if it is empty.
        '''
        assert stop_time
        assert start_time < stop_time , "start time is bigger than stop time!"
        interval_df1 = self.intervals
        mask = (interval_df1.xmin < stop_time) & (interval_df1.xmax > start_time)
        interval_df2 = interval_df1[mask].reset_index(drop=True)
        # 时间偏移, 起点会被修正为0.0（截断第一个interval）
        interval_df2.xmin = interval_df2.xmin - start_time
        interval_df2.xmax = interval_df2.xmax - start_time
        interval_df2.loc[0, 'xmin'] = 0.0
        # 处理结尾
        item_xmax = stop_time - start_time
        total_rows = len(interval_df2)
        last_interval_xmax = interval_df2.loc[total_rows-1, 'xmax']

        if item_xmax < last_interval_xmax:
            # 截断最后一个interval
            interval_df2.loc[total_rows-1, 'xmax'] = item_xmax
        elif item_xmax > last_interval_xmax:
            if append:
                if auto_extend and (interval_df2.loc[total_rows-1, 'text'].strip() in ['sil', '', '<NOISE>']):
                    # 延长最后一个interval
                    interval_df2.loc[total_rows-1, 'xmax'] = item_xmax
                else:
                    # 追加空白interval
                    interval_df2.loc[total_rows] = [last_interval_xmax, item_xmax, '']
                    total_rows += 1
            else:
                # 结束时间以最后一个interval为准
                item_xmax = last_interval_xmax

        return IntervalTier(self.name, interval_df2)

    def concat_a_tier(self, item2, offset):
        assert self.name == item2.name, "item names mismatch! %s!=%s"%(self.name, item2.name)
        assert self.xmax == offset, 'time mismatch!'
        interval_df1 = self.intervals
        interval_df2 = item2.intervals.copy()
        assert interval_df1.iloc[-1]['xmax'] == offset, "end time of last interval don't match end time of the tier"
        interval_df2.xmin = interval_df2.xmin + offset
        interval_df2.xmax = interval_df2.xmax + offset
        new_interval_df = pd.concat([interval_df1, interval_df2], ignore_index=True)
        return IntervalTier(self.name, new_interval_df)


class TextTier(object):
    """
    Represents Praat TextTiers.

    This class store all data in self.data, which is a dict.
    Points are store in panda DataFrames.
    {
        'class': "TextTier"
        'name' : string
        'xmin': float
        'xmax': float
        'size': int
        'TextTier' : panda DataFrame: 2 columns : xmin(number), text(mark)
    }

    """

    def __init__(self, name='', point_df=None, xmin=None, xmax=None):
        """Construct TextTier with name and interval_df
        interval_df: panda DataFrame: 2 columns : xmin(number), text(mark)
        """
        self.name = name
        self.points = point_df
        self._xmin = xmin
        self._xmax = xmax
        if point_df is None:
            self.points = pd.DataFrame(columns=['xmin', 'text'])

    def to_dict(self):
        self.data = {
                'class': "TextTier",
                'name' : self.name,
                'xmin': self.xmin,
                'xmax': self.xmax,
                'size': self.size,
                'points' : self.points
        }

    def __copy__(self):
        return self.__init__(name=self.name, point_df=self.point_df.copy(), xmin=self.xmin, xmax=self.xmax)

    @property
    def xmin(self):
        if self._xmin is not None:
            return self._xmin
        if len(self.points) == 0:
            return None
        return self.points.iloc[0]['xmin']
    
    @property
    def xmax(self):
        if self._xmin is not None:
            return self._xmin
        if len(self.points) == 0:
            return None
        return self.points.iloc[-1]['xmin']

    @property
    def size(self):
        return len(self.points)

    @staticmethod
    def parse_tier(ftext):
        """ parse text, return a dict.
            Parameter:
                ftext: string
            Return:
                tier: a object of TextTier:
                    {
                        'class': "TextTier"
                        'name' : string
                        'xmin': float
                        'xmax': float
                        'size': int
                        'points' : panda DataFrame: 2 columns : xmin(number), text(mark)
                    }
        """
        match = re.search(tierHeadRegex, ftext)
        if match is None:
            _log('ERROR', "Tier Head Not Found! ")
            raise Exception("No Tier Head")

        class_name = match.group(2)
        assert class_name == "TextTier", "Unsupported tier type! :%s"%class_name
        name = match.group(3)
        xmin = float(match.group(4))
        xmax = float(match.group(5))
        size = int(match.group(6))

        itemDict = {
                    'class': "TextTier",
                    'name' : name,
                    'xmin': xmin,
                    'xmax': xmax,
                    'size': size,
                    'points' : None
        }

        point_df = pd.DataFrame(columns=['xmin', 'text'])

        if size > 0:
            all_matchs = re.findall(pointsRegex, ftext) # , flags=re.DOTALL|re.MULTILINE
            #print(all_matchs)
            assert all_matchs , "points not found."
            for row_id, match_tuple in enumerate(all_matchs):
                #print(match_tuple)
                no = match_tuple[0]
                xmin = float(match_tuple[1])
                text = match_tuple[2]
                point_df.loc[row_id] = [xmin, text]

            itemDict['points'] = point_df
            if row_id+1 != size:
                _log('Warning', "points number error, '%s' should have %d points, get %d points"%(name, size, row_id+1))

        tier = TextTier(name, point_df, xmin=itemDict['xmin'], xmax=itemDict['xmax'])

        return tier

    def to_tier_ftext(self):
        "convert a tier to text, return ftext"
        ftext = ''
        ftext += '''        class = "TextTier"
        name = "%s"
        xmin = %g
        xmax = %g
        points: size = %d
'''%(self.name, self.xmin, self.xmax, self.size)

        df = self.points
        for indx in range(self.size):
            ftext += '''        points [%d]:
            number = %g
            mark = "%s"
'''%(indx+1, df.at[indx, 'xmin'], df.at[indx, 'text'])

        return ftext


    def to_table_df(self):
        table_df = pd.DataFrame(data={
                    'tmin': self.points['xmin'],
                    'tmax': None,
                    'text': self.points['text']
                })
        return table_df, self.name, self.xmin, self.xmax, self.size

    @staticmethod
    def from_table_df(table_df, name='', xmin=None, xmax=None):
        point_df = pd.DataFrame(data={
            'xmin': table_df['tmin'],
            'text': table_df['text']
        }).reset_index(drop=True)
        return TextTier(name, point_df, xmin, xmax)


    def resize_tier(self, start_time=0.0, stop_time=None, append=False, auto_extend=False):
        ''' Resize the content of a tier, the end time must be specified.'''
        assert stop_time
        assert start_time < stop_time , "start time is bigger than stop time!"
        points_df1 = self.points
        mask = (points_df1.xmin < stop_time) & (points_df1.xmin > start_time)
        points_df2 = points_df1[mask].reset_index(drop=True)
        # 时间偏移, 起点会被修正为0.0（截断第一个interval）
        points_df2.xmin = points_df2.xmin - start_time
        return TextTier(self.name, points_df2)

    def concat_a_tier(self, item2, offset):
        assert self.name == item2.name, "item names mismatch! %s!=%s"%(self.name, item2.name)
        point_df1 = self.points
        point_df2 = item2.points.copy()
        point_df2.xmin = point_df2.xmin + offset
        new_point_df = pd.concat([point_df1, point_df2], ignore_index=True)
        return TextTier(self.name, new_point_df)


class TextGrid(object):
    """
    A tool to process Praat TextGrids.
    self.tiers: Items are IntervalTier or TextTier.

    The intervals are store in pandas DataFrames.
    """
    def __init__(self, tier_list: List=None):
        ''' create a empty textgrid dict
        '''
        self.tiers = tier_list
        if not tier_list:
            self.tiers = []


    def to_dict(self):
        '''create a empty textgrid dict
        TextGrid is convert a Python dict, where
            textGridDict: A dict:
            {
                'xmin': float
                'xmax': float
                'size': int
                'tiers':[item1, item2, ...]
            }
            item: a dict, for IntervalTier:
            {
                'class': "IntervalTier"
                'name' : string
                'xmin': float
                'xmax': float
                'size': int
                'intervals' : pandas DataFrame: 3 columns : xmin, xmax, text
            }
        '''
        tier_dict_list = [x.to_dict() for x in self.tiers]

        out_tg_dict = {
                    'xmin': self.xmin,
                    'xmax': self.xmax,
                    'size': self.size,
                    'tiers': tier_dict_list
        }
        return out_tg_dict

    def __copy__(self):
        return self.__init__(tier_list=[x.copy() for x in self.tiers])

    @property
    def xmin(self):
        if len(self.tiers) == 0:
            return None
        xmin = min((x.xmin for x in self.tiers))
        return xmin
    
    @property
    def xmax(self):
        if len(self.tiers) == 0:
            return None
        xmax = max((x.xmax for x in self.tiers))
        return xmax

    @property
    def size(self):
        return len(self.tiers)

    def add_tier(self, tier_item):
        ''' add a new tier to TextGrid Dict, return a dict.  '''
        self.tiers.append(tier_item)      

    @staticmethod
    def parse_textgrid_ftext(ftext):
        '''parse a praat TextGrid file text, return a dict.
        Return:
            textGridDict: A dict contains:
            {
                'xmin': float
                'xmax': float
                'size': int
                'tiers':[item1, item2, item3]
            }
            itemx is a dict, see above
        '''
        # 检查文件头
        match = re.search(fileHeadRegex, ftext)
        if match is None:
            _log('ERROR', "Couldn't find head data of this file!")
            raise Exception("No File Head")

        f_xmin = float(match.group(1))
        f_xmax = float(match.group(2))
        f_size = int(match.group(3))

        textGridDict = {
                        'xmin': f_xmin,
                        'xmax': f_xmax,
                        'size': f_size,
                        'tiers':[]
                    }

        # 切割每个item
        tier_start_pos = []
        tier_class_names = []
        pattern = re.compile(tierHeadRegex)
        pos = 0
        for tier_no in range(f_size):
            match = pattern.search(ftext, pos)
            if match is None:
                _log('ERROR', "Cannot found Tier %d!"%(tier_no + 1))
                raise Exception("No Tier")
            tier_start_pos.append(match.start(0))
            tier_class_names.append(match.group(2))
            pos = match.end(0)

        # 处理每个item
        tier_start_pos.append(len(ftext))
        #print(tier_start_pos)
        for tier_no in range(f_size):
            tier_text = ftext[tier_start_pos[tier_no]:tier_start_pos[tier_no+1]]
            #print(tier_text)

            if tier_class_names[tier_no] == 'IntervalTier':
                one_item = IntervalTier.parse_tier(tier_text)
            else:  # 
                one_item = TextTier.parse_tier(tier_text)
            textGridDict['tiers'].append(one_item)

        tg = TextGrid(tier_list=textGridDict['tiers'])
        if tg.xmin != textGridDict['xmin']:
            _log('Error', "TextGrid time error, first xmin %f != file.xmin %f"%(tg.xmin, textGridDict['xmin']))
        if tg.xmax != textGridDict['xmax']:
            _log('Error', "TextGrid time error, last xmax %f != file.xmax %f"%(tg.xmax, textGridDict['xmax']))

        return tg

    def to_textgrid_ftext(self):
        "convert to TextGrid file, return ftext"
        ftext = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = %g
xmax = %g
tiers? <exists>
size = %d
item []:
'''%(self.xmin, self.xmax, self.size)

        for indx, item in enumerate(self.tiers):
            ftext += '    item [%d]:\n' % (indx+1)
            ftext += item.to_tier_ftext()

        return ftext

    def write_textgrid_file(self, filename, check=True):
        "write TextGrid file"
        with open(filename, 'w', encoding='utf-16-be') as f:
            ftext = self.to_textgrid_ftext()
            if check:
                self.parse_textgrid_ftext(ftext)
            f.write('\ufeff') # BOM
            f.write(ftext)

    @staticmethod
    def _read_text_all_lines(filename, encoding='utf-16-be'):
        ftext = ''
        with open(filename, encoding=encoding) as f:
            for line in f:
                ftext += line
        return ftext

    @staticmethod
    def read_textgrid_file(filename, encoding=None):
        """ parse a praat TextGrid file, return a dict.
            Parameter:
                filename: file name
            Return:
                textGridDict
        """
        # 读取TextGrid文件，检查编码
        if encoding is None:
            try:
                ftext = TextGrid._read_text_all_lines(filename)
            except UnicodeDecodeError:
                _log('Warning', "File %s is not encoded with UTF-16-BE! Trying GBK."%filename)
                ftext = TextGrid._read_text_all_lines(filename, 'gbk')
        else:
            ftext = TextGrid._read_text_all_lines(filename, encoding)
        textGridDict = TextGrid.parse_textgrid_ftext(ftext)
        return textGridDict


    def down_to_table(self, tier_indexes=None):
        ''' convert a dataframe to TextGrid object.
        parse a praat TextGrid file text, return a dict.
        pandas DataFrame: 3 or 4 columns : tmin, tmax, [tier], text 
        '''
        if self.size == 0:
            return None

        if tier_indexes is None:
            tier_indexes = list(range(self.size))

        table_list = []
        for index in tier_indexes:
            tier_table_df, tier_name, _, _, _ = self.tiers[index].to_table_df()
            tier_table_df['tier'] = tier_name
            table_list.append(tier_table_df)
            print(tier_table_df)
        table_df = pd.concat(table_list).sort_values('tmin')

        return table_df

    @staticmethod
    def create_from_table(table_df, tier_names=None):
        ''' convert a dataframe to TextGrid object.
        parse a praat TextGrid file text, return a dict.
        pandas DataFrame: 3 or 4 columns : tmin, tmax, [tier], text 
        '''
        # 每个item
        if tier_names is None:
            if 'tier' in table_df:
                tier_names = table_df['tier'].unique()
            else:
                tier_names = ['']
        
        tier_list = []
        for name in tier_names:
            if name == '':
                tier_table_df = table_df
            else:
                tier_table_df = table_df.loc[table_df['tier'] == name]

            if tier_table_df['tmax'].isna().all():
                one_item = TextTier.from_table_df(tier_table_df, name=name)
            else:
                one_item = IntervalTier.from_table_df(tier_table_df, name=name)
            tier_list.append(one_item)
        tg = TextGrid(tier_list)

        return tg

    def resize_textgrid(self, start_time=None, stop_time=None):
        """ resize the textgrid to specified duration.
        """
        out_tier_list = []
        for indx, tier in enumerate(self.tiers):
            out_tier_list.append(tier.resize_tier(start_time=start_time, stop_time=stop_time, append=True, auto_extend=True))

        return TextGrid(out_tier_list)

    def concat_a_textgrid(self, tg2):
        """ concatenate self with another TextGrid, return a TextGrid.
            Parameter:
                self: TextGrid
                tg2: TextGrid
            Return:
                out_tg: TextGrid
        """
        offset = self.xmax
        f_xmin = 0.0
        f_xmax = offset + tg2.xmax

        out_tier_list = []

        for indx, tier in enumerate(self.tiers):
            new_item = tier.concat_tier(tg2.tiers[indx], offset)
            out_tier_list.append(new_item)

        return TextGrid(out_tier_list)
