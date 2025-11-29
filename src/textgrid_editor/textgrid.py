#!/usr/bin/env python
#_*_encoding:utf-8_*_
# hellolzc 2018-2025
'''A tool to process Praat TextGrids.

1. Read and parse the TextGrid format files used by the Praat.
2. Converting between TextGrids and pandas DataFrames.
3. Resize a TextGrid; Concat two textgrid.

'''

from typing import List, Dict, Optional, Any
import re
import copy
import logging
import pandas as pd

# Define Regex patterns for file headers and interval/point formats
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



logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)



class IntervalTier(object):
    """
    Represents a Praat IntervalTier.

    This class manages time-aligned text intervals. The core data is stored 
    in a pandas DataFrame for efficient manipulation.

    Attributes:
        name (str): The name of the tier (e.g., "words", "phones").
        intervals (pd.DataFrame): A DataFrame with columns ['xmin', 'xmax', 'text'].
    """

    def __init__(self, name='', interval_df=None):
        """
        Initialize the IntervalTier.

        Args:
            name (str): The name of the tier.
            interval_df (pd.DataFrame, optional): A DataFrame containing existing intervals.
                Must have columns ['xmin', 'xmax', 'text']. Defaults to an empty DataFrame.
        """
        self.name = name
        self.intervals = interval_df
        if interval_df is None:
            self.intervals = pd.DataFrame(columns=['xmin', 'xmax', 'text'])

    def to_dict(self):
        """
        Convert the Tier object into a dictionary representation.

        Returns:
            dict: A dictionary containing class metadata and the intervals DataFrame.
                {
                    'class': "IntervalTier"
                    'name' : string
                    'xmin': float
                    'xmax': float
                    'size': int
                    'intervals' : panda DataFrame: 3 columns : xmin, xmax, text
                }
        """
        self.data = {
                'class': "IntervalTier",
                'name' : self.name,
                'xmin': self.xmin,
                'xmax': self.xmax,
                'size': self.size,
                'intervals' : self.intervals
        }

    def __copy__(self):
        """
        Create a shallow copy of the object, but deep copy the DataFrame 
        to ensure data independence.
        """
        return IntervalTier(name=self.name, interval_df=self.intervals.copy())

    @property
    def xmin(self):
        """
        Get the start time of the IntervalTier.

        Returns:
            float or None: The xmin of the first interval, or None if empty.
        """
        if len(self.intervals) == 0:
            return None
        return self.intervals.iloc[0]['xmin']
    
    @property
    def xmax(self):
        """
        Get the end time of the IntervalTier.

        Returns:
            float or None: The xmax of the last interval, or None if empty.
        """
        if len(self.intervals) == 0:
            return None
        return self.intervals.iloc[-1]['xmax']

    @property
    def size(self):
        """
        Get the number of intervals in the tier.

        Returns:
            int: The count of intervals.
        """
        return len(self.intervals)

    @staticmethod
    def parse_tier(ftext):
        """
        Parse a raw string from a Praat TextGrid file to create an IntervalTier object.

        This method extracts header information and interval data using regex, 
        validates the timeline integrity, and loads the data into a DataFrame.

        Args:
            ftext (str): The raw text string defining a single Tier from a TextGrid file.

        Returns:
            IntervalTier: An instance of the class populated with parsed data.

        Raises:
            Exception: If headers are missing, the tier type is unsupported, 
                       or parsing fails.
        """
        match = re.search(tierHeadRegex, ftext)
        if match is None:
            logger.error("Tier Head Not Found! ")
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
            # Find all interval matches in the text
            all_matchs = re.findall(intervalsRegex, ftext) # , flags=re.DOTALL|re.MULTILINE
            assert all_matchs , "intervals not found."

            for row_id, match_tuple in enumerate(all_matchs):
                no = match_tuple[0]
                xmin = float(match_tuple[1])
                xmax = float(match_tuple[2])
                text = match_tuple[3]
                
                # Validation: Start time cannot be greater than end time
                if xmax < xmin:
                    logger.error("intervals time error, xmin %f > xmax %f"%(xmin, xmax))
                    raise Exception("File Not Read")
                interval_df.loc[row_id] = [xmin, xmax, text]
                
                # Validation: Ensure continuity (no overlapping backwards in time)
                if (row_id > 0):
                    last_xmax = interval_df.loc[row_id-1, 'xmax'] 
                    if last_xmax > xmin:
                        logger.error("intervals time error, last xmax %f > this xmin %f"%(last_xmax, xmin))
                        raise Exception("File Not Read")
            itemDict['intervals'] = interval_df
            if row_id+1 != size:
                logger.warning("intervals number mismatch, '%s' should have %d interval, get %d interval"%(name, size, row_id+1))

        tier = IntervalTier(name, interval_df)
        
        # Verify the parsed boundaries match the header boundaries
        if tier.xmin != itemDict['xmin']:
            logger.warning("intervals time mismatch, first xmin %f != tier.xmin %f"%(tier.xmin, itemDict['xmin']))
        if tier.xmax != itemDict['xmax']:
            logger.warning("intervals time mismatch, last xmax %f != tier.xmax %f"%(tier.xmax, itemDict['xmax']))

        return tier

    def to_tier_ftext(self):
        """
        Convert the IntervalTier object back into Praat-formatted text.

        Returns:
            str: A formatted string suitable for writing to a .TextGrid file.
        """
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
    def fill_interval_df_gaps(df, tier_xmin=None, tier_xmax=None, tier_name=''):
        """
        Repair a TextGrid interval DataFrame by filling in missing time gaps.

        This ensures the timeline is continuous from start to end. 
        It inserts empty intervals ('') where gaps exist between defined intervals.

        Args:
            df (pd.DataFrame): The original intervals DataFrame.
            tier_xmin (float, optional): The global start time for the tier. Defaults to first interval xmin.
            tier_xmax (float, optional): The global end time for the tier. Defaults to last interval xmax.
            tier_name (str, optional): Name of the tier for logging purposes.

        Returns:
            pd.DataFrame: A new DataFrame with no time gaps between intervals.
        """
        # Fill in the missing Intervals.
        # Possible missing parts include missing headers, missing tails, and missing parts in the middle.
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
                logger.info("'%s' %f-%f: insert a empty interval." % (tier_name, last_interval_xmax, interval_xmin))
                df2.loc[indx2, :] = [last_interval_xmax, interval_xmin, '']
                indx2 += 1
                total_rows2 += 1
            df2.loc[indx2, :] = [interval_xmin, interval_xmax, interval_text]
            last_interval_xmax = interval_xmax
            indx += 1
            indx2 += 1

        # Handle trailing gap
        if tier_xmax is not None and (tier_xmax > last_interval_xmax):
            logger.info("'%s' %f-%f: insert a empty interval." % (tier_name, last_interval_xmax, tier_xmax))
            df2.loc[indx2, :] = [last_interval_xmax, tier_xmax, '']
            indx2 += 1
            total_rows2 += 1

        return df2


    def to_table_df(self, include_empty_intervals=False):
        """
        Convert the Tier data into a simplified Pandas DataFrame.

        Args:
            include_empty_intervals (bool): If False, rows with empty text ('') 
                                            will be removed.

        Returns:
            tuple: (DataFrame, name, xmin, xmax, size)
                   The DataFrame contains columns ['tmin', 'tmax', 'text'].
        """
        table_df = pd.DataFrame(data={
                    'tmin': self.intervals['xmin'],
                    'tmax': self.intervals['xmax'],
                    'text': self.intervals['text']
                })
        if not include_empty_intervals:
            mask = self.intervals.text != ''
            table_df = table_df.loc[mask].reset_index(drop=True)
        return table_df, self.name, self.xmin, self.xmax, self.size

    @staticmethod
    def from_table_df(table_df, name='', xmin=None, xmax=None):
        """
        Reconstruct an IntervalTier from a simplified DataFrame.

        This method will automatically repair gaps between intervals using 
        `fill_interval_df_gaps`.

        Args:
            table_df (pd.DataFrame): Must contain ['tmin', 'tmax', 'text'].
            name (str): Name of the tier.
            xmin (float): Start time of the tier.
            xmax (float): End time of the tier.

        Returns:
            IntervalTier: The reconstructed tier object.
        """
        interval_df = pd.DataFrame(data={
            'xmin': table_df['tmin'],
            'xmax': table_df['tmax'],
            'text': table_df['text']
        }).reset_index(drop=True)
        interval_df = IntervalTier.fill_interval_df_gaps(interval_df, tier_xmin=xmin, tier_xmax=xmax, tier_name=name)
        return IntervalTier(name, interval_df)


    def resize_tier(self, start_time=0.0, stop_time=None, append=False, auto_extend=False):
        """
        Crop and resize the tier to a specific time range. The end time must be specified.
        This method shifts timestamps so the new start_time becomes 0.0.

        (1) If stop_time is shorter than a tier's end time, the content of that tier will be truncated. 
        (2) If stop_time exceeds the tier end, the tier end is used by default.
        When append=True, empty intervals are appended; when append=False, tier duration will be based on the interval duration.
        If both append=True and auto_extend=True, the last interval is extended if it is empty.

        Args:
            start_time (float): The absolute time to start cropping from.
            stop_time (float): The absolute time to stop cropping. Must be provided.
            append (bool): If True, allows adding empty intervals if the stop_time 
                           extends beyond existing data.
            auto_extend (bool): If True (and append is True), extends the last 
                                existing interval instead of adding a new empty one,
                                provided the last interval is silence/noise.

        Returns:
            IntervalTier: A new, resized IntervalTier object.
        """
        assert stop_time
        assert start_time < stop_time , "start time is bigger than stop time!"
        interval_df1 = self.intervals
        
        # Select intervals that overlap with the desired time range
        mask = (interval_df1.xmin < stop_time) & (interval_df1.xmax > start_time)
        interval_df2 = interval_df1[mask].reset_index(drop=True)
        # 时间偏移, 起点会被修正为0.0（截断第一个interval）
        interval_df2.xmin = interval_df2.xmin - start_time
        interval_df2.xmax = interval_df2.xmax - start_time
        interval_df2.loc[0, 'xmin'] = 0.0
        
        # Calculate expected new duration
        item_xmax = stop_time - start_time
        total_rows = len(interval_df2)
        last_interval_xmax = interval_df2.loc[total_rows-1, 'xmax']

        # Handle the tail end of the tier
        if item_xmax < last_interval_xmax:
            # Case 1: Truncate the last interval (cut it short)
            interval_df2.loc[total_rows-1, 'xmax'] = item_xmax
        elif item_xmax > last_interval_xmax:
            if append:
                # Case 2: Desired length is longer than data. Extend.
                if auto_extend and (interval_df2.loc[total_rows-1, 'text'].strip() in ['sil', '', '<NOISE>']):
                    # If last interval is silence, just stretch it
                    interval_df2.loc[total_rows-1, 'xmax'] = item_xmax
                else:
                    # Otherwise, append a new empty interval
                    interval_df2.loc[total_rows] = [last_interval_xmax, item_xmax, '']
                    total_rows += 1
            else:
                # Case 3: Do not extend, just clamp to existing data end
                item_xmax = last_interval_xmax

        return IntervalTier(self.name, interval_df2)

    def concat_a_tier(self, item2, offset):
        """
        Concatenate another tier to the end of this one.

        Args:
            item2 (IntervalTier): The tier to append.
            offset (float): The time offset to add to item2's timestamps 
                            (usually self.xmax).

        Returns:
            IntervalTier: A combined tier.
        """
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
    Represents a Praat TextTier (Point Tier).

    Unlike IntervalTiers which have ranges, TextTiers represent specific points in time.
    Data is stored in self.points (pandas DataFrame).
    """

    def __init__(self, name='', point_df=None, xmin=None, xmax=None):
        """
        Construct a TextTier.

        Args:
            name (str): Name of the tier.
            point_df (pd.DataFrame): DataFrame with columns ['xmin', 'text']. 
                                     Note: 'xmin' here represents the point's time 'number'.
            xmin (float): Global start time of the tier.
            xmax (float): Global end time of the tier.
        """
        self.name = name
        self.points = point_df
        self._xmin = xmin
        self._xmax = xmax
        if point_df is None:
            self.points = pd.DataFrame(columns=['xmin', 'text'])

    def to_dict(self):
        """
        Serialize tier to dictionary.
            Points are store in a pandas DataFrame.
            {
                'class': "TextTier"
                'name' : string
                'xmin': float
                'xmax': float
                'size': int
                'TextTier' : panda DataFrame: 2 columns : xmin(number), text(mark)
            }
        """
        self.data = {
                'class': "TextTier",
                'name' : self.name,
                'xmin': self.xmin,
                'xmax': self.xmax,
                'size': self.size,
                'points' : self.points
        }

    def __copy__(self):
        """Deep copy the object."""
        return TextTier(name=self.name, point_df=self.point_df.copy(), xmin=self.xmin, xmax=self.xmax)

    @property
    def xmin(self):
        """Get start time. Returns stored _xmin or time of first point."""
        if self._xmin is not None:
            return self._xmin
        if len(self.points) == 0:
            return None
        return self.points.iloc[0]['xmin']
    
    @property
    def xmax(self):
        """Get end time. Returns stored _xmax or time of last point."""
        if self._xmin is not None:
            return self._xmin
        if len(self.points) == 0:
            return None
        return self.points.iloc[-1]['xmin']

    @property
    def size(self):
        """Number of points in the tier."""
        return len(self.points)

    @staticmethod
    def parse_tier(ftext):
        """
        Parse raw Praat text for a TextTier (PointTier).

        Args:
            ftext (str): Raw text block for this tier.

        Returns:
            TextTier: Parsed object.
        """
        match = re.search(tierHeadRegex, ftext)
        if match is None:
            logger.error("Tier Head Not Found! ")
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
            assert all_matchs , "points not found."
            for row_id, match_tuple in enumerate(all_matchs):
                no = match_tuple[0]
                xmin = float(match_tuple[1])
                text = match_tuple[2]
                point_df.loc[row_id] = [xmin, text]

            itemDict['points'] = point_df
            if row_id+1 != size:
                logger.warning("points number error, '%s' should have %d points, get %d points"%(name, size, row_id+1))

        tier = TextTier(name, point_df, xmin=itemDict['xmin'], xmax=itemDict['xmax'])

        return tier

    def to_tier_ftext(self):
        """Convert object back to Praat text format."""
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


    def to_table_df(self, include_empty_intervals='unused'):
        """
        Convert points to a simplified DataFrame.
        Note: tmax is None for PointTiers.
        """
        table_df = pd.DataFrame(data={
                    'tmin': self.points['xmin'],
                    'tmax': None,
                    'text': self.points['text']
                })
        return table_df, self.name, self.xmin, self.xmax, self.size

    @staticmethod
    def from_table_df(table_df, name='', xmin=None, xmax=None):
        """Construct TextTier from DataFrame."""
        point_df = pd.DataFrame(data={
            'xmin': table_df['tmin'],
            'text': table_df['text']
        }).reset_index(drop=True)
        return TextTier(name, point_df, xmin, xmax)


    def resize_tier(self, start_time=0.0, stop_time=None, append=False, auto_extend=False):
        """ Resize the content of a tier, the end time must be specified.
        Crop points to be within start_time and stop_time, and shift values 
        relative to start_time.
        """
        assert stop_time
        assert start_time < stop_time , "start time is bigger than stop time!"
        points_df1 = self.points
        mask = (points_df1.xmin < stop_time) & (points_df1.xmin > start_time)
        points_df2 = points_df1[mask].reset_index(drop=True)
        
        # Time Offset: Shift timestamps so start_time becomes 0.0
        points_df2.xmin = points_df2.xmin - start_time
        return TextTier(self.name, points_df2)

    def concat_a_tier(self, item2, offset):
        """Combine two TextTiers, shifting the second by offset."""
        assert self.name == item2.name, "item names mismatch! %s!=%s"%(self.name, item2.name)
        point_df1 = self.points
        point_df2 = item2.points.copy()
        point_df2.xmin = point_df2.xmin + offset
        new_point_df = pd.concat([point_df1, point_df2], ignore_index=True)
        return TextTier(self.name, new_point_df)


class TextGrid(object):
    """
    The main class for processing Praat TextGrids.
    
    Attributes:
        tiers (list): A list containing IntervalTier or TextTier objects.
    """
    def __init__(self, tier_list: List=None):
        """
        Initialize a TextGrid.
        
        Args:
            tier_list (list, optional): A list of Tier objects. Defaults to empty list.
        """
        self.tiers = tier_list
        if not tier_list:
            self.tiers = []


    def to_dict(self):
        """
        Convert the entire TextGrid (including all tiers) to a dictionary.
        Useful for debugging or JSON serialization.

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
        """
        tier_dict_list = [x.to_dict() for x in self.tiers]

        out_tg_dict = {
                    'xmin': self.xmin,
                    'xmax': self.xmax,
                    'size': self.size,
                    'tiers': tier_dict_list
        }
        return out_tg_dict

    def __copy__(self):
        """Deep copy the TextGrid object."""
        return TextGrid(tier_list=[x.copy() for x in self.tiers])

    @property
    def xmin(self):
        """Global start time (min of all tiers)."""
        if len(self.tiers) == 0:
            return None
        xmin = min((x.xmin for x in self.tiers))
        return xmin
    
    @property
    def xmax(self):
        """Global end time (max of all tiers)."""
        if len(self.tiers) == 0:
            return None
        xmax = max((x.xmax for x in self.tiers))
        return xmax

    @property
    def size(self):
        """Number of tiers in the TextGrid."""
        return len(self.tiers)

    def add_tier(self, tier_item):
        """Add a new tier object to the TextGrid."""
        self.tiers.append(tier_item)      

    @staticmethod
    def parse_textgrid_ftext(ftext):
        """
        Parse a full Praat TextGrid file string into a TextGrid object.
        
        This identifies the file format, splits the text into tier blocks,
        and delegates parsing to IntervalTier or TextTier methods.

        Args:
            ftext (str): The complete file content.

        Returns:
            TextGrid: The parsed object.
        """
        # 检查文件头
        match = re.search(fileHeadRegex, ftext)
        if match is None:
            logger.error("Couldn't find head data of this file!")
            raise Exception("No File Head")

        f_xmin = float(match.group(1))
        f_xmax = float(match.group(2))
        f_size = int(match.group(3))
        f_tiers = []

        # Identify positions where each tier begins
        tier_start_pos = []
        tier_class_names = []
        pattern = re.compile(tierHeadRegex)
        pos = 0
        for tier_no in range(f_size):
            match = pattern.search(ftext, pos)
            if match is None:
                logger.error("Cannot found Tier %d!"%(tier_no + 1))
                raise Exception("No Tier")
            tier_start_pos.append(match.start(0))
            tier_class_names.append(match.group(2))
            pos = match.end(0)

        # Slice the text and parse each tier
        tier_start_pos.append(len(ftext))
        for tier_no in range(f_size):
            tier_text = ftext[tier_start_pos[tier_no]:tier_start_pos[tier_no+1]]

            if tier_class_names[tier_no] == 'IntervalTier':
                one_item = IntervalTier.parse_tier(tier_text)
            else:  # 
                one_item = TextTier.parse_tier(tier_text)
            f_tiers.append(one_item)

        tg = TextGrid(tier_list=f_tiers)
        if tg.xmin != f_xmin:
            logger.warning("TextGrid time mismatch, first xmin %f != file.xmin %f"%(tg.xmin, f_xmin))
        if tg.xmax != f_xmax:
            logger.warning("TextGrid time mismatch, last xmax %f != file.xmax %f"%(tg.xmax, f_xmax))

        return tg

    def to_textgrid_ftext(self):
        """ Convert the TextGrid object back to a string compatible with Praat files. """
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
        """
        Write the TextGrid to a file on disk.

        Args:
            filename (str): Output path.
            check (bool): If True, attempts to re-parse the generated text 
                          to ensure validity before finishing.
        """
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
        """
        Read a TextGrid file from disk.
        
        Auto-detects UTF-16-BE (Praat standard) or GBK encoding.
        
        Args:
            filename (str): Path to file.
            encoding (str, optional): Force specific encoding.

        Returns:
            TextGrid: Parsed object.
        """
        # 读取TextGrid文件，检查编码
        if encoding is None:
            try:
                ftext = TextGrid._read_text_all_lines(filename)
            except UnicodeDecodeError:
                logger.warning("File %s is not encoded with UTF-16-BE! Trying GBK."%filename)
                ftext = TextGrid._read_text_all_lines(filename, 'gbk')
        else:
            ftext = TextGrid._read_text_all_lines(filename, encoding)
        textGridDict = TextGrid.parse_textgrid_ftext(ftext)
        return textGridDict


    def down_to_table(self, tier_indexes=None, include_empty_intervals=False):
        """
        Flatten the TextGrid into a single Pandas DataFrame.
        
        Useful for analytics. The resulting DataFrame has columns:
        tmin, tmax, text, and 'tier' (name).

        Args:
            tier_indexes (list[int], optional): Indices of tiers to include. 
                                                Defaults to all.
            include_empty_intervals (bool): Whether to include empty intervals.

        Returns:
            pd.DataFrame: A combined DataFrame of all selected tiers.
        """
        if self.size == 0:
            return None

        if tier_indexes is None:
            tier_indexes = list(range(self.size))

        table_list = []
        for index in tier_indexes:
            tier_table_df, tier_name, _, _, _ = self.tiers[index].to_table_df(include_empty_intervals)
            tier_table_df['tier'] = tier_name
            table_list.append(tier_table_df)
        table_df = pd.concat(table_list).sort_values('tmin').reset_index(drop=True)

        return table_df

    @staticmethod
    def create_from_table(table_df, tier_names=None):
        """
        Create a TextGrid object from a flat DataFrame.

        Args:
            table_df (pd.DataFrame): 4 columns: tmin, tmax, tier, text.
            tier_names (list[str], optional): Order of selected tier names.

        Returns:
            TextGrid: The reconstructed object.
        """
        # Set tier_names if needed
        if tier_names is None:
            tier_names = table_df['tier'].unique()
        
        tier_list = []
        for name in tier_names:
            tier_table_df = table_df.loc[table_df['tier'] == name]

            if tier_table_df['tmax'].isna().all():
                one_item = TextTier.from_table_df(tier_table_df, name=name)
            else:
                one_item = IntervalTier.from_table_df(tier_table_df, name=name)
            tier_list.append(one_item)
        tg = TextGrid(tier_list)

        return tg

    def resize_textgrid(self, start_time=None, stop_time=None):
        """
        Resize the textgrid to a specific duration.
        """
        out_tier_list = []
        for indx, tier in enumerate(self.tiers):
            out_tier_list.append(tier.resize_tier(start_time=start_time, stop_time=stop_time, append=True, auto_extend=True))

        return TextGrid(out_tier_list)

    def concat_a_textgrid(self, tg2):
        """
        Concatenate this TextGrid with another one (tg2).
        
        The tiers of tg2 are appended to the tiers of this grid.
        Time in tg2 is shifted by this grid's duration.

        Args:
            tg2 (TextGrid): The grid to append.

        Returns:
            TextGrid: A new concatenated grid.
        """
        offset = self.xmax
        f_xmin = 0.0
        f_xmax = offset + tg2.xmax

        out_tier_list = []

        for indx, tier in enumerate(self.tiers):
            new_item = tier.concat_a_tier(tg2.tiers[indx], offset)
            out_tier_list.append(new_item)

        return TextGrid(out_tier_list)
