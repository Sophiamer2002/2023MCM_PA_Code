import numpy as np
import pygrib as pg
import pandas as pd
import os

class gribparser():
    def __init__(self, filename: str):
        '''
        to deal with a continuous month-average weather data.
        each month has four average data, indicating the average
        weather info at 00:00, 06:00, 12:00 and 18:00. 
        --------------------------------
        dataset1
            time: 1950-01-01 00:00:00 -- 2022-12-01 00:00:00
            content1: Evaporation from vegetation transpiration
            content2: Skin temperature
            content3: Total precipitation
        --------------------------------
        dataset2
            time: 1950-01-01 00:00:00 -- 2022-12-01 00:00:00
            content1: Leaf area index, high vegetation
            content2: Leaf area index, low vegetation
            content3: Skin temperature
            content4: Evaporation
            content5: Total precipitation
        --------------------------------
        dataset3
            time: 1950-01-01 00:00:00 -- 2022-12-01 00:00:00
            content1: Skin temperature
            content2: Volumetric soil water layer 1
            content3: Volumetric soil water layer 2
            content4: Volumetric soil water layer 3
            content5: Volumetric soil water layer 4
        --------------------------------
        '''        
        # judge the legality of the file
        if not os.path.isfile(filename):
            raise FileNotFoundError("File not found!")
        if not filename.endswith(".grib"):
            raise TypeError("File type error!")
        
        # open the file
        self.grbs = pg.open(filename)
        self.lat, self.lon = self.__latloninfo()
        self.content = self.__contentinfo()
        self.timeinfo = self.__timeinfo()

    def __latloninfo(self):
        '''
        return the longitude and latitude information of the grib file
        '''
        lat, lon = self.grbs[1].latlons()
        return lat[:, 0], lon[0, :]
    
    def __contentinfo(self):
        '''
        return the content information of the grib file
        '''
        content = []
        for grb in self.grbs:
            name = grb.name
            if name not in content:
                content.append(name)
            else:
                break
        return content


    def __timeinfo(self) -> pd.DatetimeIndex:
        '''
        return the time of the file, using pandas.DatetimeIndex
        suppose that the time is every month
        '''
        total = self.grbs.messages
        start = self.grbs[1].validDate.replace(day=1)
        end = self.grbs[total].validDate.replace(day=1)
        return pd.date_range(start, end, freq='MS')


    def __focuslatlonidx(self, lat: float, lon: float):
        '''
        given the longitude and latitude, return the index of the nearest point
        '''
        lonidx = np.argmin(np.abs(self.lon - lon))
        latidx = np.argmin(np.abs(self.lat - lat))
        return latidx, lonidx


    def getcontentname(self):
        '''
        return the content name of the grib file
        '''
        return self.content
    
    def getcontent(self, name: str, lat: float, lon: float):
        '''
        given the column name, longitude and latitude,
        return the content of the grib file, using panda.Series
        '''
        latidx, lonidx = self.__focuslatlonidx(lat, lon)
        content = []
        count = 0
        timespan = len(self.timeinfo)
        for grb in self.grbs.select(name=name):
            content.append(grb.values[latidx, lonidx])
            count += 1
        ret_val = pd.DataFrame([self.timeinfo.repeat(count/timespan), content]).T.rename(columns={0: "time", 1: name})
        return ret_val
