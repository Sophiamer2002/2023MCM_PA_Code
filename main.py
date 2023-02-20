from utils.gribparser import gribparser
from simulator.simulator import simulator
import pandas as pd

def getGDDandTw(lat, lon, filename):
    '''
    given the longitude and latitude, return the GDD of the grib file, using panda.Series
    '''
    grib = gribparser(filename)
    df = grib.getcontent('Skin temperature', lat=lat, lon=lon)

    month = df.time.dt.month
    year = df.time.dt.year

    # convert Kelvin to Celsius, base temperature is 5.5
    GDD_E = (df['Skin temperature'] - 5.5 - 273.15).groupby(year).sum()/4 * 30.5

    GDD_D = (df[(month>=4) & (month<=10)]['Skin temperature'] - 5.5 - 273.15).groupby(year).sum()/4 * 30.5

    Tw = (df[(month<=2) | (month==12)]['Skin temperature'] - 5.5 - 273.15).groupby(year).sum()/4

    return pd.DataFrame({'GDD_E': GDD_E, 'GDD_D': GDD_D, 'Tw': Tw})

def main():
    
