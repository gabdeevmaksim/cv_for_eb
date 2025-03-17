import sys
import os,warnings

from astroquery.vizier import Vizier

from astropy.coordinates.sky_coordinate import SkyCoord, Angle
from astropy.table import Table, Column, vstack
from astropy import units as u
from astropy.io import ascii

import numpy as np
import numpy.ma as ma


GAIA_Objects='I/358/veb'
GAIA_LC='I/355/epphot'
GAIA_DR3='I/355/gaiadr3'


gaia_objects=ascii.read('gaia_eb.dat')

new_table=Table()
row_count=0

for star in gaia_objects[:]:
    source=star['Source']
    RA_ICRS=star['RA_ICRS']
    DE_ICRS=star['DE_ICRS']
    TimeRef=star['TimeRef']
    Period=1/star['Freq']
    print(row_count, source)
    v=Vizier(catalog=GAIA_DR3, columns=['RA_ICRS','DE_ICRS', 'GLON', 'GLAT', 'Plx','e_Plx', 'PM', 'pmRA','e_pmRA','pmDE','e_pmDE', 'Teff', 'Gmag', 'BPmag', 'RPmag', 'BP-RP', 'RV', 'e_RV', 'RVamp', 'logg', 'Dist'])
                          #
    try:
        r = v.query_constraints(Source=source)[0]
    # print(r)
        tab=Table()
        tab['Source']=Column(dtype='U20')
        tab['RA_ICRS']=[]
        tab['DE_ICRS']=[]
        tab['TimeRef']=[]
        tab['Period']=[]
        new_row = {'Source': str(source), 'RA_ICRS': RA_ICRS, 'DE_ICRS':DE_ICRS, 'TimeRef':TimeRef, 'Period':Period}
        tab.add_row(new_row)

        for col_name in r.colnames:
        # print(col_name)
            original_column = r[col_name]
            if isinstance(original_column, ma.MaskedArray): #check if column is masked
                tab[col_name] = ma.masked_array(original_column.data, mask=original_column.mask, dtype=original_column.dtype) #copy masked array
            else:
                tab[col_name] = original_column.copy() #copy regular column
    # print(tab)
        new_table=vstack([new_table, tab])
        # print(new_table)
        row_count +=1
        if row_count % 1000 == 0:
            print('Saving catalog...')
            new_table.write('catalog_eb_gaia.csv', format='csv', overwrite=True)  # Replace 'fits' with your desired format (e.g., 'csv', 'ascii')

    except:
        pass
