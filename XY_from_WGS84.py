#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyproj
import math

lon = 141.091886244067


lat = 38.7206632931




WGS84 = pyproj.Proj('+init=EPSG:4326')
UTM54S = pyproj.Proj('+init=EPSG:32654')

x,y = pyproj.transform(WGS84, UTM54S, lon, lat)

print "x = %s" % x
print "y = %s" % y
