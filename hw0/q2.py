# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:21:38 2016

@author: gary830317
"""

from PIL import Image
import sys
img = Image.open(sys.argv[1])
img2 = img.rotate(180)
img2.save("ans2.png")