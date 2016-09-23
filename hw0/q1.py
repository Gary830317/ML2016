# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import sys

col = int(sys.argv[1])
filename = str(sys.argv[2])

content = np.loadtxt(filename)

a = np.sort( content[:, col], axis=None) 
f = open('ans1.txt', 'w')
f.write(','.join('%f' % i for i in a.tolist()))
