#!/usr/bin/env python
from __future__ import division,print_function

xs = [1,5,6,0,4,7,None,8,9]


print('demo 1: catching any error, and proceeding beyond:')
for x in xs:
    try:
        print('1/x = {}, 1+x = {}'.format(1/x, 1+x))
    except:
        print('error with {}'.format(x))


print('demo 1: catching only ZeroDivisionError:')
for x in xs:
    try:
        print(1/x, 1+x)
    except ZeroDivisionError:
        print('trying to divide by zero, silly! Can\'t do that.')


