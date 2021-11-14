def bisect(lst, value, key=itemgetter(0), sign=1, compare=lambda a, b: a < b):
    """reimplemented bisect (ref: https://docs.python.org/2/library/bisect.html) with
    
    1. specify which entry of the list item to sort
    
    2. allow for negative sort (see last comment, needed for list with Y
       strict increasing and X strict decreasing) 
    
    return the idx of the target element 

	>>> from _operator import itemgetter
    >>> L = [[0,5,3],[1,3,13],[2,2,21],[4,1,12],[5,0,1]]
    >>> print(bisect(L, 5, key=itemgetter(0)))           # bisect_left over the first entry of each list element 
    4

    """
    lo, hi = 0, len(lst)
    while compare(lo, hi):                   # bisect_left 
        mid = (lo + hi) // 2
        # if -key(lst[mid]) < value: # to do bisect.bisect_left([-x[0] for x in L], -p[0])
        if sign * key(lst[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


