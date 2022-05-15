def watertrapped(lst):
 
    l = 0               # left pointer
    r = len(lst) - 1   # right pointer
    res = 0
 
    maxl = lst[l]               # max left 
    maxr = lst[r]               # max right
 
    while l < r:
        if lst[l] <= lst[r]:
            l = l + 1
            maxl = max(maxl, lst[l])
            res =  res + maxl - lst[l]
        else:
            r = r - 1
            maxr = max(maxr, lst[r])
            res = res + maxr - lst[r]
 
    return res
    
a = [3,0,0,2,0,4]

print(watertrapped(a))