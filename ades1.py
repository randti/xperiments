from math import sqrt,ceil,sqrt
def rouns(a):
    if a-int(a)>=0.5:
        return ceil(a)
    else:
        return int(a)
def builds(a,b):
    c=[]
    while b!=0:
        c.append(rouns(a/b))
        a-=rouns(a/b)
        b-=1
    return c
def mins(a,k):
    import heapq
    n=len(a)
    h = list(zip(a[:k], range(k)))
    heapq.heapify(h)
    p=h[0][0]-(k-1)
    for i in range(k, n):
        heapq.heappush(h, (a[i], i));
        while h[0][1] <= i - k:
            heapq.heappop(h)
        p+=h[0][0]-(k-1)
    return p
n=int(input())
if n>=4:
  h=int(sqrt(n))
  a=(builds(n,h))
  a=sorted(a)[::-1]
  d=0
  for i in range(2,len(a)+1):
      d+=mins(a,i)
  print(d)
else:
  print(0)




