import _heapq
n=int(input())
h=list(map(int,input().split()))
_heapq._heapify_max(h)
for i in range(n-1):
  c=h.copy()
  n=_heapq.heappop(h)
  _heapq._heapify_max(h)
  print(c.index(h[0])+1,n)
