"""import _heapq
n=int(input())
h=list(map(int,input().split()))
_heapq._heapify_max(h)
for i in range(n-1):
  c=h.copy()
  n=_heapq.heappop(h)
  _heapq._heapify_max(h)
  print(c.index(h[0])+1,n)
"""
n=int(input())
a=[]
while n!=3:
  n=n-2
  a.append(2)
  if n==0:
    break
if n==3:
  a.append(3)
print(len(a))
print(*a)
