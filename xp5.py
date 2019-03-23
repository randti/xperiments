import heapq
n=int(input())
a=list(map(int,input().split()))
b=int(input())
d=[]
for i in range(b):
    m,k=map(int,input().split())
    g=a[m-1:k]
    g=[[g[i],i+m] for i in range(len(g))]
    heapq._heapify_max(g)
    d.append(heapq.heappop(g))
for i in d:
    print(*i)