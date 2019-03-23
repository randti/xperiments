INF = float('inf')
n,m = (int(i) for i in input().split())
a=[]
dist = [INF]*n
dist[0]=0
for i in range(m):
    a.append(list(map(int,input().split())))
for i in range(n - 1):
    for j in range(m):
        dist[a[j][1] - 1] = min(dist[a[j][1] - 1], dist[a[j][0]-1]+a[j][2])
dist=[dist[i] if dist[i]!=INF else 30000 for i in range(n)]
print(*dist)