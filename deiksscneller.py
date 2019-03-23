import heapq
inf = float('INf')
n, s, f = map(int, input().split())
s -= 1
f -= 1

matrix = [list(map(int, input().split())) for i in range(n)]
dist = [inf for i in range(n)]
dist[s] = 0
q = list(zip(dist[:n],range(n)))
heapq.heapify(q)
while q:
    index_to_use = heapq.heappop(q)
    index_to_use = index_to_use[1]
    for j in range(n):
        if  matrix[index_to_use][j] != -1:
            if dist[index_to_use] + matrix[index_to_use][j] < dist[j]:
                dist[j] = dist[index_to_use] + matrix[index_to_use][j]
                heapq.heappush(q,(dist[j],j))
if dist[f] != inf:
    print(dist[f])
else:
    print(-1)
