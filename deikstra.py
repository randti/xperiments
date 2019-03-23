inf = float('INf')
n, s, f = map(int, input().split())
s -= 1
f -= 1
d = [-1 for i in range(n)]
used = [False for i in range(n)]
matrix = [list(map(int, input().split())) for i in range(n)]
dist = [inf for i in range(n)]
dist[s] = 0
debug = []
for i in range(n):
    index_to_use = -1
    for j in range(n):
        if not used[j] and (index_to_use == -1 or dist[j] < dist[index_to_use]):
            index_to_use = j
    for j in range(n):
        if j != index_to_use and matrix[index_to_use][j] != -1:
            if dist[index_to_use] + matrix[index_to_use][j] < dist[j]:
                d[j] = index_to_use
            dist[j] = (min(dist[j], dist[index_to_use] + matrix[index_to_use][j]))
    used[index_to_use] = True
if dist[f] != inf:
    path = [f + 1]
    x = d[f]
    while x != -1:
        path.append(x + 1)
        x = d[x]
    # print(*reversed(path))
    print(dist)
else:
    print(-1)