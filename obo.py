"""





def bfs(input_g: dict, start_vertex):
    n = len(input_g)
    dist = [n] * n
    dist[start_vertex] = 0
    queue = [start_vertex]
    while queue:
        current_vertex = queue.pop(0)
        for i in input_g[current_vertex]:
            if dist[i] > dist[current_vertex] + 1:
                dist[i] = dist[current_vertex] + 1
                print(i,input_g[current_vertex],queue,dist)
                queue.append(i)
    return dist
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    paths=[]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set(graph[vertex]) - set(path):
            if next == goal:
               paths.append(path + [next])
            else:
                queue.append((next, path + [next]))

    return paths


graph = {0:[1,3],
         1:[2,5],
         2:[3,7],
         3:[4],
         4:[5],
         5:[6],
         6:[7],
         7:[7]
         }


graph1={0:[1],
        1:[2],
        3:[4]}
v=[]
n=(max(max(graph1.keys()),max(max(graph1.values()))))
def dfs(graph1:dict,start):
    v.append(start)
    if start in graph1.keys():

        for i in graph1[start]:
            if not i in v:

                dfs(graph1,i)
    return v
print(dfs(graph1,0))

print(bfs(graph,0))
#упр3 стр 112 и упр 8,9,10
"""
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
    print(*reversed(path))
    print(dist[f])
else:
    print(-1)









import heapq
inf = float('INf')
n,m, s = map(int, input().split())

f =n-1
d = [-1 for i in range(n)]
matrix = [[-1] * n for i in range(n)]
for it in range(m):
    b, k,c = map(int, input().split())
    matrix[b - 1][k - 1] =c
    matrix[k - 1][b - 1] =c

dist = [[inf,i] for i in range(n)]
dist[s][0] = 0
q=dist.copy()
heapq.heapify(q)
while q:
    index_to_use = heapq.heappop(q)
    index_to_use=index_to_use[1]
    for j in range(n):
        if j != index_to_use and matrix[index_to_use][j] != -1:
            if dist[index_to_use][0] + matrix[index_to_use][j] < dist[j][0]:
                d[j] = index_to_use
            dist[j][0] = (min(dist[j][0], dist[index_to_use][0] + matrix[index_to_use][j]))
if dist[f] != inf:
    print(dist)
else:
    print(-1)