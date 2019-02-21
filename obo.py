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

#упр3 стр 112 и упр 8,9,10