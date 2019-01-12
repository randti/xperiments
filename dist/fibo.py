n=int(input())
players=[]
for i in range(n):
    a=int(input())
    for j in range(a):
        f,b=map(str,input().split())
        players.append(tuple((f,b)))
players=sorted(players, key=lambda x: float(x[0]))[::-1]
print(len(players))
for i in players:
    print(*i)

