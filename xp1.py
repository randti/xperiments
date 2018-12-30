a=[]
for i in range(10):
    for j in range(10):
        for p in range(10):
            a.append(tuple(((i,j,p))))
print(*a)