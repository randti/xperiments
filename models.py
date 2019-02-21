a=[0]*100000
def get_hash(value):
    value = str(value)
    return sum([ord(value[i]) ** (i + 1) for i in range(0, len(value))]) % 100000
def adds(i,n):
    a[get_hash(i)]=n
for p in range(5):
    c,b=input().split()
    adds(c,b)

