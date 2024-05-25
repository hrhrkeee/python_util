import io,sys,os
os.system("Cls")
with open("input.txt") as TxtOpen:
    INPUT=TxtOpen.read() 
sys.stdin=io.StringIO(INPUT)
# --------------------------------------------------------

N, M = list(map(int, input().split()))
Am = list(map(int, input().split()))

count = 0
for i in range(1, N+1, 1):
    print(Am[count] - i)
    if Am[count] == i:
        count += 1