import random
import time
start = time.time()
MAX = 300000
hits = 0
for i in range(1, MAX):
    random.seed()
    x, y = random.random(), random.random()
    # 判断距离 运算hits
    if x*x + y*y <= 1.0:
        hits = hits + 1
pi = 4 * (hits / MAX) #为1/4圆，pi=pi*1^2对应4乘hits的占比
print("Using Monte Carlo method, the approximate pi:", pi)
print()
end = time.time()
print("finish in", end-start,"second")
