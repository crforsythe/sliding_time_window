from multiprocessing import Pool
# from pathos.multiprocessing import ProcessPool
# from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm
# import ray
#
# ray.init()



# @ray.remote(num_cpus=4)
# def g(x,y):
#     return f(x)+y

def f(x):
    return x ** 4


# if __name__=='__main__':
#
#
#     def g2(x, y):
#         return f(x) + y
#
#     n = 100000
#     x = list(range(n))
#     y = list(range(n))
#
#     args = []
#     for i in range(n):
#         args.append((x,y))
#
#     # t0 = datetime.now()
#     # a = [g.remote(i, j) for i,j in zip(x,y)]
#     # ab = ray.get(a)
#     # t1 = datetime.now()
#     # a = [g.remote(i, j) for i, j in zip(x, y)]
#     # ab = ray.get(a)
#     t2 = datetime.now()
#     b = []
#     for i, j in tqdm(zip(x, y)):
#         b.append(g2(i, j))
#     t3 = datetime.now()
#     #
#     # print(t1-t0)
#     # print(t2 - t1)
#
#     with Pool(4) as pool:
#         c = pool.starmap(g2, args)
#
#
#     a = Parallel(prefer='threads')(delayed(g2)(i,j) for i,j in zip(x,y))
#     t4 = datetime.now()
#
#     print(t3 - t2)
#     print(t4 - t3)


