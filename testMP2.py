from multiprocessing import Pool
import multiprocessing as mp
from testMP import f

if __name__ == '__main__':
    mp.set_start_method('fork')
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))