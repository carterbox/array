import time
import threading
import os
import warnings

from numpy.random import default_rng
import matplotlib.pyplot as plt
import numba
import numexpr as ne
import numpy as np

import custom

rng = default_rng()


@numba.njit(parallel=True)
def cp(x):
    y = np.empty_like(x)
    n = len(x)
    for i in numba.prange(n):
        y[i] = x[i]
    return y

@numba.njit(parallel=True)
def update(m, a, x):
    y = np.empty_like(x)
    n = len(x)
    for i in numba.prange(n):
        y[i] = m[i] + a[i] * x[i]
    return y

def fill_array(res, res0, st, end):
    res[st:end] = res0[st:end]


def update_array(b, m, a, x, st, end):
    b[st:end] = m[st:end] + a[st:end] * x[st:end]


def array_split_index(length, num_chunks):
    if num_chunks > length:
        raise ValueError()
    chunks = [length // num_chunks] * num_chunks
    for i in range(length % num_chunks):
        chunks[i] = chunks[i] + 1
    chunks = [0] + chunks
    return np.cumsum(chunks)

def thread_copy(a, num_threads):
    b = np.empty_like(a)
    mthreads = []
    chunks = array_split_index(len(a), num_threads)
    for k in range(num_threads):
        th = threading.Thread(
            target=fill_array,
            args=(b, a, chunks[k], chunks[k+1]),
        )
        th.start()
        mthreads.append(th)
    for k in range(num_threads):
        mthreads[k].join()
    return b

def thread_update(m, a, x, num_threads):
    b = np.empty_like(a)
    mthreads = []
    chunks = array_split_index(len(a), num_threads)
    for k in range(num_threads):
        th = threading.Thread(
            target=update_array,
            args=(b, m, a, x, chunks[k], chunks[k+1]),
        )
        th.start()
        mthreads.append(th)
    for k in range(num_threads):
        mthreads[k].join()
    return b

# NOTE: Let's not use multiprocess because each process has its own memory so
# there is additional copy overhead.

if __name__ == '__main__':
    plt.figure()

    if int(os.environ['NUMBA_NUM_THREADS']) != int(os.environ['NUMEXPR_MAX_THREADS']):
        warnings.warn('Number of NUMBA and NUMEXPR threads is not the same!')
    NUM_THREADS = int(os.environ['NUMBA_NUM_THREADS'])

    ntrials = 5
    noptions = 6 - 1

    for L in [2048]:

        m = rng.random(size=L * L * L, dtype=np.double)
        a = rng.random(size=L * L * L, dtype=np.double)
        x = rng.random(size=L * L * L, dtype=np.double)
        print(L, a.dtype)

        # times = list()
        # b = a.copy()
        # for _ in range(ntrials):
        #     t0 = time.perf_counter()
        #     b = np.empty_like(a)
        #     memoryview(b)[:] = memoryview(a)
        #     t1 = time.perf_counter()
        #     times.append(t1 - t0)
        # assert not np.shares_memory(a, b)
        # print(a, b)
        # assert np.array_equal(a, b)
        # print(f'memoryview takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        # s1 = plt.scatter([len(a)] * ntrials,
        #                  times,
        #                  marker='x',
        #                  color=plt.cm.cividis(0/noptions))

        times = list()
        b = ne.evaluate('a')
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = ne.evaluate('m + a * x')
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(f'numexpr takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s2 = plt.scatter([len(a) * 1.01] * ntrials,
                         times,
                         marker='o',
                         color=plt.cm.cividis(1/noptions))

        times = list()
        b = cp(a)
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = update(m, a, x)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(f'numba takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s3 = plt.scatter([len(a) * 1.02] * ntrials,
                         times,
                         marker='>',
                         color=plt.cm.cividis(2/noptions))

        times = list()
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = thread_update(m, a, x, num_threads=NUM_THREADS)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(
            f'threads takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s0 = plt.scatter([len(a) * 1.03] * ntrials,
                         times,
                         marker='+',
                         color=plt.cm.cividis(3/noptions))

        times = list()
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = np.empty_like(a)
            custom.updated(m, a, x, b)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(
            f'nanobind takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s4 = plt.scatter([len(a) * 1.04] * ntrials,
                         times,
                         marker='s',
                         color=plt.cm.cividis(4/noptions))

    plt.loglog()
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Number of Elements Updated (b = m + a * x)')
    plt.ylabel('Time to Update [s]')
    plt.title(f'Time to Update a {a.dtype} NumPy Array')
    plt.legend(
        handles=[s2, s3, s0, s4],
        labels=[
            'numexpr.eval()', 'numba.njit()', f'{NUM_THREADS} Thread', 'nanobind',
        ],
    )
    plt.savefig('update.png', dpi=600)


def direction_dy(d, grad1, grad0):
    return (
        - grad1
        + d * np.sum(np.square(grad1))
        / (np.sum(d * (grad1 - grad0)) + 1e-32)
    )
