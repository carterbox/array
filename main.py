import time
import threading
import os
import warnings

from numpy.random import default_rng
import matplotlib.pyplot as plt
import numba
import numexpr as ne
import numpy as np
import cupy as cp
import cupyx

import custom

rng = default_rng()


def direction_dy_cupy(d, grad1, grad0, num_streams=1):

    num_chunks = (np.prod(d.shape) * 4) // (400 * 1024 * 1024) + 1
    num_chunks = max(num_streams, num_chunks)
    print(f'cupy streams using {num_chunks} chunks with {num_streams} streams')

    chunks = array_split_index(len(d), num_chunks)
    result = cupyx.empty_like_pinned(d)
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    a = cp.empty_like(d, shape=(num_chunks, ))
    b = cp.empty_like(d, shape=(num_chunks, ))

    for k in range(num_chunks):
        with streams[k % num_streams] as stream:
            gpu_d = cp.empty_like(d[chunks[k]:chunks[k + 1]])
            gpu_d.set(d[chunks[k]:chunks[k + 1]])
            gpu_grad1 = cp.empty_like(grad1[chunks[k]:chunks[k + 1]])
            gpu_grad1.set(grad1[chunks[k]:chunks[k + 1]])
            gpu_grad0 = cp.empty_like(grad0[chunks[k]:chunks[k + 1]])
            gpu_grad0.set(grad0[chunks[k]:chunks[k + 1]])

            a[k] = np.sum(np.square(gpu_grad1))
            b[k] = np.sum(gpu_d * (gpu_grad1 - gpu_grad0))

    [stream.synchronize() for stream in streams]

    a_over_b = np.sum(a) / (np.sum(b) + 1e-32)

    for k in range(num_chunks):
        with streams[k % num_streams] as stream:
            gpu_d = cp.empty_like(d[chunks[k]:chunks[k + 1]])
            gpu_d.set(d[chunks[k]:chunks[k + 1]])
            gpu_grad1 = cp.empty_like(grad1[chunks[k]:chunks[k + 1]])
            gpu_grad1.set(grad1[chunks[k]:chunks[k + 1]])

            gpu_result = -gpu_grad1 + gpu_d * a_over_b

            gpu_result.get(out=result[chunks[k]:chunks[k + 1]])

    [stream.synchronize() for stream in streams]

    return result


@numba.njit(parallel=True)
def direction_dy_numba(d, grad1, grad0):
    return (-grad1 + d * np.sum(np.square(grad1)) /
            (np.sum(d * (grad1 - grad0)) + 1e-32))

def direction_dy_numpy(d, grad1, grad0):
    return (-grad1 + d * np.sum(np.square(grad1)) /
            (np.sum(d * (grad1 - grad0)) + 1e-32))


# NOTE: Not valid numexpr because reduce isn't last
direction_dy_expression = '-a + m * sum(a * a) / (1e-32 + sum(m * (a - x)))'


@numba.njit(parallel=True)
def copy_numba(x):
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
            args=(b, a, chunks[k], chunks[k + 1]),
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
            args=(b, m, a, x, chunks[k], chunks[k + 1]),
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

    # if int(os.environ['NUMBA_NUM_THREADS']) != int(os.environ['NUMEXPR_MAX_THREADS']):
    #     warnings.warn('Number of NUMBA and NUMEXPR threads is not the same!')
    NUM_THREADS = int(os.environ['NUMBA_NUM_THREADS'])

    ntrials = 5
    noptions = 3
    dtype = np.single

    for L in [256, 512, 1024, 2048]:

        m = cupyx.empty_pinned(shape=(L * L * L), dtype=dtype)
        a = cupyx.empty_pinned(shape=(L * L * L), dtype=dtype)
        x = cupyx.empty_pinned(shape=(L * L * L), dtype=dtype)
        rng.random(size=L * L * L, dtype=dtype, out=m)
        rng.random(size=L * L * L, dtype=dtype, out=a)
        rng.random(size=L * L * L, dtype=dtype, out=x)
        print(L, a.dtype)

        times = list()
        b = direction_dy_numba(m, a, x)
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = direction_dy_numba(m, a, x)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(f'numba takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s0 = plt.scatter([len(a) * 1.00] * ntrials,
                         times,
                         marker='>',
                         color=plt.cm.cividis(0 / noptions))

        times = list()
        b = direction_dy_cupy(m, a, x)
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = direction_dy_cupy(m, a, x)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(f'cupy takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s1 = plt.scatter([len(a) * 1.01] * ntrials,
                         times,
                         marker='o',
                         color=plt.cm.cividis(1 / noptions))

        times = list()
        b = direction_dy_cupy(m, a, x)
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = direction_dy_cupy(m, a, x, num_streams=8)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(f'cupy takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s2 = plt.scatter([len(a) * 1.02] * ntrials,
                         times,
                         marker='x',
                         color=plt.cm.cividis(2 / noptions))

        times = list()
        b = direction_dy_numpy(m, a, x)
        for _ in range(ntrials):
            t0 = time.perf_counter()
            b = direction_dy_numpy(m, a, x)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        assert not np.shares_memory(a, b)
        if __debug__:
            print(a, b)
        # assert np.array_equal(a, b)
        print(f'numpy takes {np.mean(times):.3e} +/- {np.std(times):.3e}')
        s3 = plt.scatter([len(a) * 1.00] * ntrials,
                         times,
                         marker='s',
                         color=plt.cm.cividis(3 / noptions))


    plt.loglog()
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Number of Elements Updated (daiyuan search direction)')
    plt.ylabel('Time to Update [s]')
    plt.title(f'Time to Update a {a.dtype} NumPy Array')
    plt.legend(
        handles=[s0, s1, s2, s3],
        labels=[f'numba {NUM_THREADS}-threads', 'cupy 1-stream', 'cupy 8-streams', 'numpy'],
    )
    plt.savefig('update.png', dpi=600)
