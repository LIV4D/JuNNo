from multiprocessing import Queue, Process
from multiprocessing.pool import ThreadPool
import threading
import traceback
import os
import math
from .j_log import log

import gc
import time

N_CORE = (os.cpu_count() * 3 // 4)


def parallel_exec(f, seq, static_args=None, n_process=None, cb=None):
    if static_args is None:
        static_args = {}
    if n_process is None:
        n_process = min(N_CORE, len(seq))

    pool = []
    queue = Queue()
    length = len(seq)
    l = math.floor(length/n_process)
    lock = threading.Lock()

    def process_f(f, seq, seq_id, kwargs, q):
        for i in seq_id:
            with lock:
                try:
                    r = f(seq[i], **kwargs)
                except Exception as e:
                    print('Parralellism ERROR')
                    traceback.print_exc()
                    q.put((i, None))
                    continue
                q.put((i, r))

    start_id = 0
    for p_id in range(n_process):
        end_id = start_id + l + (1 if len(seq)-l*n_process-p_id>0 else 0)
        seq_id = list(range(start_id, end_id))
        kwargs = dict(f=f, seq=seq, seq_id=seq_id, kwargs=static_args, q=queue)
        p = Process(target=process_f, kwargs=kwargs)
        p.start()
        pool.append(p)
        start_id = end_id

    if cb is None:
        r = [None] * len(seq)
        for i in range(len(seq)):
            seq_i, result = queue.get(block=True)
            r[seq_i] = result
    else:
        for i in range(len(seq)):
            id, r = queue.get(block=True)
            if r is not None:
                cb(r)

    for p in pool:
        p.terminate()
        p.join()
    queue.empty()
    queue.close()
    gc.collect()

    if cb is None:
        return r


def intime_generator(gen):
    thread = ThreadPool(processes=1)

    def read_gen():
        return next(gen)

    thread_result = thread.apply_async(read_gen)

    while 'StopIteration is not raised':
        r = thread_result.get()
        thread_result = thread.apply_async(read_gen)
        yield r
