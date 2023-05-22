import multiprocessing as mp
# from https://stackoverflow.com/questions/9601802/python-pool-apply-async-and-map-async-do-not-block-on-full-queue?rq=1
from threading import Semaphore
from multiprocessing import Pool
import tqdm


class TaskManager(object):
    def __init__(self, processes, queue_size, total, callback=None, pbar=None, pbar_desc="", use_pbar=False):
        self.use_pbar = use_pbar
        if use_pbar:
            self.pbar = tqdm.tqdm(total=total,desc=pbar_desc) if pbar is None else pbar
        self.pool = Pool(processes=processes)
        self.workers = Semaphore(processes + queue_size)
        self.callback = callback
        self.outputs = []
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self,type, value, traceback):
        self.pool.close()
        self.pool.join()
        self.outputs = [(i, r.get()) for i, r in self.outputs]
        if self.use_pbar:
            self.pbar.close()

    def new_task(self, function, *args, **kwargs):
        """Start a new task, blocks if queue is full."""
        self.workers.acquire()
        res = self.pool.apply_async(function, args, kwargs, callback=self.task_done, error_callback=self.release_and_print)
        self.outputs += [(self.index, res)]
        self.index += 1

    def task_done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        self.workers.release()
        if self.callback is not None:
            self.callback(*args, **kwargs)
        if self.use_pbar:
            self.pbar.update(1)

    def release_and_print(self, e):
        self.workers.release()
        print(e)