import argparse
import logging
import queue
import threading
import weakref

from typing import Optional, Sequence, Sized

import torch
from torch.distributed import rpc
from torch.distributed.rpc.api import RRef
from torch.utils.data import BatchSampler, default_collate, IterableDataset, RandomSampler, Sampler, SequentialSampler


logger = logging.getLogger(__name__)


def _rref_len(a: RRef):
    return len(a.local_value())


def _rref_isinstance(a: RRef, t):
    return isinstance(a.local_value(), t)


class _SizedPlaceholder(Sized):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


class ItemFetcher:
    def __init__(self, dataset_rref) -> None:
        self.dataset = dataset_rref.local_value()

    def fetch(self, i, item):
        return self.dataset[item]


class BatchItemFetcher:
    def __init__(self, dataset_rref, collate_fn=None) -> None:
        self.dataset = dataset_rref.local_value()
        self.collate_fn = collate_fn

    def fetch(self, i, items):
        batch = [self.dataset[i] for i in items]
        return self.collate_fn(batch) if self.collate_fn else batch


class IterFetcher:
    def __init__(self, dataset_rref) -> None:
        self.dataset = iter(dataset_rref.local_value())
        self.condition = threading.Condition()
        self.i = 0

    def fetch(self, i):
        with self.condition:
            while self.i != i:
                self.condition.wait()

            try:
                return next(self.dataset)
            except StopIteration as e:
                return e
            finally:
                self.i += 1
                self.condition.notify_all()


class BatchIterFetcher:
    def __init__(self, dataset_rref, batch_size, collate_fn=None) -> None:
        self.dataset = iter(dataset_rref.local_value())
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.condition = threading.Condition()
        self.i = 0

    def fetch(self, i):
        with self.condition:
            if self.i != i:
                self.condition.wait_for(lambda: self.i == i)
            self.i += 1
            self.condition.notify_all()

            batch = []
            for _ in range(self.batch_size):
                try:
                    batch.append(next(self.dataset))
                except StopIteration as e:
                    if len(batch) == 0:
                        return e

            return self.collate_fn(batch) if self.collate_fn else batch


class RPCDispatcher:
    def __init__(self, datasets, iterable, batch_size, collate_fn=None):
        if iterable and batch_size is None:
            fetchers = [rpc.remote(d.owner(), IterFetcher, (d,)) for d in datasets]
        elif iterable:
            fetchers = [rpc.remote(d.owner(), BatchIterFetcher, (d, batch_size, collate_fn)) for d in datasets]
        elif batch_size is None:
            fetchers = [rpc.remote(d.owner(), ItemFetcher, (d,)) for d in datasets]
        else:
            fetchers = [rpc.remote(d.owner(), BatchItemFetcher, (d, collate_fn)) for d in datasets]

        self.fetchers = [d.rpc_async() for d in fetchers]  # remote handles
        self.i = [0] * len(fetchers)  # number of submitted jobs
        self.exhausted = [False] * len(fetchers)  # StopIteration was raise
        self.rr = 0  # round-robin index
        self.queue = queue.Queue()  # submitted jobs

        weakref.finalize(self, self._finalize, self.queue)

    @staticmethod
    def _finalize(q):
        # purge remaining jobs because non-retrieved futures generate error logs
        while True:
            try:
                _, fut = q.get_nowait()
            except queue.Empty:
                break

            try:
                fut.wait()
            except:  # noqa: E722
                pass

    def put(self, *kargs, **kwargs):
        if all(self.exhausted):
            logger.warn("ignoring job since all iterators are exhausted")
            return

        while True:
            fetcher = self.fetchers[self.rr]
            if fetcher is None:
                self.rr = (self.rr + 1) % len(self.fetchers)
            else:
                self.queue.put((self.rr, fetcher.fetch(self.i[self.rr], *kargs, **kwargs)))
                self.i[self.rr] += 1
                self.rr = (self.rr + 1) % len(self.fetchers)
                break

    def __next__(self):
        while True:
            try:
                rr, f = self.queue.get_nowait()
            except queue.Empty:
                raise StopIteration()

            value = f.wait()

            if isinstance(value, StopIteration):
                self.fetchers[rr] = None
            else:
                return value


class RPCDataloader:
    def __init__(
        self,
        datasets,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        generator: Optional[torch.Generator] = None,
        *,
        prefetch_factor: int = 2,
    ):
        iterable = rpc.rpc_sync(datasets[0].owner(), _rref_isinstance, (datasets[0], IterableDataset))

        if not iterable and sampler is None:
            size = rpc.rpc_sync(datasets[0].owner(), _rref_len, (datasets[0],))
            placeholder = _SizedPlaceholder(size)
            if shuffle:
                sampler = RandomSampler(placeholder, generator=generator)
            else:
                sampler = SequentialSampler(placeholder)

        if not iterable and batch_size is not None and batch_sampler is None:
            assert sampler is not None
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.datasets = datasets
        self.iterable = iterable
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.batch_size = batch_size
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

    def __len__(self):
        if self.iterable:
            raise TypeError()
        else:
            return rpc.rpc_sync(self.datasets[0].owner(), _rref_len, args=(self.datasets[0],))

    def __iter__(self):
        if self.iterable:
            indices_it = None
        elif self.batch_size is None:
            indices_it = iter(self.sampler)
        else:
            indices_it = iter(self.batch_sampler)

        dispatcher = RPCDispatcher(self.datasets, self.iterable, self.batch_size, self.collate_fn)

        # preload jobs
        for _ in range(len(self.datasets) * self.prefetch_factor):
            try:
                indices = tuple() if indices_it is None else [next(indices_it)]
            except StopIteration:
                break
            else:
                dispatcher.put(*indices)

        while True:
            # retrieve data
            try:
                yield next(dispatcher)
            except StopIteration:
                break

            # queue another job
            try:
                indices = tuple() if indices_it is None else [next(indices_it)]
            except StopIteration:
                continue
            else:
                dispatcher.put(*indices)


def run_rpc_worker(*args, **kwargs):
    if __name__ == "__main__":
        argparser = argparse.ArgumentParser("run RPC Dataloader worker")
        argparser.add_argument("--name", required=True, help="RPC node name")
        argparser.add_argument("--backend", help="RPC node name")
        argparser.add_argument("--rank", type=int, help="RPC node rank")
        argparser.add_argument("--world-size", type=int, help="RPC world size")
        argparser.add_argument(
            "--init-method",
            default="env://",
            help="URL specifying how to initialize the process group",
        )
        argparser.add_argument("--timeout", default=60.0, type=float, help="RPC timeout")

        args = argparser.parse_args()

        kwargs = {
            "name": args.name,
            "backend": None if args.backend is None else rpc.BackendType[args.backend],
            "rank": args.rank,
            "world_size": args.world_size,
            "rpc_backend_options": rpc.RpcBackendOptions(rpc_timeout=args.timeout, init_method=args.init_method),
        }
        args = []

    rpc.init_rpc(*args, **kwargs)
    rpc.shutdown()


if __name__ == "__main__":
    run_rpc_worker()
