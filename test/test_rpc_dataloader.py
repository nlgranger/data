import functools
import itertools

import pytest
import torch
from tblib import pickling_support
from torch import multiprocessing as mp
from torch.distributed import rpc
from torch.utils.data import Dataset, IterableDataset

from torchdata.rpc import RPCDataloader


pickling_support.install()

num_workers = 2


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()

    def run(self):
        try:
            out = super().run()
            self._cconn.send((True, out))
        except Exception as e:
            self._cconn.send((False, e))

    def return_value(self, timeout):
        if self._pconn.poll(timeout=timeout):
            success, value = self._pconn.recv()
            self._pconn.close()

            if success:
                return value
            else:
                raise value

        else:
            self._pconn.close()
            raise TimeoutError()


def run_in_subprocess(timeout):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            p = Process(target=f, args=args, kwargs=kwargs)
            p.start()
            return p.return_value(timeout)

        return wrapped

    return decorator


def worker_main(name, rank, world_size, init_method):
    rpc.init_rpc(
        name=name,
        backend=rpc.BackendType.TENSORPIPE,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
    )
    rpc.shutdown()


def init_rpc(num_workers):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            port = torch.randint(29400, 29500, (1,)).item()
            world_size = num_workers + 1
            init_method = f"tcp://127.0.0.1:{port}"

            workers = [
                mp.Process(target=worker_main, args=[f"worker{i}", i, world_size, init_method])
                for i in range(world_size - 1)
            ]
            for w in workers:
                w.start()

            rpc.init_rpc(
                name="main",
                backend=rpc.BackendType.TENSORPIPE,
                rank=num_workers,
                world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
            )

            try:
                return f(*args, **kwargs)
            finally:
                rpc.shutdown()

        return wrapped

    return decorator


class D(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= self.n:
            raise IndexError()
        return index


class ID(IterableDataset):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        yield from range(self.n)


@pytest.mark.parametrize("batch_size", [None, 3])
@run_in_subprocess(600)
@init_rpc(num_workers=num_workers)
def test_iterable(batch_size):
    n = 59

    datasets = [rpc.remote(f"worker{i}", ID, (n,)) for i in range(num_workers)]

    dataloader = RPCDataloader(datasets, batch_size=batch_size)

    if batch_size:
        indices = torch.arange(n).split(batch_size)
        indices = [indices[i // num_workers] for i in range(num_workers * len(indices))]
        for i, v in itertools.zip_longest(indices, dataloader):
            assert i.eq(v).all()
    else:
        indices = torch.arange(num_workers * n).div_(num_workers, rounding_mode="floor").tolist()
        for i, v in itertools.zip_longest(indices, dataloader):
            assert i == v


@pytest.mark.parametrize("batch_size", [None, 3])
@run_in_subprocess(10)
@init_rpc(num_workers=num_workers)
def test_mappable(batch_size):
    n = 59

    datasets = [rpc.remote(f"worker{i}", D, (n,)) for i in range(num_workers)]

    if batch_size:
        sampler = None
        batch_sampler = torch.randperm(n).split(batch_size)
    else:
        sampler = torch.randperm(n)
        batch_sampler = None

    dataloader = RPCDataloader(datasets, batch_size=batch_size, sampler=sampler, batch_sampler=batch_sampler)

    if batch_size:
        for i, v in itertools.zip_longest(batch_sampler, dataloader):
            assert i.eq(v).all()
    else:
        for i, v in itertools.zip_longest(sampler, dataloader):
            assert i == v
