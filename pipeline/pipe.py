from functools import partial
import math
from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:

    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    # construct each columns of the above table separately
    jobs = [
        (
            # bubbles at the start of the column
            j * [None]
            # the job_steps in the job
            + [(i, j) for i in range(num_partitions)]
            # bubbles at the end of the column
            + (num_batches - 1 - j) * [None]
        ) for j in range(num_batches)
    ]

    # yield each row of the schedule by zipping the columns
    for clock_cycle in zip(*jobs):
        yield [job_step for job_step in clock_cycle if job_step]  # drop bubbles
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.

        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        n, *_ = x.shape
        m = math.ceil(n / self.split_size)  # number of microbatches
        microbatches = [{0: mb)} for mb in torch.split(x, self.split_size, dim=0)]

        for schedule in _clock_cycles(num_batches=m, num_partitions=len(self.devices)):
            self.compute(microbatches, schedule)

        return torch.cat([microbatches[i][len(self.devices)] for i in range(m)], dim=0).to(device=self.devices[-1])
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker.
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        # dispatch the tasks
        for (mb_id, device_id) in schedule:
            job_step = partial(self.partitions[device_id], batches[mb_id][device_id].to(self.devices[device_id]))
            self.in_queues[device_id].put(Task(job_step))

        # process the results
        for (mb_id, device_id) in schedule:
            success, result = self.out_queues[device_id].get()  # blocks until results are available
            if not success:
                raise RuntimeError(result)
            _, batches[mb_id][device_id + 1] = result  # result is task, batch
        # END ASSIGN5_2_2
