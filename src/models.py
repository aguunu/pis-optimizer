import attr
import numpy as np
import numpy.typing as npt


@attr.define(slots=True)
class Problem:
    task_dependencies: npt.NDArray[np.int_]
    task_demand: npt.NDArray[np.int_]
    task_duration: npt.NDArray[np.int_]
    task_type: npt.NDArray[np.int_]
    worker_restrictions: list[list[tuple[int, int]]]
    task_preferences: npt.NDArray[np.int_]
    group_preferences: npt.NDArray[np.int_]
    total_tasks: int = attr.field(init=False)
    total_workers: int = attr.field(init=False)
    min_slot: int = attr.field(init=False)
    max_slot: int = attr.field(init=False)
    slot_size: int
    days: int
    problem_tag: str

    def __attrs_post_init__(self):
        self.total_tasks = self.task_duration.shape[0]
        self.total_workers = self.group_preferences.shape[0]
        self.min_slot = 0
        self.max_slot = int((60 / self.slot_size) * 24 * self.days)
