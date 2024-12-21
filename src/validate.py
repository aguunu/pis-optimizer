import numpy.typing as npt
from .ga.problem import Problem
from itertools import combinations
import logging
from .utils import overlap


def validate(problem: Problem, assignments: npt.NDArray, starts: npt.NDArray):
    for w in range(problem.total_workers):
        for t1, t2 in combinations(range(problem.total_tasks), 2):
            i1 = (starts[t1], starts[t1] + problem.task_duration[t1])
            i2 = (starts[t2], starts[t2] + problem.task_duration[t2])

            if assignments[w][t1] == 1 and assignments[w][t2] == 1 and overlap(i1, i2):
                logging.error("Viola superposición.")
                return 1

        for t in range(problem.total_tasks):
            i1 = (starts[t], starts[t] + problem.task_duration[t])
            if assignments[w][t] and any(
                overlap(r, i1) for r in problem.worker_restrictions[w]
            ):
                logging.error("Viola restricción horaria.")
                return 1

    for t in range(problem.total_tasks):
        if assignments[:, t].sum() != problem.task_demand[t]:
            logging.error("Viola demanda.")
            return 1

    for t1, t2 in combinations(range(problem.total_tasks), 2):
        if (
            problem.task_dependencies[t1][t2] == 1
            and starts[t2] + problem.task_duration[t2] >= starts[t1]
        ):
            logging.error(f"Viola dependencia. {t1} -> {t2}")
            return 1
    return 0
