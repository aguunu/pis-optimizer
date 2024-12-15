import numpy as np
from collections import deque
import random
from itertools import combinations
from .models import Problem
from .utils import search_start


def topological_order(d) -> list[int]:
    n = len(d)
    in_degree = np.sum(d, axis=0)  # número de dependencias de cada tarea
    queue = deque([i for i in range(n) if in_degree[i] == 0])  # tareas sin dependencias

    orden = []
    while queue:
        random.shuffle(queue)
        node = queue.popleft()
        orden.append(node)

        # reducir el grado de entrada de las tareas dependientes
        for i in range(n):
            if d[node, i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    if len(orden) == n:  # si no hay ciclos, el orden tiene todos los nodos
        return orden
    else:
        raise ValueError(
            "El grafo tiene un ciclo, no se puede realizar un orden topológico."
        )


def greedy(problem: Problem):
    total_tasks = problem.total_tasks
    total_workers = problem.total_workers

    assignments = np.zeros(shape=(total_workers, total_tasks), dtype=np.int64)
    starts = np.zeros(shape=(total_tasks), dtype=np.int64)

    work_load = np.zeros(shape=(total_workers))
    for task in reversed(topological_order(problem.task_dependencies)):
        inf = max(
            (
                starts[t] + problem.task_duration[t] + 1
                for t, depends in enumerate(problem.task_dependencies[task])
                if depends == 1
            ),
            default=0,
        )
        best_workers = work_load.argsort()

        for workers in combinations(best_workers, problem.task_demand[task]):
            # armar las restricciones del grupo
            rs = []
            for w in workers:
                rs.extend(problem.worker_restrictions[w])
                rs.extend(
                    [
                        (
                            starts[assignment],
                            starts[assignment] + problem.task_duration[assignment],
                        )
                        for assignment in range(total_tasks)
                        if assignments[w][assignment] == 1
                    ]
                )

            # encontrar inicio valido
            if start := search_start(
                rs, problem.task_duration[task], inf, problem.max_slot
            ):
                starts[task] = start
                for w in workers:
                    work_load[w] += problem.task_duration[task]
                    assignments[w][task] = 1
                break

        if assignments[:, task].sum() != problem.task_demand[task]:
            return None

    return assignments, starts
