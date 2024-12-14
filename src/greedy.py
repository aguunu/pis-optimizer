import numpy as np
from collections import deque
import random
from itertools import combinations
from .models import Problem
from .utils import search_start, overlap

def orden_topologico(d) -> list[int]:
    """Devuelve el orden topológico aleatorio de las tareas, si existe."""
    n = len(d)
    in_degree = np.sum(d, axis=0)  # Número de dependencias de cada tarea (entradas)
    queue = deque([i for i in range(n) if in_degree[i] == 0])  # Tareas sin dependencias

    orden = []
    while queue:
        random.shuffle(queue)
        node = queue.popleft()
        orden.append(node)

        # Reducir el grado de entrada de las tareas dependientes
        for i in range(n):
            if d[node, i] == 1:  # Si node depende de i
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    if len(orden) == n:  # Si no hay ciclos, el orden tiene todos los nodos
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

    greedy_variance = 1  # random.randint(0, 1)

    work_load = np.zeros(shape=(total_workers))
    # done = [False] * total_tasks
    for task in reversed(orden_topologico(problem.task_dependencies)):
        # done[task] = True
        # inf = max([starts[t] + duration[t] + 1 for t, x in enumerate(transitive_dependencies[task]) if x == 1 and done[t]], default=0)
        inf = max(
            [
                starts[t] + problem.task_duration[t] + 1
                for t, depends in enumerate(problem.task_dependencies[task])
                if depends == 1
            ],
            default=0,
        )
        if greedy_variance == 0:
            worker_pref = problem.task_preferences[:, task]
            best_workers = worker_pref.argsort()[::-1]
        else:
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
            start = search_start(rs, problem.task_duration[task], inf, problem.max_slot)
            if not start:
                continue

            starts[task] = start
            for w in workers:
                work_load[w] += problem.task_duration[task]
                assignments[w][task] = 1
            break
        if assignments[:, task].sum() != problem.task_demand[task]:
            print("GREEDY NOT VALID")
            return None

    return assignments, starts
