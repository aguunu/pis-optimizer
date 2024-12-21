from jmetal.core.operator import Mutation
from jmetal.util.ckecking import Check
import random
from jmetal.core.problem import IntegerSolution
import numpy as np

from ..models import Problem
from ..utils import search_start


class CustomMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, data: Problem):
        super(CustomMutation, self).__init__(probability=probability)
        self.data = data

    def execute(self, source: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(source), IntegerSolution), "Solution type invalid")

        total_tasks = self.data.total_tasks
        total_workers = self.data.total_workers
        start_offset = total_tasks * total_workers

        assignments = np.array(source.variables[:start_offset]).reshape(
            (total_workers, total_tasks)
        )

        for t in range(total_tasks):
            if random.random() > self.probability:
                continue

            if self.data.task_demand[t] == total_workers:
                # podría reasignarse el inicio de la tarea
                continue

            # deallocate
            weights = [
                assignments @ self.data.task_duration,
                -(
                    (self.data.group_preferences @ assignments[:, t])
                    * assignments[:, t]
                    + assignments[:, t] * self.data.task_preferences[:, t]
                ),
            ]
            deallocate = assignments[:, t] * random.choice(weights).astype(np.float64)
            allocate = -(1 - assignments[:, t]) * random.choice(weights).astype(
                np.float64
            )

            # normalize
            eps = 1e-10
            deallocate /= np.max(abs(deallocate)) + eps
            allocate /= np.max(abs(allocate)) + eps

            # apply softmax
            e_x1 = np.exp(deallocate)
            p_deallocate = e_x1 / np.sum(e_x1)

            e_x2 = np.exp(allocate)
            p_allocate = e_x2 / np.sum(e_x2)

            w1 = np.random.choice(self.data.total_workers, p=p_deallocate)
            w2 = np.random.choice(self.data.total_workers, p=p_allocate)

            assignments[w1][t] = 0
            assignments[w2][t] = 1

            # armar restricciones
            rs = []
            for w in np.where(assignments[:, t] == 1)[0]:
                rs.extend(self.data.worker_restrictions[w])
                rs.extend(
                    [
                        (
                            source.variables[start_offset + other_t],
                            source.variables[start_offset + other_t]
                            + self.data.task_duration[other_t],
                        )
                        for other_t in range(total_tasks)
                        if source.variables[w * total_tasks + other_t] == 1
                    ]
                )

            # buscar máximo fin de tarea precedente
            inf = max(
                (
                    source.variables[start_offset + dependency]
                    + self.data.task_duration[dependency]
                    + 1
                    for dependency, depends in enumerate(self.data.task_dependencies[t])
                    if depends == 1
                ),
                default=0,
            )

            # buscar mínimo inicio de tarea sucesora
            sup = min(
                (
                    source.variables[start_offset + dependency]
                    for dependency, depends in enumerate(
                        self.data.task_dependencies.T[t]
                    )
                    if depends == 1
                ),
                default=0,
            )

            # buscar un start para t
            start = search_start(
                rs, duration=self.data.task_duration[t], inf=inf, sup=sup
            )
            if start:
                (
                    source.variables[total_tasks * w1 + t],
                    source.variables[total_tasks * w2 + t],
                ) = (
                    source.variables[total_tasks * w2 + t],
                    source.variables[total_tasks * w1 + t],
                )
                source.variables[start_offset + t] = start
            else:  # si no se pudo reparar pasar a la siguiente iteracion sin modificar la tarea
                continue
        return source

    def get_name(self):
        return "Custom Mutation"
