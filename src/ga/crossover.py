import copy
import random
from collections import defaultdict

from jmetal.core.operator import Crossover
from jmetal.core.problem import IntegerSolution
from ..models import Problem
from ..utils import search_start


class CustomCrossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float, data: Problem):
        super(CustomCrossover, self).__init__(probability=probability)
        self.data = data

    def execute(self, source: list[IntegerSolution]) -> list[IntegerSolution]:
        if random.random() > self.probability:
            return source

        offspring = copy.deepcopy(source[0])

        total_tasks = self.data.total_tasks
        total_workers = self.data.total_workers
        start_time_offset = total_tasks * total_workers
        mid_point = random.choice(range(total_tasks))

        current_restrictions = defaultdict(list)
        for w in range(total_workers):
            current_restrictions[w].extend(self.data.worker_restrictions[w])

        for t in sorted(
            range(total_tasks),
            key=lambda t: source[0].variables[start_time_offset + t],
        ):
            # seleccionar padre
            parent = t < mid_point
            # seleccionar asignados a t del padre seleccionado
            workers = [
                w
                for w in range(total_workers)
                if source[parent].variables[total_tasks * w + t] == 1
            ]
            # encontrar minimo tiempo de inicio
            inf = max(
                (
                    offspring.variables[start_time_offset + dependency]
                    + self.data.task_duration[dependency]
                    + 1
                    for dependency, depends in enumerate(self.data.task_dependencies[t])
                    if depends == 1
                ),
                default=self.data.min_slot,
            )
            # armar restricciones
            rs = []
            for w in workers:
                rs.extend(current_restrictions[w])
            # buscar nuevo inicio
            t_start = search_start(
                rs, self.data.task_duration[t], inf, self.data.max_slot
            )
            if not t_start:
                return source  # abort
            # crossover
            offspring.variables[start_time_offset + t] = t_start  # type: ignore
            for w in workers:
                current_restrictions[w].append(
                    (t_start, t_start + self.data.task_duration[t])
                )
            for w in range(total_workers):
                w_offset = total_tasks * w + t
                offspring.variables[w_offset] = source[parent].variables[w_offset]
        return [offspring]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "Custom Crossover"
