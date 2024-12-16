from jmetal.core.problem import IntegerProblem
import numpy as np
import numpy.typing as npt
import random
from jmetal.core.problem import IntegerSolution
from ..models import Problem


def evaluate(
    data: Problem, assignments: npt.NDArray
) -> tuple[np.floating, np.floating]:
    task_preferences = np.sum(assignments * data.task_preferences, axis=1)

    acc_group_preferences = (data.group_preferences @ assignments) * assignments
    mean_group_preference_by_task_worker = acc_group_preferences / data.task_demand
    mean_group_preference_by_worker = mean_group_preference_by_task_worker.sum(axis=1)

    preferences = task_preferences + mean_group_preference_by_worker

    # normalize
    tasks_by_worker = assignments.sum(axis=1)
    preference_score = np.sum(preferences / tasks_by_worker)

    f1 = -preference_score
    f2 = np.var(assignments @ data.task_duration)

    return (f1, f2)


class CustomProblem(IntegerProblem):
    def __init__(self, data: Problem, greedy_pool):
        super(CustomProblem, self).__init__()

        self.data = data
        self.greedy_pool = greedy_pool

        # assignments
        assignments_lower_bound = [0] * (data.total_tasks * data.total_workers)
        assignments_upper_bound = [0] * (data.total_tasks * data.total_workers)

        # start time
        start_lower_bound = [data.min_slot] * data.total_tasks
        start_upper_bound = [data.max_slot] * data.total_tasks

        # define bounds
        self.lower_bound = assignments_lower_bound + start_lower_bound
        self.upper_bound = assignments_upper_bound + start_upper_bound

        # define objectives
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["affinity", "workload variance"]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        total_tasks = self.data.total_tasks
        total_workers = self.data.total_workers

        offset = total_workers * total_tasks
        assignments = np.array(solution.variables[:offset]).reshape(
            (total_workers, total_tasks)
        )

        f1, f2 = evaluate(self.data, assignments)

        solution.objectives[0] = f1
        solution.objectives[1] = f2

        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives(),
            self.number_of_constraints(),
        )

        assignments, starts = random.choice(self.greedy_pool)

        for t in range(self.data.total_tasks):
            new_solution.variables[
                self.data.total_tasks * self.data.total_workers + t
            ] = starts[t]
            for w in range(self.data.total_workers):
                new_solution.variables[self.data.total_tasks * w + t] = assignments[w][
                    t
                ]

        return new_solution

    def name(self):
        return "Custom Problem"
