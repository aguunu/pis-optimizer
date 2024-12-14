from jmetal.core.problem import IntegerProblem
import numpy as np
import random
from jmetal.core.problem import IntegerSolution
from ..models import Problem


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
        # preference_score = 0
        # work_load = np.zeros(shape=self.data.total_workers)

        total_tasks = self.data.total_tasks
        total_workers = self.data.total_workers

        offset = total_workers * total_tasks
        assignments = np.array(solution.variables[:offset]).reshape((total_workers, total_tasks))

        task_preferences = np.sum(assignments * self.data.task_preferences, axis=1)

        acc_group_preferences = (self.data.group_preferences @ assignments) * assignments
        mean_group_preference_by_task_worker = acc_group_preferences / self.data.task_demand
        mean_group_preference_by_worker = mean_group_preference_by_task_worker.sum(axis=1)

        preferences = task_preferences + mean_group_preference_by_worker
        
        # normalize
        tasks_by_worker = assignments.sum(axis=1)
        preference_score = np.sum(preferences / tasks_by_worker)

        solution.objectives[0] = - preference_score
        solution.objectives[1] = np.var(assignments @ self.data.task_demand)
        
        return solution
        # task_workers = [
        #     [
        #         w
        #         for w in range(total_workers)
        #         if solution.variables[total_tasks * w + t] == 1
        #     ]
        #     for t in range(total_tasks)
        # ]
        # acc_preferences = [0.0] * total_workers
        # assignments_per_worker = [
        #     [
        #         t
        #         for t in range(total_tasks)
        #         if solution.variables[total_tasks * w + t] == 1
        #     ]
        #     for w in range(total_workers)
        # ]
        #
        # for t, workers in enumerate(task_workers):
        #     # work load
        #     for w1 in workers:
        #         work_load[w1] += self.data.task_duration[t]
        #
        #         # task preference
        #         acc_preferences[w1] += self.data.task_preferences[w1][t]
        #
        #         # group preference
        #         acc = 0.0
        #         for w2 in workers:
        #             acc += self.data.group_preferences[w1][w2]
        #         acc_preferences[w1] += acc / len(workers)
        #
        # preference_score = (
        #     sum(
        #         [
        #             acc_preferences[w] / len(tasks)
        #             for w, tasks in enumerate(assignments_per_worker)
        #             if tasks
        #         ]
        #     )
        #     / total_workers
        # )
        #
        # solution.objectives[0] = -1.0 * preference_score
        # solution.objectives[1] = float(np.var(work_load))
        #
        # return solution

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
