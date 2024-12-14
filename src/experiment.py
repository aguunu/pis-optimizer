from collections.abc import Iterable
import os
import concurrent.futures
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.util.comparator import DominanceWithConstraintsComparator
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
import gc

from .models import Problem
from .ga import CustomMutation, CustomProblem, CustomCrossover

from jmetal.lab.experiment import print_function_values_to_file, print_variables_to_file


def run_job(x):
    collected_results = []
    algorithm_tag, data, pm, pc, p, runs, generations, greedy_pool = x

    for _ in range(runs):
        problem = CustomProblem(data=data, greedy_pool=greedy_pool)

        algorithm = NSGAII(
            problem=problem,
            population_size=p,
            offspring_population_size=p,
            mutation=CustomMutation(pm, data),
            crossover=CustomCrossover(pc, data),
            termination_criterion=StoppingByEvaluations(max_evaluations=generations),
            dominance_comparator=DominanceWithConstraintsComparator(),
        )
        print("F\n", end="")

        algorithm.run()
        collected_results.append(
            (algorithm_tag, algorithm.result(), algorithm.total_computing_time)
        )

        del algorithm
        gc.collect()

    return collected_results


def run_jobs(
    problems: Iterable[Problem],
    greedy_pool_by_problem,
    configurations,
    runs: int,
    generations: int,
    output_dir: str = "",
):
    reference_fronts_dir = "reference_fronts"
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data in problems:
            solutions = []
            results = executor.map(
                run_job,
                [
                    (
                        algorithm_tag,
                        data,
                        pm,
                        pc,
                        p,
                        runs,
                        generations,
                        greedy_pool_by_problem[data.problem_tag],
                    )
                    for algorithm_tag, (pm, pc, p) in configurations.items()
                ],
            )
            for r in list(results):
                for run, (algorithm_tag, result, time) in enumerate(r):
                    dir_name = os.path.join(output_dir, algorithm_tag, data.problem_tag)

                    filename = os.path.join(dir_name, "FUN.{}.tsv".format(run))
                    print_function_values_to_file(result, filename=filename)

                    filename = os.path.join(dir_name, "VAR.{}.tsv".format(run))
                    print_variables_to_file(result, filename=filename)

                    filename = os.path.join(dir_name, "TIME.{}.tsv".format(run))
                    with open(filename, "w+") as of:
                        of.write(str(time))
                    solutions.extend(result)

            print("E\n", end="")
            front = get_non_dominated_solutions(solutions)

            print_function_values_to_file(
                front, os.path.join(reference_fronts_dir, data.problem_tag + ".pf")
            )
            print_variables_to_file(
                front, os.path.join(reference_fronts_dir, data.problem_tag + ".VAR.tsv")
            )
            del results
            del solutions
            del front
            gc.collect()
