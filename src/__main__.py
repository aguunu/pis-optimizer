from collections.abc import Iterable
from .experiment import run_jobs
from .models import Problem
from collections import defaultdict
from .greedy import greedy
from itertools import product
from concurrent import futures
import os
from .reader import load_data


def create_greedy_pool(problems: Iterable[Problem], size: int):
    greedy_pool_by_problem = defaultdict(list)
    with futures.ProcessPoolExecutor() as executor:
        for problem in problems:
            greedy_pool_by_problem[problem.problem_tag] = [
                result for result in executor.map(greedy, [problem] * size) if result
            ]
    return greedy_pool_by_problem


if __name__ == "__main__":
    runs = 30
    generations = 200
    p_mutations = [0.1, 0.01, 0.001]
    p_crossovers = [1.0, 0.9, 0.6]
    populations = [80, 120, 160]

    configurations = {
        f"C{i:02}": config
        for i, config in enumerate(product(p_mutations, p_crossovers, populations))
    }

    data_path = os.path.join("data")
    problems = load_data(data_path)

    greedy_pool_size = 200
    greedy_pool_by_problem = create_greedy_pool(problems, greedy_pool_size)

    output_dir = "results"
    run_jobs(
        problems, greedy_pool_by_problem, configurations, runs, generations, output_dir
    )
