import os
import numpy as np
import pandas as pd
from .models import Problem


def hhmm_to_slot(hhmm: str, slot_size: int):
    hh, mm = hhmm.split(":")
    return int((int(hh) * 60 + int(mm)) / slot_size)


def load_problem(problem_path: str):
    # info.csv
    df_info = pd.read_csv(os.path.join(problem_path, "info.csv"))
    days = df_info["days"][0]
    problem_tag = df_info["problem_tag"][0]

    slot_size = 30

    # integrantes.csv
    # id, nombre
    df_integrantes = pd.read_csv(
        os.path.join(problem_path, "integrantes.csv"), header=None
    )
    total_members = len(df_integrantes)

    # tareas.csv
    # id, duración, id_tipo, demanda
    df_tareas = pd.read_csv(os.path.join(problem_path, "tareas.csv"), header=None)
    total_tasks = len(df_tareas)

    duration = (
        df_tareas[1]
        .apply(lambda x: hhmm_to_slot(x, slot_size))
        .to_numpy(dtype=np.int64)
    )
    task_types = df_tareas[2].to_numpy()
    task_demand = df_tareas[3].to_numpy()

    # id_worker, id_tipo, preferencia
    dict_task_preferences = {}
    df_preferences = pd.read_csv(
        os.path.join(problem_path, "preferencias_tareas.csv"), header=None
    )
    for w, task_type, preference in df_preferences.itertuples(index=False):
        dict_task_preferences[(w, task_type)] = preference

    preferences = np.zeros(shape=(total_members, total_tasks), dtype=np.int64)
    for w in range(total_members):
        for t in range(total_tasks):
            key = (w, task_types[t])
            preferences[w][t] = (
                dict_task_preferences[key] if key in dict_task_preferences.keys() else 0
            )

    df_group_preferences = pd.read_csv(
        os.path.join(problem_path, "preferencias_grupos.csv"), header=None
    )
    preferences_group = np.zeros(shape=(total_members, total_members), dtype=np.int64)
    for w1, w2, preference in df_group_preferences.itertuples(index=False):
        preferences_group[w1][w2] = preference
        preferences_group[w2][w1] = preference
    np.fill_diagonal(preferences_group, 0)

    df_dependencies = pd.read_csv(
        os.path.join(problem_path, "dependencias.csv"), header=None
    )
    dependencies = np.zeros(shape=(total_tasks, total_tasks), dtype=np.int64)
    for t1, t2 in df_dependencies.itertuples(index=False):
        dependencies[t1][t2] = 1

    df_restrictions = pd.read_csv(
        os.path.join(problem_path, "restricciones_horarias.csv"), header=None
    )
    restrictions = [list() for _ in range(total_members)]
    for w, start, end in df_restrictions.itertuples(index=False):
        for i in range(days):
            day_offset = i * ((60 / slot_size) * 24)
            interval = (
                int(day_offset + hhmm_to_slot(start, slot_size)),
                int(day_offset + hhmm_to_slot(end, slot_size)),
            )
            restrictions[w].append(interval)

    return Problem(
        task_dependencies=dependencies,
        task_demand=task_demand,
        task_duration=duration,
        task_type=task_type,
        worker_restrictions=restrictions,
        task_preferences=preferences,
        group_preferences=preferences_group,
        slot_size=slot_size,
        days=days,
        problem_tag=problem_tag,
    )


def load_data(data_path: str):
    problems = []
    for entry in os.listdir(data_path):
        entry_path = os.path.join(data_path, entry)
        if os.path.isdir(entry_path):
            problem = load_problem(entry_path)
            problems.append(problem)

    return problems
