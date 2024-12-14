def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def search_start(rs: list[tuple[int, int]], duration: int, inf: int, sup: int):
    task_start = None
    possible_starts = list(range(inf, sup - duration))
    for start in possible_starts:
        interval = (start, start + duration)

        if all(not overlap(r, interval) for r in rs):
            task_start = start
            break
    return task_start
