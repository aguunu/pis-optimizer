def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def search_start(rs: list[tuple[int, int]], duration: int, inf: int, sup: int):
    for start in range(inf, sup - duration):
        interval = (start, start + duration)

        if all(not overlap(r, interval) for r in rs):
            return start
    return None
