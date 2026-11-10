"""Keep the physical 8x8 source array independent of numerical padding."""


def source_indices(config):
    size = config.N_super_lattice
    count = 8
    if size < count or (size - count) % 2:
        raise ValueError("N_super_lattice must be even and at least 8 for the 8x8 source array")
    start = (size - count) // 2
    return range(start, start + count)
