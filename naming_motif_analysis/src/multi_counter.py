from collections import Counter


class MultiCounter:
    """
    Tracks counts for an n-dimensional matrix with named dimensions.

    'add' operations must be fully specified.
    'get' operations can be partial to sum over dimensions.
    """

    def __init__(self, dimensions: list[str]):
        self.dimensions = dimensions
        self.dim2index = {dim: i for i, dim in enumerate(self.dimensions)}
        self.dim2keys = [list() for _ in self.dimensions]
        self.counts = Counter()

    def get_keys(self, dimension: str):
        """
        Return the keys for the specified dimension
        """
        index = self.dim2index.get(dimension)
        if index is None:
            raise KeyError(f"Error: '{dimension}' is not a valid dimension.")
        return self.dim2keys[index]

    def _get_index(self, dimension: str):
        if dimension not in self.dim2index:
            raise KeyError(f"Error: '{dimension}' is not a valid dimension.")
        return self.dim2index[dimension]

    def _add_key(self, dim_index: int, key: str):
        if key is None:
            raise ValueError("Error: 'add' cannot accept None as a key.")
        keys = self.dim2keys[dim_index]
        if not key in keys:
            keys.append(key)

    def add_keys(self, dimension: str, new_keys: list[str]):
        dim_index = self._get_index(dimension)
        for new_key in new_keys:
            self._add_key(dim_index, new_key)

    def add(self, amount: float, **keys):
        """
        Adds a count to a single, fully-specified coordinate.

        All dimensions must be provided, and no values can be None.
        """
        if len(keys) != len(self.dimensions):
            raise ValueError(
                f"Error: 'add' requires all dimensions. "
                f"Expected {self.dimensions}, got {list(keys.keys())}"
            )

        # Build the key tuple in the correct order
        key_list = [None] * len(self.dimensions)
        for dim, key in keys.items():
            index = self._get_index(dim)
            key_list[index] = key
            self._add_key(index, key)

        key_tuple = tuple(key_list)
        self.counts[key_tuple] += amount

    def _matches(self, data_key: tuple, query_key: tuple) -> bool:
        """
        Checks if a specific data_key matches a (possibly partial) query_key.

        A query_key's None is a wildcard.
        """
        for d_key, q_key in zip(data_key, query_key):
            if q_key is not None and d_key != q_key:
                return False
        return True

    def get(self, **keys) -> float:
        """
        Gets a count or a sum over dimensions.

        Omitted dimensions, or dimensions set to None, are treated as
        wildcards for summation.
        """
        # Build the (potentially partial) query tuple
        query_list = [None] * len(self.dimensions)
        for dim, key in keys.items():
            if dim not in self.dim2index:
                raise KeyError(f"Error: '{dim}' is not a valid dimension.")
            if key is not None:
                query_list[self.dim2index[dim]] = key

        query_tuple = tuple(query_list)

        # Use a generator expression to sum matching keys
        return sum(
            count
            for data_key, count in self.counts.items()
            if self._matches(data_key, query_tuple)
        )
