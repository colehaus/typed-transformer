from typing import Any, type_check_only

from jax.numpy import ndarray

@type_check_only
class Leaves[A](list[ndarray[*tuple[Any, ...], float]]): ...

def tree_leaves[A](tree: A, is_leaf: None = None) -> Leaves[A]: ...
