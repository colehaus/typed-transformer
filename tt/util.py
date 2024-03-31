from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from jax.numpy import ndarray

if TYPE_CHECKING:
    from numpy import Fin, Product


def unzip_list[A, B](l: Sequence[tuple[A, B]]) -> tuple[list[A], list[B]]:
    """Significantly faster than the naive `zip(*l)` implementation"""
    a = cast(list[A], [None] * len(l))
    b = cast(list[B], [None] * len(l))
    for i, (x, y) in enumerate(l):
        a[i] = x
        b[i] = y
    return a, b


def human_bytes(size: float, decimal_places: int = 2) -> str:
    unit = "B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:  # noqa: B007
        if size < 1024.0:  # noqa: PLR2004
            break
        size /= 1024.0

    formatted_num = f"{size:.{decimal_places}f}".rstrip("0").rstrip(".")
    return f"{formatted_num:>4} {unit}"


def product_[*Shape](operands: tuple[*Shape]) -> Product[*Shape]:
    """Retain type info when computing the product of a tuple."""
    return cast(Any, prod(cast(tuple[int, ...], operands)))


class fin[Int: int](int):  # noqa: N801
    """A type which witnesses that that the value is in the range [0, Int).
    Useful for e.g.:
        - Ensuring that an array index is within bounds
        - Describing the relationship between a vocabulary size and a token ID
        (i.e. a token ID is in the range [0, VocabSize))
    See https://hackage.haskell.org/package/fin
    """

    def __new__(cls, x: int | ndarray[int], max_: Int) -> Fin[Int]:
        assert 0 <= x < max_
        return cast(Any, x)


class LTE[A: int, B: int](int):
    """A refinement type on `A` that witnesses that `A <= B`.
    Or, from another view, it's a `Fin` in which we don't discard the input type
    but retain info about both values passed to the constructor.

    `LTE[Literal[2], Literal[3]]` is simply a `2` with additional evidence that `2 <= 3`.
    `LTE[Literal[4], Literal[3]]` OTOH is uninhabitable
    (i.e. any attempt to construct such a value would raise an assertion error).
    """

    def __new__(cls, a: A, b: B) -> LTE[A, B]:
        assert a <= b
        return cast(Any, a)


class InstanceSingleton[Label: str](int):
    """We ensure that there's only one value with a given label for a given instance.
    i.e. Any two values with type self.InstanceSingleton[Literal["foo"]] are equal.
    This is primarily used for defining "internal type variables".
    i.e. It's sometimes the case that the implementation of a class requires an annotating type
    (for e.g. an array dimension).
    But the class's user is uninterested in this type so we don't want to "pollute" the class with a type variable.
    (If we simply declare the type variable inside the class, pyright basically treats it as `Any`.)
    So instead we create an `InstanceSingleton` as the annotating type—it still gives us the guarantee
    a type variable would that any two arrays with this type for a dimension have the same size at runtime.
    """

    history: dict[tuple[int, str], int] = {}  # noqa: RUF012

    def __new__(cls, instance: Any, label: Label, value: int) -> InstanceSingleton[Label]:
        match cls.history.get((id(instance), label)):
            case None:
                cls.history[(id(instance), label)] = value
                return cast(Any, value)
            case v:
                assert v == value, (instance, label, v, value)
                return cast(Any, v)


class declare_axis[A]:  # noqa: N801
    """A helper for casting one axis of an array to a specific type."""

    @overload
    def __new__[*Shape, DType](
        cls, axis: Literal[0], array: ndarray[Any, *Shape, DType]
    ) -> ndarray[A, *Shape, DType]:
        ...

    @overload
    def __new__[Dim1, *Shape, DType](
        cls, axis: Literal[1], array: ndarray[Dim1, Any, *Shape, DType]
    ) -> ndarray[Dim1, A, *Shape, DType]:
        ...

    def __new__[*Shape, DType](cls, axis: int, array: ndarray[*Shape, DType]) -> ndarray[*Shape, DType]:
        return array


class declare_dtype[DType]:  # noqa: N801
    def __new__[*Shape](cls, array: ndarray[*Shape, Any]) -> ndarray[*Shape, DType]:
        return cast(Any, array)
