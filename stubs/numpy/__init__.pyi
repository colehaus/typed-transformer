# pylint: skip-file

from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypeVar, TypeVarTuple, overload

Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
DType = TypeVar("DType")
DType2 = TypeVar("DType2")
Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)
Dim3 = TypeVar("Dim3", bound=int)
One = Literal[1]
Float = TypeVar("Float", bound=float)
Float2 = TypeVar("Float2", bound=float)

class Sum[*Shape](int): ...
class Product[*Shape](int): ...
class Fin[Int: int](int): ...

class ndarray[*Shape, DType]:
    dtype: type[DType]
    nbytes: int
    def item(self: ndarray[DType]) -> DType: ...
    def __ne__(self, other: int) -> ndarray[*Shape, bool]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @overload
    def __eq__(self: ndarray[Dim1, One, DType], other: ndarray[One, Dim2, DType]) -> ndarray[Dim1, Dim2, bool]: ...
    @overload
    def __eq__(self, other: int) -> ndarray[*Shape, bool]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def __lt__(self: ndarray[*Shape, Float], other: float) -> ndarray[*Shape, bool]: ...
    def __ge__(self: ndarray[*Shape, int], other: float) -> ndarray[*Shape, bool]: ...
    def __add__(self, other: ndarray[*Shape, DType]) -> ndarray[*Shape, DType]: ...
    def __sub__(self, other: float) -> ndarray[*Shape, DType]: ...
    def __rsub__(self, other: int) -> ndarray[*Shape, DType]: ...
    def __truediv__(
        self: ndarray[*Shape, Float] | ndarray[*Shape, int], other: int | Float | ndarray[Float]
    ) -> ndarray[*Shape, Float]: ...
    @overload
    def __mul__(
        self: ndarray[*Shape, Float], other: ndarray[*Shape, int] | ndarray[*Shape, Float]
    ) -> ndarray[*Shape, Float]: ...
    @overload
    def __mul__(
        self: ndarray[Dim1, One, DType], other: ndarray[One, Dim2, DType]
    ) -> ndarray[Dim1, Dim2, DType]: ...
    def astype(self, dtype: type[DType2]) -> ndarray[*Shape, DType2]: ...
    @overload
    def __getitem__(
        self: ndarray[Dim1, *Shape2, DType], key: tuple[ndarray[Fin[Dim1]], ellipsis]
    ) -> ndarray[*Shape2, DType]: ...
    @overload
    def __getitem__(self: ndarray[Dim1, DType], key: slice) -> ndarray[int, DType]: ...
    @overload
    def __setitem__(self: ndarray[*Shape2, Dim1, DType], key: tuple[ellipsis, int], value: DType) -> None: ...
    @overload
    def __setitem__(self: ndarray[Dim1, Dim2, DType], key: tuple[int, slice], value: list[DType]) -> None: ...
    def __iter__(self: ndarray[Dim1, *Shape2, DType]) -> Iterator[ndarray[*Shape2, DType]]: ...
    @property
    def shape(self: ndarray[*Shape, DType]) -> tuple[*Shape]: ...
    @property
    def T(self: ndarray[Dim1, Dim2, DType]) -> ndarray[Dim2, Dim1, DType]: ...
    def __matmul__(
        self: ndarray[Dim1, Dim2, DType], other: ndarray[Dim2, Dim3, DType]
    ) -> ndarray[Dim1, Dim3, DType]: ...

@overload
def array(object: tuple[DType]) -> ndarray[One, DType]: ...
@overload
def array(object: Sequence[Float]) -> ndarray[int, Float]: ...
@overload
def array(object: Float) -> ndarray[Float]: ...
@overload
def array(object: Sequence[ndarray[*Shape, DType]]) -> ndarray[Any, *Shape, DType]: ...
@overload
def mean(a: ndarray[*Shape, Float]) -> ndarray[Float]: ...
@overload
def mean(
    a: ndarray[Dim1, Dim2, *Shape, Float],
    axis: Literal[1],
    where: ndarray[Dim1, Dim2, *Shape, bool] | None = None,
) -> ndarray[Dim1, *Shape, Float]: ...
def arange[Stop: int](stop: Stop) -> ndarray[Stop, Fin[Stop]]: ...
@overload
def ones(shape: Dim1, dtype: type[DType] = float) -> ndarray[Dim1, DType]: ...
@overload
def ones(shape: tuple[*Shape], dtype: type[DType] = float) -> ndarray[*Shape, DType]: ...
@overload
def where(
    condition: ndarray[*Shape, bool],
    x: ndarray[*Shape, DType] | ndarray[DType],
    y: ndarray[*Shape, DType] | ndarray[DType],
) -> ndarray[*Shape, DType]: ...
@overload
def where(
    condition: ndarray[Dim1, One, bool], x: ndarray[DType], y: ndarray[Dim1, Dim2, DType]
) -> ndarray[Dim1, Dim2, DType]: ...
def full(shape: tuple[*Shape], fill_value: DType) -> ndarray[*Shape, DType]: ...
@overload
def expand_dims(
    a: ndarray[*Shape, Dim1, DType], axis: Literal[-2]
) -> ndarray[*Shape, Literal[1], Dim1, DType]: ...
@overload
def expand_dims(a: ndarray[*Shape, DType], axis: Literal[-1]) -> ndarray[*Shape, Literal[1], DType]: ...
@overload
def expand_dims(a: ndarray[*Shape, DType], axis: Literal[0]) -> ndarray[Literal[1], *Shape, DType]: ...
@overload
def expand_dims(a: ndarray[Dim1, *Shape, DType], axis: Literal[1]) -> ndarray[Dim1, Literal[1], *Shape, DType]: ...
@overload
def expand_dims(
    a: ndarray[Dim1, DType], axis: tuple[Literal[0], Literal[2]]
) -> ndarray[Literal[1], Dim1, Literal[1], DType]: ...
@overload
def argmax(a: ndarray[Dim1, DType]) -> ndarray[Fin[Dim1]]: ...
@overload
def argmax(a: ndarray[*Shape, Dim1, DType], axis: Literal[-1]) -> ndarray[*Shape, Fin[Dim1]]: ...
def tril(m: ndarray[Dim1, Dim1, DType], k: int = 0) -> ndarray[Dim1, Dim1, DType]: ...
def pad(
    array: ndarray[Dim1, DType],
    pad_width: tuple[int, int],
    constant_values: DType,
    mode: Literal["constant"] = "constant",
) -> ndarray[int, DType]: ...
def roll(a: ndarray[*Shape, DType], shift: int, axis: int) -> ndarray[*Shape, DType]: ...
def concatenate(
    arrays: Sequence[ndarray[Any, *Shape, DType]], axis: Literal[0]
) -> ndarray[int, *Shape, DType]: ...
def flip(a: ndarray[*Shape, DType], axis: int) -> ndarray[*Shape, DType]: ...
def any(a: ndarray[*Shape, Dim1, bool], axis: Literal[-1]) -> ndarray[*Shape, bool]: ...
def sqrt(x: int) -> ndarray[float32]: ...
def reshape(a: ndarray[*Shape, DType], newshape: tuple[*Shape2]) -> ndarray[*Shape2, DType]: ...
def dot(a: ndarray[Dim1, DType], b: ndarray[Dim1, DType]) -> ndarray[DType]: ...
def average(
    a: ndarray[Dim1, Dim2, DType], axis: Literal[0], weights: ndarray[Dim1, DType]
) -> ndarray[Dim2, DType]: ...

class floating(float): ...
class float64(floating): ...
class float32(floating): ...

class finfo[DType]:
    def __init__(self, dtype: type[DType]) -> None: ...
    max: DType
    min: DType
    eps: DType
    bits: int
