from typing import Protocol, TypeVar

T = TypeVar("T")

class Comparable(Protocol[T]):
    def __eq__(self: T, other: T) -> bool:
        pass

    def __lt__(self: T, other: T) -> bool:
        '''
        if other == np.NINF, __lt__ should return False
        '''
        pass

    def __ge__(self: T, other: T) -> bool:
        '''
        if other == np.NINF, __lt__ should return True
        '''
        pass 