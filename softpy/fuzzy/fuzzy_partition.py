
from .fuzzyset import FuzzySet


class FuzzyPartition():
    '''
    Implements a fuzzy partition, that is a collection of fuzzy sets that generalizes a standard partition.
    It does not enforce any specific constraints (e.g., membership degrees summing to 1 as in Ruspini partitions), so it
    can be used also to implement non-standard definitions of fuzzy partition

    Parameters
    ----------
    :param name: the name of the fuzzy partition
    :type name: str

    :param fuzzy_sets: a dictionary of fuzzy sets
    :type fuzzy_sets: dict[str, FuzzySet]
    '''
    def __init__(self, name: str, fuzzy_sets: dict[str, FuzzySet]) -> None:
        if not isinstance(name, str):
            raise TypeError('name should be a string')
        
        if name == '':
            raise ValueError('name should be non empty strings')
        
        if not isinstance(fuzzy_sets, dict):
            raise TypeError('fuzzy_sets should be a dict')
        
        if len(fuzzy_sets) == 0:
            raise ValueError('fuzzy_sets should be a non empty dict')
        
        for k, v in fuzzy_sets.items():
            if not isinstance(k, str):
                raise TypeError('every keys should be a string')
            
            if k == '':
                raise TypeError('keys should be non empty strings')
            
            if not isinstance(v, FuzzySet):
                raise TypeError('every values should be a fuzzy set')
        
        self.__name: str = name
        self.__fuzzy_sets: dict = fuzzy_sets
    
    @property
    def name(self) -> str:
        return self.__name
    
    def get_fuzzy_set_names(self):
        return list(self.__fuzzy_sets.keys())
    
    def __getitem__(self, name: str):
        if name not in self.get_fuzzy_set_names():
            raise ValueError('fuzzy set called name doesn\'t exists')

        return self.__fuzzy_sets[name]