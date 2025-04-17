from abc import ABC, ABCMeta
from typing import Type, TypeVar, List

import weakref

T = TypeVar("T", bound="GlobalAccess")

class GlobalAccessMeta(ABCMeta):
    _instances: List[weakref.ReferenceType[T]]

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._instances: List["GlobalAccess"] = list()

    def _refresh_refs(cls):
        cls._instances = [ref for ref in cls._instances if ref() is not None]

class GlobalAccess(ABC, metaclass=GlobalAccessMeta):
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls)
        cls._instances.append(weakref.ref(instance))
        return instance

    @classmethod
    def instance(cls: Type[T]) -> T:
        cls._refresh_refs()
        if len(cls._instances) <= 0:
            raise ValueError(f"Instance of {cls} has not been created yet.")
        if len(cls._instances) > 1:
            raise ValueError(
                f"Multiple instances of {cls} have been created, "
                "please use `{cls}.instances`"
            )
        return cls._instances[0]()

    @classmethod
    def instances(cls: Type[T]) -> List[T]:
        cls._refresh_refs()
        return [ref() for ref in cls._instances]
            
