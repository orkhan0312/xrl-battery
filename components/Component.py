# from dataclasses import dataclass, asdict
#
# class ComponentError(Exception):
#     pass
#
# @dataclass
# class BaseComponent:
#     """Base class for all components"""
#     name: str
#
#     def to_dict(self) -> dict:
#         return asdict(self)
#
#     @classmethod
#     def from_dict(cls, data: dict):
#         return cls(**data)
#
