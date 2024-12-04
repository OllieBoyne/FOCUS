import dataclasses
import argparse
import json
from pathlib import Path

@dataclasses.dataclass(frozen=True)
class Hyperparameters:

    @classmethod
    def add_to_argparse(cls, parser: argparse.ArgumentParser):
        """Add hyperparameters to an argparse parser."""
        for field in dataclasses.fields(cls):
            if field.type == bool:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action='store_false', help=f"{field.name}: {field.default}")
                parser.add_argument(f"--{field.name}", dest=field.name, action='store_true', help=f"{field.name}: {field.default}")
                parser.set_defaults(**{field.name: field.default})
            else:
                parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)

    @classmethod
    def from_args(cls, args: argparse.Namespace, **kwargs) -> 'Hyperparameters':
        """Create a Hyperparameter object from argparse arguments. Override with any kwargs"""
        inputs = {field.name: getattr(args, field.name) for field in dataclasses.fields(cls)}
        inputs.update(kwargs)
        return cls(**inputs)

    def save(self, path: str | Path):
        """Save hyperparameters to a file."""
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f)
