from torch import Tensor 

def print_fields(dataclass_obj):
    """
    print the fields of a dataclass object
    Args:
        dataclass_obj:
    """
    from dataclasses import fields
    for f in fields(dataclass_obj):
        val = getattr(dataclass_obj, f.name)
        if isinstance(val, Tensor):
            print(f"{f.name}: {val.shape}")
        else:
            print(f"{f.name}: {type(val)}")
