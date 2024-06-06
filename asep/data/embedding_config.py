"""
Base configuration for the ASEP dataset module

Embedding Config Example
Schema:
    {
        'node_feat_type': str, # choices=['pre_cal', 'one_hot', 'custom']
        'ab': {
            'embedding_model': 'igfold', # choices=['igfold', 'esm2', 'custom']
            'custom_embedding_method': null  # Optional[Callable]
            'custom_embedding_method_src': {
                'script_path': str,
                'method_name': str
            }
        },
        'ag': {
            'embedding_model': 'esm2', # choices=['esm2', 'custom']
            'custom_embedding_method': null  # Optional[Callable]
            'custom_embedding_method_src': {
                'script_path': str,
                'method_name': str
            }
        }
    }

"""

import re
from pathlib import Path
from typing import Callable, Dict, Optional

from loguru import logger
from pydantic import BaseModel, ValidationError, root_validator, validator

# ----------------------------------------
# EmbeddingConfig
# ----------------------------------------
ALLOWED_NODE_FEAT_TYPES = ["custom", "pre_cal", "one_hot"]
ALLOWED_AG_EMBEDDING_MODELS = ["custom", "one_hot", "esm2"]
ALLOWED_AB_EMBEDDING_MODELS = ["custom", "one_hot", "esm2", "igfold"]


class CustomEmbeddingMethodSrc(BaseModel):
    script_path: Optional[str] = None
    method_name: Optional[str] = None

    @validator("script_path")
    def check_script_path(cls, v):
        if v is not None:
            try:
                assert Path(v).exists()
            except AssertionError:
                raise ValueError(f"Provided script_path '{v}' does not exist.")
        return v

    @validator("method_name")
    def check_method_name(cls, v, values):
        script_path = values.get("script_path", None)
        if script_path is not None:
            try:
                assert v is not None
            except AssertionError:
                raise ValueError(
                    "method_name must be provided if script_path is provided."
                )
            try:
                with open(script_path) as f:
                    # assert 'def ' + v in f.read()
                    assert re.search(r"def\s+" + v, f.read())
            except Exception as e:
                raise ValueError(
                    f"Provided method_name '{v}' not found in the script at '{script_path}'."
                )
        return v

    class Config:
        validate_assignment = True  # validation on assignment


class BaseConfig(BaseModel):
    custom_embedding_method: Optional[Callable] = (
        None  # either None or a callable,  if callable, it is loaded from script_path and function_name from custom_embedding_method_src
    )
    custom_embedding_method_src: CustomEmbeddingMethodSrc = CustomEmbeddingMethodSrc()

    class Config:
        validate_assignment = True  # validation on assignment


class ABConfig(BaseConfig):
    embedding_model: str = (
        "igfold"  # Default for 'ab', choices=['igfold', 'esm2', 'custom']
    )

    @validator("embedding_model")
    def check_embedding_model(cls, v):
        if v not in (l := ALLOWED_AB_EMBEDDING_MODELS):
            raise ValueError(f"embedding_model for AB must be among {l}")
        return v

    @root_validator
    def adjust_embedding_model_by_src(cls, values):
        """ if src.script_path or src.method_name is provided, set embedding_model to custom """
        src = values.get("custom_embedding_method_src")
        # if src.script_path is not None or src.method_name is not None, set embedding_model to custom
        if src.script_path is not None or src.method_name is not None:
            values["embedding_model"] = "custom"
        return values

    # examine if embedding_model is custom, then custom_embedding_method_src must be provided
    @root_validator
    def check_custom_embedding_method_src(cls, values):
        embedding_model = values.get("embedding_model")
        custom_embedding_method = values.get("custom_embedding_method")
        src = values.get("custom_embedding_method_src")
        if embedding_model == "custom":
            if custom_embedding_method is not None:
                return values
            else:
                if src.method_name is None or src.script_path is None:
                    raise ValueError(
                        "When `embedding_model` is set to 'custom' and "
                        "`custom_embedding_method` callable is still None, "
                        "both `script_path` and `method_name` in "
                        "`custom_embedding_method_src` ""must be provided."
                    )
        return values


class AGConfig(BaseConfig):
    embedding_model: str = (
        "esm2"  # Default for 'ag', choices=['esm2', 'one_hot', 'custom']
    )

    @validator("embedding_model")
    def check_embedding_model(cls, v):
        if v not in (l := ALLOWED_AG_EMBEDDING_MODELS):
            raise ValueError(f"embedding_model for AG must be among {l}")
        return v

    @root_validator
    def adjust_embedding_model_by_src(cls, values):
        """ if src.script_path or src.method_name is provided, set embedding_model to custom """
        src = values.get("custom_embedding_method_src")
        # if src.script_path is not None or src.method_name is not None, set embedding_model to custom
        if src.script_path is not None or src.method_name is not None:
            values["embedding_model"] = "custom"
        return values

    # examine if embedding_model is custom, then custom_embedding_method_src must be provided
    @root_validator
    def check_custom_embedding_method_src(cls, values):
        embedding_model = values.get("embedding_model")
        custom_embedding_method = values.get("custom_embedding_method")
        src = values.get("custom_embedding_method_src")
        if embedding_model == "custom":
            if custom_embedding_method is not None:
                return values
            else:
                if src.script_path is None or src.method_name is None:
                    raise ValueError(
                        "When `embedding_model` is set to 'custom' and \
                        `custom_embedding_method` callable is still None, \
                        both `script_path` and `method_name` in \
                        `custom_embedding_method_src` must be provided."
                    )
        return values


class EmbeddingConfig(BaseModel):
    # !!! ORDER OF DECLARATION MATTERS!!!
    ab: ABConfig = ABConfig()  # Default instance of ABConfig for 'ab'
    ag: AGConfig = AGConfig()  # Default instance of AGConfig for 'ag'
    node_feat_type: str = (
        "pre_cal"  # Default shared value choices=['pre_cal', 'one_hot', 'custom']
    )

    @validator("node_feat_type")
    def check_node_feat_type(cls, v):
        if v not in (l := ALLOWED_NODE_FEAT_TYPES):
            raise ValueError(f"node_feat_type must be among {l}")
        return v

    @root_validator
    def adjust_embedding_model(cls, values):
        node_feat_type = values.get("node_feat_type")
        if node_feat_type and node_feat_type == "one_hot":
            values["ab"].embedding_model = "one_hot"
            values["ag"].embedding_model = "one_hot"
        if node_feat_type and node_feat_type == "custom":
            values["ab"].embedding_model = "custom"
            values["ag"].embedding_model = "custom"
        if values["ab"].embedding_model == "custom" or values["ag"].embedding_model == "custom":
            """
            If `embedding_model` or either `ab` or `ag` is set to custom,
            then node_feat_type must be set to custom as well.
            """
            values['node_feat_type'] = 'custom'
        return values


# ********** Add more configs **********
