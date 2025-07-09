from __future__ import print_function

from typing import Literal

import numpy as np
from scipy import spatial

d2s_typing = Literal["exp", "log", "logistic", "gaussian", "inverse"]


def get_score(distance_transform: d2s_typing,**args,) -> callable:
    """
    Transforms the given distance into a score.
    """
    if distance_transform == "exp":
        return lambda x: np.exp(-x)
    elif distance_transform == "gaussian":
         sigma = args.get("sigma", 1)
         return lambda x: np.exp(-(x ** 2) / (2 * sigma ** 2))
    elif distance_transform == "log":
        return lambda x: -np.log(x)
    elif distance_transform == "logistic":
        return lambda x: 1 / (1 + np.exp(-x))
    elif distance_transform == "inverse":
        return lambda x: 1 / (1 + x)
    else:
        raise ValueError(f"Invalid distance : {distance_transform}")


def distance(
        v1, v2, d_type: Literal["absolute", "cosine", "square", "d2-norm"] = "cosine"
):
    assert v1.shape == v2.shape, "Shape of 2 vector is not equal !!!"

    if d_type == "absolute":
        return np.sum(np.absolute(v1 - v2))
    elif d_type == "d2-norm":
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == "cosine":
        return spatial.distance.cosine(v1, v2)
    elif d_type == "square":
        return np.sum((v1 - v2) ** 2)
    else:
        raise ValueError(f"Invalid distance type: {d_type}")