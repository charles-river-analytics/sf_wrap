"""
Author: David Rushing Dewhurst, ddewhurst@cra.com
Copyright Charles River Analytics Inc., 2022 - present
License: LGPL v3
"""

from collections import defaultdict
import logging

from jax import random
import jax.numpy as np
import numpyro

from sf_wrap import wrap


class Service:

    def __init__(self,):
        self._db = defaultdict(int)

    def get(self, key: str):
        return self._db[key]

    def post(self, x: dict):
        for (k, v) in x.items():
            self._db[k] = v


api_spec = [
        {
            "key": "apples",
            "options": [0, 1, 2, 3, 4],
            "probs": np.ones((5,)) / 5
        },
        {
            "key": "oranges",
            "options": [0, 1, 2, 3, 4, 5],
            "probs": np.ones((6,)) / 6
        }
    ]


input_wrapper = wrap.FunctionDictChoice(api_spec, random.PRNGKey(8))
post_generator = input_wrapper(dict)


def test_service():
    service = Service()

    input_a = post_generator()
    logging.info(f"Input a = {input_a}")
    service.post(input_a)
    num_oranges = service.get("oranges")
    num_apples = service.get("apples")
    logging.info(f"Number of apples in inventory: {num_apples}")
    logging.info(f"Number of oranges in inventory: {num_oranges}")

    # we could, in fact, be wrong about the number of things in our inventory
    # let's hedge our bets and instead return a random quantity that's centered
    # around the amount we think we have
    get_wrapper = wrap.ClassNonNegativeInteger(
        "get",
        cls=Service,
        key=random.PRNGKey(10),
        interpretation="increment"
    )

    with get_wrapper:
        num_oranges = service.get("oranges")
        num_apples = service.get("apples")
    
    logging.info(f"Number of apples could be {num_apples}")
    logging.info(f"Number of oranges could be {num_oranges}")

    # the entire process is traceable

    def post_and_get():
        input_a = post_generator()
        logging.info(f"Input a = {input_a}")
        service.post(input_a)
        with get_wrapper:
            num_oranges = service.get("oranges")
            num_apples = service.get("apples")
        logging.info(f"Number of apples could be {num_apples}")
        logging.info(f"Number of oranges could be {num_oranges}")

    log_joint, the_trace = numpyro.infer.util.log_density(
        post_and_get, tuple(), dict(), dict(),
    )
    logging.info(f"Traced post and get: {the_trace}")
    logging.info(f"Log prob = {log_joint}")

    # users can now see what downstream queries could reflect conditioned
    # on different data in POST, GET, etc.

    conditioned_post_and_get = numpyro.handlers.condition(
        post_and_get,
        data={"dict-likelihood-apples": np.array(3)}  # we guaranteed post 3 apples
    )
    log_joint, the_trace = numpyro.infer.util.log_density(
        conditioned_post_and_get, tuple(), dict(), dict(),
    )
    logging.info(f"Traced post and get (three appples guaranteed): {the_trace}")
    logging.info(f"Log prob = {log_joint}")
