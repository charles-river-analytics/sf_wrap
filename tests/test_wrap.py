"""
Author: David Rushing Dewhurst, ddewhurst@cra.com
Copyright Charles River Analytics Inc., 2022 - present
License: LGPL v3
"""

import logging

from jax import random
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from sf_wrap import wrap


class A:
    def f(self, x):
        return x

    def g(self, x):
        return self.f(x) ** 2.0


@wrap.ClassTruncatedReal("f", low=-3.0, high=3.0, scale=1.0,)
class RandomA(A):
    pass


@wrap.ClassTruncatedReal("g", low=-3.0, high=3.0, scale=1.0,)
@wrap.ClassTruncatedReal("f", low=-3.0, high=3.0, scale=1.0,)
class ReallyRandomA(A):
    pass


@wrap.ClassDeterministic("f")
class TrackedA(A):
    pass


def test_deterministic():
    a = TrackedA()
    x = 10.0

    log_joint, the_trace = numpyro.infer.util.log_density(
        a.f, (x,), dict(), dict(),
    )
    logging.info(f"Got trace: {the_trace}")
    logging.info(f"Got log joint probability: {log_joint}")


def test_basic():

    a = A()
    a_rand = RandomA()
    a_really_rand = ReallyRandomA()

    x = 1.0
    res_det = a.f(x)

    log_joint, the_trace = numpyro.infer.util.log_density(
        a_rand.f, (x,), dict(), dict(),
    )
    logging.info(f"For RandomA, computed trace: {the_trace}")

    log_joint, the_trace = numpyro.infer.util.log_density(
        a_really_rand.g, (x,), dict(), dict(),
    )
    logging.info(f"For ReallyRandomA, computed trace: {the_trace}")

    logging.info(a.f(x))


def test_unique_addresses():
    a = A()
    a_rand = RandomA()
    another_rand = RandomA()

    x = 1.0
    res_det = a.f(x)

    log_joint, the_trace = numpyro.infer.util.log_density(
        a_rand.f, (x,), dict(), dict(),
    )
    another_log_joint, another_trace = numpyro.infer.util.log_density(
        another_rand.f, (x,), dict(), dict(),
    )

    logging.info(f"First trace: {the_trace}")
    logging.info(f"Second trace: {another_trace}")
    

def my_function(x):
    return x / 12


def test_function_wrap():
    wrapper = wrap.FunctionTruncatedReal(0.0, 5.0, scale=0.5)
    sf = wrapper(my_function)

    x = 12.0
    assert my_function(x) == 1.0
    log_joint, the_trace = numpyro.infer.util.log_density(
        sf, (x,), dict(), dict(),
    )
    logging.info(f"Function trace: {the_trace}")
    logging.info(f"The log joint: {log_joint}")


def test_lambda_wrap():
    wrapper = wrap.FunctionTruncatedReal(0.0, 5.0, scale=0.5)
    sf1 = wrapper(lambda x: x + 4)
    sf2 = wrapper(lambda x: x - 4)
    x = 4
    log_joint, the_trace = numpyro.infer.util.log_density(
        sf1, (x,), dict(), dict(),
    )
    another_log_joint, another_trace = numpyro.infer.util.log_density(
        sf2, (x,), dict(), dict(),
    )
    logging.info(f"trace lambda 1: {the_trace}")
    logging.info(f"trace lambda 2: {another_trace}")


def test_dict_choice():

    # define a dependently-typed dictionary
    spec = [
        {
            "key": "apples",
            "options": [1, 2, 3, 4],
            "probs": np.ones((4,)) / 4
        },
        {
            "key": "oranges",
            "options": [2, 4, 6, 8, 10],
            "probs": np.ones((5,)) / 5
        }
    ]
    logging.info(f"Using data spec: {spec}")

    # the support of this sf is the class of dictionaries that satisfy
    # the above type `spec`
    wrapper = wrap.FunctionDictChoice(spec, random.PRNGKey(20220720))

    # hack to define a unconditional distribution over dicts with
    # a precisely-defined key/value set
    empty_sf = wrapper(dict)
    test = empty_sf()
    logging.info(f"Using empty dict, generated {test}")

    log_joint, the_trace = numpyro.infer.util.log_density(
        empty_sf, tuple(), dict(), dict(),
    )
    logging.info(f"Traced empty sf: {the_trace}")
    logging.info(f"Log prob = {log_joint}")

    # execute on a full dict -- using f, we'll first modify
    # the dict, then sample
    def f(x: dict):
        x["apples"] = 1
        x["oranges"] += 2
        return x

    full_dict = {"apples": 2, "oranges": 2}
    full_sf = wrapper(f)
    test = full_sf(full_dict)
    logging.info(f"Using full dict, generated {test}")
    
    log_joint, the_trace = numpyro.infer.util.log_density(
        full_sf, (full_dict,), dict(), dict(),
    )
    logging.info(f"Traced full sf: {the_trace}")
    logging.info(f"Log prob = {log_joint}")

    # define a partially-conditioned distribution over dicts
    def g(x: dict):
        x["oranges"] += 6
        return x

    partial_dict = {"oranges": 2}
    partial_sf = wrapper(g)
    test = partial_sf(partial_dict)
    logging.info(f"Using partial dict, generated {test}")

    # Be careful! Does not check that functions are pure
    # (e.g., here modified state of partial_dict)
    partial_dict = {"oranges": 2}
    log_joint, the_trace = numpyro.infer.util.log_density(
        partial_sf, (partial_dict,), dict(), dict(),
    )
    logging.info(f"Traced partial sf: {the_trace}")
    logging.info(f"Log prob = {log_joint}")

    with pytest.raises(TypeError):
        partial_dict = {"oranges": 12}
        partial_sf(partial_dict)


def test_context_manager():
    
    class A:
        def __init__(self, value):
            self.value = value 
        def f(self, x):
            return x - self.value

    value = 3
    a = A(value)
    x = 8
    a_f_ret = a.f(x)
    logging.info(f"Without context manager, returned {a_f_ret}")

    with wrap.ClassTruncatedReal("f", low=-10.0, high=10.0, scale=1.0, cls=A):
        # you do not have to create it inside the context manager
        a_f_ret = a.f(x)
        log_joint, the_trace = numpyro.infer.util.log_density(
            a.f, (x,), dict(), dict(),
        )
        # ... though creation works just fine inside as well
        a2 = A(value + 1)
        log_joint2, the_trace2 = numpyro.infer.util.log_density(
            a2.f, (x,), dict(), dict(),
        )

    logging.info(f"With context manager, returned {a_f_ret}")
    logging.info(f"Traced a.f in the context manager: {the_trace}")
    logging.info(f"Log prob = {log_joint}")
    logging.info(f"Traced a2.f in the context manager: {the_trace2}")
    logging.info(f"Log prob = {log_joint2}")

    a_f_ret = a.f(x)
    logging.info(f"Without context manager (again!), returned {a_f_ret}")

    wrapper = wrap.ClassTruncatedReal(
        "f",
        low=-10.0,
        high=10.0,
        scale=2.0,
        cls=A,
        interpretation="increment"
    )
    
    def calls_twice():
        with wrapper:
            # normally, this would complain about
            # sampling twice at the same addres, but we've used
            # the increment interpretation...
            return a.f(x) + a.f(x)
    
    log_joint, the_trace = numpyro.infer.util.log_density(
        calls_twice, tuple(), dict(), dict(),
    )
    logging.info(f"With context manager and increment interpretation, trace is {the_trace}")


def test_nested_context_manager():   
    # demonstrate nested context managers
    my_a = A()
    x = 3.0
    my_a_f_val = my_a.f(x)
    logging.info(f"Created value {my_a_f_val} from {my_a} f method")

    with wrap.ClassTruncatedReal("f", low=-4, high=4, scale=0.1, cls=A,):
        logging.info(f"In stochastic f context")
        my_stoch_a_f_val = my_a.f(x)
        logging.info(f"Created value {my_stoch_a_f_val} from {my_a} f method")
        
        with wrap.ClassTruncatedReal("g", low=-16, high=16, scale=0.1, cls=A):
            logging.info(f"In stochastic g context")
            my_stoch_a_g_val = my_a.g(x)
            logging.info(f"Created value {my_stoch_a_g_val} from {my_a} g method")
            log_joint, the_trace = numpyro.infer.util.log_density(
                my_a.g, (x,), dict(), dict(),
            )
            logging.info(f"Traced my_a.g in stochastic g context")
            logging.info(f"Trace: {the_trace}, logprob = {log_joint}")
            logging.info(f"Leaving stochastic g context")
        
        logging.info(f"Leaving stochastic f context")

    my_a_g_val = my_a.g(x)
    logging.info(f"Created value {my_a_g_val} from {my_a} g method")


class AA:
    def __init__(self, value):
        self.value = value 
    def f(self, x):
        return x - self.value
    def g(self, x):
        return int(self.f(x) + 1)


class BB:
    def __init__(self, a: A):
        self.a = a
    def h(self, x):
        return 2 * self.a.g(x)


def test_real():
    a_real_wrapper = wrap.ClassReal("f", 1.0, key=random.PRNGKey(2), cls=AA)
    a_int_wrapper = wrap.ClassNonNegativeInteger("g", key=random.PRNGKey(3), cls=AA)
    b_int_wrapper = wrap.ClassNonNegativeInteger("h", key=random.PRNGKey(4), cls=BB)

    a = AA(2)
    b = BB(a)

    with a_int_wrapper, a_real_wrapper, b_int_wrapper:
        log_joint, the_trace = numpyro.infer.util.log_density(
            b.h, (10,), dict(), dict(),
        )
    logging.info(f"Traced execution of b.h, wrapping parent class methods too: {the_trace}")
    logging.info(f"Log probability = {log_joint}")

    # note the difference, e.g., just wrapping the B.h method
    with b_int_wrapper:
        log_joint, the_trace = numpyro.infer.util.log_density(
            b.h, (10,), dict(), dict(),
        )
    logging.info(f"Traced execution of b.h *onlyy* wrapping b.h: {the_trace}")
    logging.info(f"Log probability = {log_joint}")
