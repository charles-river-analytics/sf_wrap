"""
Author: David Rushing Dewhurst, ddewhurst@cra.com
Copyright Charles River Analytics Inc., 2022 - present
License: LGPL v3
"""

import abc
from collections import defaultdict
from typing import Callable, Literal, Optional

import jax.numpy as np
from jax import random
import numpyro
import numpyro.distributions as dist


class ClassDecorator(abc.ABC):
    """
    Class decorator interface.

    Must supply a __call__ method that takes one argument of type `type`.
    """
        
    @abc.abstractmethod
    def __call__(self, c: type):
        """
        Abstract method to be overridden.

        :param c: a type to wrap
        :type c: type
        """
        ...


class SFDecorator(abc.ABC):
    """
    An abstract stochastic class decorator that defines a likelihood
    function for a `Node` subclass.
    """

    def __init__(self, key: random.PRNGKey=random.PRNGKey(2022)):
        self.rng_key = key
        self._n_called = defaultdict(int)

    @abc.abstractmethod
    def sample(self, address, value, rng_key):
        """
        Abstract method to be overridden.
        
        :param address: the address in the trace.
        :type address: hashable
        :param value: value to use in the sample statement.
        :type value: any, though it is up to the implementer to ensure
            that the likelihood function chosen makes sense for the value.
        :param rng_key: a numpyro PRNG key
        :type rng_key: numpyro.random.PRNGKey
        """
        ...


class FunctionDecorator(abc.ABC):
    """
    Function decorator interface.

    Must supply a __call__ method that takes one argument of type `Callable`.
    TODO: make this more explicitly a function, not any other callable
    """

    @abc.abstractmethod
    def __call__(self, f: Callable):
        """
        Abstract method to be overridden.

        :param f: a function to wrap
        :type f: function
        """
        ...


class SFFunctionDecorator(FunctionDecorator, SFDecorator):
    """
    Implements an interface for a decorator that lifts an ordinary python function
    to a stochastic function.

    :param key: key for the numpyro PRNG
    :type key: random.PRNGKey
    """

    _lambda_ix = 0

    def __init__(
        self,
        key: random.PRNGKey=random.PRNGKey(2022)
    ):
        SFDecorator.__init__(self, key,)
        FunctionDecorator.__init__(self,)

    def _get_address(self, f,):
        f_name = f.__name__ 
        if f_name == "<lambda>":
            f_name += "-" + str(type(self)._lambda_ix)
            type(self)._lambda_ix += 1
        return f"{f_name}-likelihood"

    def __call__(self, f: Callable):
        """
        Rewrites the wrapped function to use
        a likelihood function.

        :param f: the function to wrap
        :type f: Callable, intended to represent a function only
        """

        def new_f(*args, **kwargs):
            value = f(*args, **kwargs)
            self.rng_key, rng_key = random.split(self.rng_key)
            return self.sample(self._get_address(f,), value, rng_key)

        return new_f 


class SFClassDecorator(SFDecorator, ClassDecorator):
    """
    Implements an interface for a decorator that lifts a method in an ordinary
    python class to a stochastic method.

    Also, defines the interface for a context manager that enables
    stochastic interpretation of deterministic, pre-defined code.

    :param method: the name of the method to lift
    :type method: str
    :param key: key for the numpyro PRNG
    :type key: random.PRNGKey
    :param cls: if passed, the class whose `method` behavior will be overwritten if
        an instance of this `SFClassDecorator` is used as a context manager.
    :type cls: Optional[type]
    :param interpretation: whether to interpret sample statements as strict or increment.
        If strict, will follow normal (num)pyro behavior if attempt to sample multiple
        times as the same sample site.
        If increment, will autoincrement sample number, appending the sample number to the
        end of the address.
    :type interpretation: Literal["strict", "increment"]
    """

    def __init__(
        self,
        method: str,
        key: random.PRNGKey=random.PRNGKey(2022),
        cls: Optional[type]=None,
        interpretation: Literal["strict", "increment"]="strict",
    ):
        SFDecorator.__init__(self, key,)
        ClassDecorator.__init__(self,)
        self._to_overwrite = method

        if cls is not None:
            tcls = type(cls)
            if (tcls is not type) and (tcls is not abc.ABCMeta):
                raise TypeError("Only pass objects of type `type` to cls keyword")
            else:
                self._c = cls
        self._interpretation = interpretation
    
    def _set_ctr(self, c: type):
        if not hasattr(c, "__cls_ctr__"):
            c.__cls_ctr__ = 0

    def _inc_ctr(self, c: type):
        c.__cls_ctr__ += 1

    def _get_address(self, c, wrapped_self, interpretation="strict",):
        if interpretation == "strict":
            return f"{c.__name__}-ncls[{wrapped_self.__obj_id__}]-{self._to_overwrite}-likelihood"
        else:
            addr = f"{c.__name__}-ncls[{wrapped_self.__obj_id__}]-{self._to_overwrite}-likelihood-{self._n_called[(str(c), self._to_overwrite)]}"
            self._n_called[(str(c), self._to_overwrite)] += 1
            return addr

    def _make_new_init(self, c: type, old_init: Callable,):
        def new_init(wrapped_self, *args, **kwargs):
            wrapped_self.__obj_id__ = c.__cls_ctr__
            value = old_init(wrapped_self, *args, **kwargs)
            self._inc_ctr(c,)
            return value
        return new_init

    def _make_new_f(self, c: type, old_f: Callable,):
        def new_f(wrapped_self, *args, **kwargs):
            if not hasattr(wrapped_self, "__obj_id__"):
                wrapped_self.__obj_id__ = c.__cls_ctr__
                self._inc_ctr(c,)
            address = self._get_address(c, wrapped_self, interpretation=self._interpretation)
            value = old_f(wrapped_self, *args, **kwargs)
            self.rng_key, rng_key = random.split(self.rng_key)
            return self.sample(
                address,
                value,
                rng_key
            )
        return new_f

    def __enter__(self,):
        self._old_f = getattr(self._c, self._to_overwrite)
        self._old_init = getattr(self._c, "__init__")
        self.__call__(self._c, return_=False,)

    def __exit__(self, type, value, traceback,):
        setattr(self._c, self._to_overwrite, self._old_f)
        setattr(self._c, "__init__", self._old_init)
        self._n_called[(str(self._c), self._to_overwrite)] = 0

    def __call__(self, c: type, return_=True,):
        """
        Rewrites the specified method of the wrapped type to use
        a likelihood function.

        :param c: the class to wrap
        :type c: type
        """
        old_f = getattr(c, self._to_overwrite,)
        old_init = getattr(c, "__init__")
        self._set_ctr(c,)

        new_init = self._make_new_init(c, old_init)
        new_f = self._make_new_f(c, old_f)

        setattr(c, self._to_overwrite, new_f)
        setattr(c, "__init__", new_init)
        if return_:
            return c


class ClassTruncatedReal(SFClassDecorator):
    """
    A stochastic class decorator that implements a truncated normal likelihood.

    The likelihood is constrained between the low and high values. Its loc parameter
    is equal to the passed value, and its scale parameter is set by the user.

    :param method: the method to overwrite
    :param low: the lower bound of the support
    :param high: the upper bound of the support
    :param scale: the scale of the distribution
    :param key: the prng key, used by numpyro to sample when inference is not
        being performed
    :param cls: if passed, the class whose `method` behavior will be overwritten if
        an instance of this `SFClassDecorator` is used as a context manager.
    :type cls: Optional[type]
    :param interpretation: whether to interpret sample statements as strict or increment.
        If strict, will follow normal (num)pyro behavior if attempt to sample multiple
        times as the same sample site.
        If increment, will autoincrement sample number, appending the sample number to the
        end of the address.
    :type interpretation: Literal["strict", "increment"]
    """

    def __init__(
        self,
        method: str,
        low: float,
        high: float,
        scale: float,
        key: random.PRNGKey=random.PRNGKey(2022),
        cls=None,
        interpretation: Literal["strict", "increment"]="strict",
    ):
        super().__init__(method, key, cls=cls, interpretation=interpretation,)
        self.low = low
        self.high = high
        self.scale = scale

    def sample(self, address, value, rng_key):
        """
        Samples from the truncated normal likelihood function.

        :param address: the address of the sampled value to record in the trace
        :type addres: str
        :param value: the value to use as the location parameter
        :type value: float
        :param rng_key: key for the numpyro PRNG
        :type rng_key: random.PRNGKey
        """
        return numpyro.sample(
            address,
            dist.TruncatedNormal(value, self.scale, low=self.low, high=self.high),
            rng_key=rng_key,
        ).item()


class ClassReal(SFClassDecorator):
    """
    A stochastic class decorator that implements a normal likelihood.

    Its loc parameter
    is equal to the passed value, and its scale parameter is set by the user.

    :param method: the method to overwrite
    :param scale: the scale of the distribution
    :param key: the prng key, used by numpyro to sample when inference is not
        being performed
    :param cls: if passed, the class whose `method` behavior will be overwritten if
        an instance of this `SFClassDecorator` is used as a context manager.
    :type cls: Optional[type]
    :param interpretation: whether to interpret sample statements as strict or increment.
        If strict, will follow normal (num)pyro behavior if attempt to sample multiple
        times as the same sample site.
        If increment, will autoincrement sample number, appending the sample number to the
        end of the address.
    :type interpretation: Literal["strict", "increment"]
    """

    def __init__(
        self,
        method: str,
        scale: float,
        key: random.PRNGKey=random.PRNGKey(2022),
        cls=None,
        interpretation: Literal["strict", "increment"]="strict",
    ):
        super().__init__(method, key, cls=cls, interpretation=interpretation,)
        self.scale = scale

    def sample(self, address, value, rng_key):
        """
        Samples from the normal likelihood function.

        :param address: the address of the sampled value to record in the trace
        :type addres: str
        :param value: the value to use as the location parameter
        :type value: float
        :param rng_key: key for the numpyro PRNG
        :type rng_key: random.PRNGKey
        """
        return numpyro.sample(
            address,
            dist.Normal(value, self.scale,),
            rng_key=rng_key,
        ).item()


class ClassNonNegativeInteger(SFClassDecorator):
    """
    A stochastic class decorator that implements a poisson likelihood.

    The likelihood is constrained between zero and positive integer infinity.
    The location parameter is set equal to the deterministic return value
    of the underlying method code plus one.

    :param method: the method to overwrite
    :param key: the prng key, used by numpyro to sample when inference is not
        being performed
    :param cls: if passed, the class whose `method` behavior will be overwritten if
        an instance of this `SFClassDecorator` is used as a context manager.
    :type cls: Optional[type]
    :param interpretation: whether to interpret sample statements as strict or increment.
        If strict, will follow normal (num)pyro behavior if attempt to sample multiple
        times as the same sample site.
        If increment, will autoincrement sample number, appending the sample number to the
        end of the address.
    :type interpretation: Literal["strict", "increment"]
    """

    def sample(self, address, value, rng_key):
        """
        Samples from the poisson likelihood function.

        :param address: the address of the sampled value to record in the trace
        :type addres: str
        :param value: the value to use as the location parameter
        :type value: float
        :param rng_key: key for the numpyro PRNG
        :type rng_key: random.PRNGKey
        """
        if value < 0:
            raise TypeError("Can't sample from poisson with negative location!")
        return numpyro.sample(
            address,
            dist.Poisson(value + 1),
            rng_key=rng_key,
        ).item()


class ClassDeterministic(SFClassDecorator):
    """
    A class decorator that simply records the value in a trace.

    :param method: the method to overwrite
    """

    def sample(self, address, value, rng_key):
        """
        Records the `value` in the trace at `address`. `rng_key` is not used
        and present only for API compliance.
        """
        return numpyro.deterministic(address, value)


class FunctionTruncatedReal(SFFunctionDecorator):
    """
    A stochastic class decorator that implements a truncated normal likelihood.

    The likelihood is constrained between the low and high values. Its loc parameter
    is equal to the passed value, and its scale parameter is set by the user.

    :param low: the lower bound of the support
    :param high: the upper bound of the support
    :param scale: the scale of the distribution
    :param key: the prng key, used by numpyro to sample when inference is not
        being performed
    """

    def __init__(
        self,
        low: float,
        high: float,
        scale: float,
        key: random.PRNGKey=random.PRNGKey(2022)
    ):
        super().__init__(key,)
        self.low = low
        self.high = high
        self.scale = scale

    def sample(self, address, value, rng_key):
        """
        Samples from the truncated normal likelihood function.

        :param address: the address of the sampled value to record in the trace
        :type addres: str
        :param value: the value to use as the location parameter
        :type value: float
        :param rng_key: key for the numpyro PRNG
        :type rng_key: random.PRNGKey
        """
        return numpyro.sample(
            address,
            dist.TruncatedNormal(value, self.scale, low=self.low, high=self.high),
            rng_key=rng_key,
        ).item()


def _check_mapping(mapping: list[dict]) -> bool:
    for d in mapping:
        if "key" not in d.keys():
            return False
        if "options" not in d.keys():
            return False
        if "probs" not in d.keys():
            return False
        if type(d["options"]) is not list:
            return False
        if type(d["probs"]) is not np.ndarray:
            return False
        if len(d["options"]) != len(d["probs"]):
            return False
    return True


class FunctionDictChoice(SFFunctionDecorator):
    """
    A function wrapper that defines a discrete probability distribution over
    a strongly-typed dictionary.

    The distribution is supported over an enumerated set of keys, each of which
    maps to an enumerated set of values.

    The `mapping` should have the structure

        [
            {
                "key": <str>,
                "options": [<any>],
                "probs": np.ndarray
            }
        ]


    :param mapping: a mapping describing the keys, allowed values, and probabilities
        of each allowed value
    :type mapping: list[dict]
    """

    def __init__(
        self,
        mapping: list[dict],
        key: random.PRNGKey=random.PRNGKey(2022)
    ):
        super().__init__(key,)
        _check_mapping(mapping)
        self.mapping = mapping

    def sample_enter(self, address: str, value: dict, entry: dict, rng_key,):
        """
        Sets ("enter"s) a random value into a dictionary. 

        :param address: the beginning of the address at which to enter the value
        :type address: str
        :param value: dictionary into which to enter the randomly-chosen element.
            A copy of the dictionary will be returned.
        :type value: dict
        :param entry: a dictionary describing what is allowed for the value, and
            the associated probabilities.
        :type entry: dict
        :param rng_key: key for the numpyro PRNG
        
        :returns: a copy of the `value` with an updated entry
        :rtype: dict
        """
        address += f"-{entry['key']}"
        ix_to_opt = {i: v for (i, v) in enumerate(entry['options'])}
        sample_ix = numpyro.sample(
            address,
            dist.Categorical(entry['probs']),
            rng_key=rng_key,
        ).item()
        value[entry['key']] = ix_to_opt[sample_ix]
        return value

    def sample_score(self, address: str, value: dict, entry: dict, rng_key, obs,):
        """
        Scores a value against the value in a dictionary.

        :param address: the beginning of the address at which to enter the value
        :type address: str
        :param value: dictionary into which to enter the randomly-chosen element.
            A copy of the dictionary will be returned.
        :type value: dict
        :param entry: a dictionary describing what is allowed for the value, and
            the associated probabilities.
        :type entry: dict
        :param rng_key: key for the numpyro PRNG
        :param obs: the observed value. Throws `KeyError` if `obs` is not in the
            support defined by `mapping` in the constructor.
        """
        address += f"-{entry['key']}"
        opt_to_ix = {v: i for (i, v) in enumerate(entry['options'])}
        numpyro.sample(
            address,
            dist.Categorical(entry['probs']),
            rng_key=rng_key,
            obs=opt_to_ix[obs]
        )

    def sample(self, address, value: dict, rng_key):
        """
        Samples a value into the dict, or scores the value against the value in
        the dict, depending on if there is a value in the dict or not. 

        See the documentation of `sample_enter` or `sample_score` for more
        detail.
        """
        v_keys = list(value.keys())
        for entry in self.mapping:
            try:
                if entry["key"] not in v_keys:
                    value = self.sample_enter(address, value, entry, rng_key,)
                else:
                    self.sample_score(address, value, entry, rng_key, value[entry["key"]])
            except KeyError as e:
                raise TypeError(f"type specified by {entry} conflicts with {value}")
        return value
