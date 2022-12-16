# `sf_wrap`

`sf_wrap` lifts arbitrary object-oriented python code to probabilistic programs expressed in [numpyro](https://github.com/pyro-ppl/numpyro) via automatic method overwriting. It defines function decorators that implement stochastic functions and class decorators that augment the definitions of object methods. It is simple, self-contained, and permissively-licensed, with all functionality in a [single file](./sf_wrap/wrap.py) that you could copy into your project.

## Usage example

Here's a silly class:

```
class A:
    def f(self, x: float):
        return x

    def g(self, x: float):
        return self.f(x) ** 2.0
```

There are two ways to use `sf_wrap` functionality: as context managers with existing code; or as class decorators when classes are defined. The mechanisms are equivalent for many purposes; which mechanism is used should be dictated by design of the larger software system.

### Context managers

```
my_a = A()
x = 3.0
my_a_f_val = my_a.f(x)

with wrap.ClassTruncatedReal("f", low=-4, high=4, scale=0.1, cls=A,):
    my_stoch_a_f_val = my_a.f(x)

    with wrap.ClassTruncatedReal("g", low=-16, high=16, scale=0.1, cls=A):
        my_stoch_a_g_val = my_a.g(x)
```

`my_a_f_val == 3.0`, while `my_stoch_a_f_val` will deviate slightly from this value, as `my_stoch_a_g_val` will deviate slightly from `my_stoch_a_f_val ** 2`.

### Class decorators

```
from sf_wrap import wrap


@wrap.ClassTruncatedReal("f", low=-3.0, high=3.0, scale=1.0,)
class RandomA(A): pass


@wrap.ClassTruncatedReal("g", low=-3.0, high=3.0, scale=1.0,)
@wrap.ClassTruncatedReal("f", low=-3.0, high=3.0, scale=1.0,)
class ReallyRandomA(A): pass
```

+ Calling the `f` method of a `RandomA` object now returns the result of a truncated normal distribution with location parameter
equal to the return value of the deterministic function `A.f(self, x)`. 
+ Calling the `g` method of a `ReallyRandomA` object now returns the result of a truncated normal distribution with location parameter equal to the return value of the deterministic function `A.g(self, y)` where `y` is...
    + ...the result of a truncated normal distribution with location parameter equal to the the return value of the deterministic function `A.f(self, x)`.

And so on and so forth. For more, please see the documentation.

## Installation and testing

Installation is standard:
```
conda create --name my_env_name python=3.9
conda activate my_env_name
/stuff/my_env_name/bin/python -m pip install -r requirements.txt
/stuff/my_env_name/bin/python -m pip install .
```

Testing is standard:
```
/stuff/my_env_name/bin/python -m pytest --cov=sf_wrap
```

`sf_wrap` has been installed and tested on only MacOS (Catalina) and Ubuntu.
It should work on any posix-compliant system, but YMMV.
It has not been installed or tested on Windows.

## Other information

`sf_wrap` is licensed under version 3 of the GNU Lesser General Public License (LGPL v3). 
Copyright Charles River Analytics, Inc., 2022 - present.  

This material is based upon work supported by the Naval Information Warfare Center (NIWC) Atlantic under Contract No. N653622C8011. 
Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the
NIWC Atlantic. 
Approved for public release, distribution is unlimited.