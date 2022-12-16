Quickstart
==========
`sf_wrap` provides methods for lifting arbitrary python code to probabilistic programs.
That's an almost-true statement (we'll explain the lies in due time), and immediately
engenders two additional questions:

* Who cares?
* How does it work?

**Who cares?**: suppose that you are testing the robustness of business logic implementation to errors and
noisiness in an http endpoint. You could attempt to inject errors and noise yourself, or you could instead use `sf_wrap`
and model them via generative probabilistic programs while still using all of your existing object-oriented 
code. Here's your (really complex rite?) http service code:

.. code-block::

    class Service:

    def __init__(self,):
        self._db = defaultdict(int)

    def get(self, key: str):
        return self._db[key]

    def post(self, x: dict):
        for (k, v) in x.items():
            self._db[k] = v

Because you run a specialized fruit warehouse, your service should
only ever handle POST and GET related to apples and oranges. Additionally,
the trucks that service your warehouse are really small, so they only ever
send between zero and four apples and zero and 5 oranges at a time. 
(You should really be in a different business.)
You therefore have a strongly-typed API spec:

.. code-block::

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

Here you defined prior probabilities for seeing each of the options (which, 
in this case, are incoming quantities of fruit).
First, you make a stochastic function that generates posts for you:

.. code-block::

    input_wrapper = wrap.FunctionDictChoice(api_spec, random.PRNGKey(8))
    post_generator = input_wrapper(dict)

After you do that, it's time to test your API. You could just do something
basic, like

.. code-block::
    input_a = post_generator()
    logging.info(f"Input a = {input_a}")
    service.post(input_a)
    num_oranges = service.get("oranges")
    num_apples = service.get("apples")
    logging.info(f"Number of apples in inventory: {num_apples}")
    logging.info(f"Number of oranges in inventory: {num_oranges}")

This would let you "fuzz" the inputs to your business logic, but doesn't reflect the
unfortunate fact that the real world is messy. Maybe, between the time that you
wrote to the underlying database (using POST) and read from it (using GET), someone
else did the same thing, or maybe the database had a consistency issue. Instead,
you can use `sf_wrap` to temporarily change the interpretation of http methods
to return addressable, stochastic results:

.. code-block::

    get_wrapper = wrap.ClassNonNegativeInteger(
        "get",
        cls=Service,
        key=random.PRNGKey(10),
        interpretation="increment"
    )

You've wrapped the `get` method so that it now returns stochastic results. 
You then execute the same GET requests within a context manager that applies the
stochastic interpretation:

.. code-block::

    with get_wrapper:
        num_oranges = service.get("oranges")
        num_apples = service.get("apples")
    
    logging.info(f"Number of apples could be {num_apples}")
    logging.info(f"Number of oranges could be {num_oranges}")

In fact, you can use this whole process to complement hypothesis-based unit
testing: write a data flow program that you can trace (i.e., perform inference
about):

.. code-block::

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

When you execute this, you might see something like::

    2022-08-01 12:02:11 [    INFO] Traced post and get: OrderedDict([('dict-likelihood-apples', {'type': 'sample', 'name': 'dict-likelihood-apples', 'fn': <numpyro.distributions.discrete.CategoricalProbs object at 0x7fe398bc5b50>, 'args': (), 'kwargs': {'rng_key': DeviceArray([2881719429, 3889922613], dtype=uint32), 'sample_shape': ()}, 'value': DeviceArray(1, dtype=int32), 'scale': None, 'is_observed': False, 'intermediates': [], 'cond_indep_stack': [], 'infer': {}}), ('dict-likelihood-oranges', {'type': 'sample', 'name': 'dict-likelihood-oranges', 'fn': <numpyro.distributions.discrete.CategoricalProbs object at 0x7fe398bc5c70>, 'args': (), 'kwargs': {'rng_key': DeviceArray([2881719429, 3889922613], dtype=uint32), 'sample_shape': ()}, 'value': DeviceArray(1, dtype=int32), 'scale': None, 'is_observed': False, 'intermediates': [], 'cond_indep_stack': [], 'infer': {}}), ('Service-ncls[0]-get-likelihood-0', {'type': 'sample', 'name': 'Service-ncls[0]-get-likelihood-0', 'fn': <numpyro.distributions.discrete.Poisson object at 0x7fe398bdf9a0>, 'args': (), 'kwargs': {'rng_key': DeviceArray([3092750537, 3312654149], dtype=uint32), 'sample_shape': ()}, 'value': DeviceArray(3, dtype=int32), 'scale': None, 'is_observed': False, 'intermediates': [], 'cond_indep_stack': [], 'infer': {}}), ('Service-ncls[0]-get-likelihood-1', {'type': 'sample', 'name': 'Service-ncls[0]-get-likelihood-1', 'fn': <numpyro.distributions.discrete.Poisson object at 0x7fe398bdfbb0>, 'args': (), 'kwargs': {'rng_key': DeviceArray([ 523915524, 3328915131], dtype=uint32), 'sample_shape': ()}, 'value': DeviceArray(4, dtype=int32), 'scale': None, 'is_observed': False, 'intermediates': [], 'cond_indep_stack': [], 'infer': {}})]) (test_integration_service.py:86)
    2022-08-01 12:02:11 [    INFO] Log prob = -7.518982410430908 (test_integration_service.py:87)



**How does it work?**: it is easy to overwrite the definition of a function or an object method
in python. `sf_wrap` overwrites functions and object methods using either higher-order
functions/decorators (for functions) and decorators or context managers (for object methods).
It exposes two abstract base classes, `SFClassDecorator` and `SFFunctionDecorator`, that
use `numpyro` to implement probabilistic semantics. They require the user to
implement a single concrete method,
`sample(self, address: str, value: any, rng_key: numpyro.random.PRNGKey)`,
that specifies the probability law that the concrete subclass should implement.
`sf_wrap` implements some useful concrete subclasses, including

* `ClassTruncatedReal`: :math:`x \sim \mathrm{TruncatedNormal}(\mathrm{value}, \sigma, l, h)`, with support :math:`x \in [l, h]`, `value :: float`
* `ClassReal`: :math:`x \sim \mathrm{Normal}(\mathrm{value}, \sigma)`, `value :: float`
* `ClassNonNegativeInteger`: :math:`n \sim \mathrm{Poisson}(\mathrm{value + 1})`, `value :: int`, throwing runtime `TypeError` if :math:`\mathrm{value} < 0`
* `FunctionTruncatedReal`: :math:`x \sim \mathrm{TruncatedNormal}(\mathrm{value}, \sigma, l, h)`, with support :math:`x \in [l, h]`, `value :: float`
* `FunctionDictChoice`: :math:`m[x] \sim \mathrm{Categorical}(\rho)`, with support :math:`x \in \{0, 1, ..., N\}` and where `m :: dict[int, any]` and :math:`\rho \in \mathrm{simplex}(N)`.

:math:`\mathrm{value}` represents the original return value of the non-lifted method or function.
For example, 

.. code-block::
    
    f = lambda: 3.0
    f_wrap = wrap.FunctionTruncatedReal(0.0, 5.0, scale=1.0)(f)

defines a random variable :math:`x \sim \mathrm{TruncatedNormal}(3, 1, 0, 5)` which is generated by `x = f_wrap()`.
Multiple methods of the same class can be lifted. For example, supposing that you have class `A` with methods `f` and
`g` each with signature `A.{f,g} :: self -> float`, you can write

.. code-block::

    from sf_wrap import wrap

    a = A()

    f_wrap = wrap.ClassReal("f", cls=A, scale=1.0,)
    g_wrap = wrap.ClassReal("g", cls=A, scale=2.0,)

    with f_wrap, g_wrap:
        my_f_res = a.f()
        my_g_res = a.g()

