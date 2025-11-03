"""Models of the foregrounds."""

import attrs
import numpy as np
from cached_property import cached_property
from yabf import Component, Parameter


def damped_oscillation_model(freqs, period, damping_index, amp_sin, amp_cos):
    """Damped sinusoid systematic model from Sims+2021."""
    phase = 2 * np.pi * (freqs / period)
    f_c = freqs[0] + (freqs[-1] - freqs[0]) / 2
    return (freqs / f_c) ** damping_index * (
        amp_sin * np.sin(phase) + amp_cos * np.cos(phase)
    )


@attrs.define(frozen=True)
class Foreground(Component):
    """Base class for all foreground models.

    Don't use this directly.
    """

    freqs: np.ndarray = attrs.field(kw_only=True, eq=attrs.cmp_using(eq=np.array_equal))
    nuc = attrs.field(default=75.0, kw_only=True, converter=float)

    @cached_property
    def provides(self):
        """The foreground model provides a spectrum."""
        return [f"{self.name}_spectrum"]

    @cached_property
    def f(self):
        """The frequency array normalized by the reference frequency."""
        return self.freqs / self.nuc

    def calculate(self, ctx=None, **params):
        """Calculate the foreground model."""
        return self.model(**params)

    def model(self, **params):
        """Abstract method specifying that subclasses must implement a model."""


@attrs.define(frozen=True)
class Tcmb(Component):
    """A simple component adding the CMB temperature to a set of foregrounds."""

    T = attrs.field(default=2.7255, kw_only=True, converter=float)

    @cached_property
    def provides(self):
        """The Tcmb model provides a scalar value."""
        return [f"{self.name}_scalar"]

    def calculate(self, ctx=None, **params):
        """Calculate the Tcmb model."""
        return self.T


class _PhysicalBase(Foreground):
    base_parameters = [
        Parameter("b0", 1750, min=0, max=1e5, latex=r"b_0 [K]"),
        Parameter("b1", 0, latex=r"b_1 [K]"),
        Parameter("b2", 0, latex=r"b_2 [K]"),
        Parameter("b3", 0, latex=r"b_3 [K]"),
        Parameter("ion_spec_index", -2, min=-3, max=-1, latex=r"\alpha_{\rm ion}"),
    ]

    def model(self, **p):
        """Compute the PhysicalBase model."""
        b = [p[f"b{i}"] for i in range(4)]
        alpha = p["ion_spec_index"]

        x = np.exp(-b[3] * self.f**alpha)
        return b[0] * self.f ** (b[1] + b[2] * np.log(self.f) - 2.5) * x, x


class PhysicalHills(_PhysicalBase):
    """Eq 6. from Hills et al."""

    base_parameters = [
        *_PhysicalBase.base_parameters,
        Parameter("electron_temperature", 1000, min=0, max=5000, latex=r"T_e [K]"),
    ]

    def model(self, electron_temperature, **p):
        """Compute the PhysicalHills model."""
        first_term, x = super().model(**p)
        return first_term + electron_temperature * (1 - x)


class PhysicalSmallIonDepth(_PhysicalBase):
    """Eq. 7 from Hills et al."""

    base_parameters = [
        *_PhysicalBase.base_parameters,
        Parameter("b4", 0, latex=r"b_4 [K]"),
    ]

    def model(self, **p):
        """Comput the PhysicalSmallIonDepth model."""
        first_term, _ = super().model(**p)
        b4 = p["b4"]
        return first_term + b4 / self.f**2

    # Possible derived quantities
    def electron_temperature(self, ctx, **params):
        """Approximate value of Te in the small-ion-depth limit."""
        return params["b4"] / params["b3"]


class PhysicalLin(Foreground):
    """Eq. 8 from Hills et al."""

    base_parameters = [Parameter("p0", fiducial=1750, latex=r"p_0")] + [
        Parameter(f"p{i}", fiducial=0, latex=rf"p_{i}") for i in range(1, 5)
    ]

    def model(self, **p):
        """Compute the PhysicalLin model."""
        p = [p[f"p{i}"] for i in range(5)]

        return (
            self.f**-2.5 * (p[0] + np.log(self.f) * (p[1] + p[2] * np.log(self.f)))
            + p[3] * self.f**-4.5
            + p[4] * self.f**-2
        )

    # Possible derived parameters
    def b0(self, ctx, **p):
        """The corresponding b0 from PhysicalHills."""
        return p["p0"]

    def b1(self, ctx, **p):
        """The corresponding b1 from PhysicalHills."""
        return p["p1"] / p["p0"]

    def b2(self, ctx, **p):
        """The corresponding b2 from PhysicalHills."""
        return p["p2"] / p["p0"] - self.b1(ctx, **p) ** 2 / 2

    def b3(self, ctx, **p):
        """The corresponding b3 from PhysicalHills."""
        return -p["p3"] / p["p0"]

    def b4(self, ctx, **p):
        """The corresponding b4 from PhysicalHills."""
        return p["p4"]


@attrs.define
class IonContrib(Foreground):
    """Absorption and emission due to the ionosphere."""

    base_parameters = [
        Parameter("absorption", fiducial=0, latex=r"\tau"),
        Parameter("emissivity", fiducial=0, latex=r"T_{elec}"),
    ]

    def model(self, **p):
        """Compute the IonContrib model."""
        return p["absorption"] * self.f**-4.5 + p["emissivity"] * self.f**-2


@attrs.define
class LinLog(Foreground):
    """
    LinLog model from Memo #122.

    Notes
    -----
    The model there is slightly ambiguous. Actually taking the Taylor Expansion
    of the ExpLog model makes it clear that a_1 (here p_1) should be
    equivalently zero. See Memo #122 for details.

    We leave that parameter open, but suggest not letting it vary, and leaving
    it as zero.

    Parameters
    ----------
    poly_order : int
        The maximum polynomial order will be `poly_order - 1`. There are
        `poly_order + 1` total parameters, including `beta` and `p1` (which should
        usually be set to zero).
    """

    poly_order = attrs.field(default=5, converter=int, kw_only=True)
    use_p1 = attrs.field(default=False, converter=bool)

    def __attrs_post_init__(self):
        """Perform validation after all parameters are set."""
        super().__attrs_post_init__()

        if not self.use_p1 and "p1" in self.child_active_params:
            raise ValueError(
                "You are attempting to fit p1, but it won't affect anything!"
            )

    @cached_property
    def base_parameters(self):
        """The base parameters of the model."""
        p = [
            Parameter("beta", -2.5, min=-5, max=0, latex=r"\beta"),
            Parameter("p0", 1750, latex=r"p_0"),
            Parameter("p1", 0, latex=r"p_1"),
        ]

        assert self.poly_order >= 1, "poly_order must be >= 1"

        # First create the parameters.
        p.extend(
            Parameter(f"p{i}", 0, latex=rf"p_{i}") for i in range(2, self.poly_order)
        )
        return tuple(p)

    @cached_property
    def logf(self):
        """The logarithm of the frequency array."""
        return np.log(self.f)

    @cached_property
    def basis(self):
        """Compute the basis functions of the linlog model."""
        return np.array([self.logf**i for i in range(self.poly_order)])

    def model(self, **p):
        """Compute the LinLog model."""
        pp = [p[f"p{i}"] for i in range(self.poly_order)]
        if not self.use_p1:
            pp[1] = 0

        terms = [pp[i] * self.basis[i] for i in range(self.poly_order)]
        return self.f ** p["beta"] * np.sum(terms, axis=0)


@attrs.define
class Sinusoid(Foreground):
    """An additive sinusoidal model."""

    base_parameters = [
        Parameter("amp", 0, min=0, max=1, latex=r"A_{\rm sin}"),
        Parameter("lambda", 10, min=1, max=30, latex=r"\lambda_{\rm sin}"),
        Parameter("phase", 0, min=-np.pi, max=np.pi, latex=r"\phi_{\rm sin}"),
    ]

    def model(self, **p):
        """Compute the sinusoid model."""
        return p["amp"] * np.sin(2 * np.pi * self.freqs / p["lambda"] + p["phase"])


@attrs.define
class DampedOscillations(Foreground):
    """A DampedOscillations model with a given period and amplitude of sin/cos."""

    base_parameters = [
        Parameter("amp_sin", 10e-10, min=-10, max=1000, latex=r"A_{\rm sin}"),
        Parameter("amp_cos", 10e-10, min=-10, max=1000, latex=r"A_{\rm cos}"),
        Parameter("P", 10, min=1, max=np.inf, latex=r"P_{\rm MHz}"),
        Parameter("b", 0, min=-10, max=10, latex=r"b"),
    ]

    def model(self, **p):
        """Compute the DampedOscillations model."""
        phase = 2 * np.pi * self.freqs / p["P"]
        return (self.f) ** p["b"] * (
            p["amp_sin"] * np.sin(phase) + p["amp_cos"] * np.cos(phase)
        )


@attrs.define
class DampedSinusoid(Component):
    """A damped sinusoid model, ala Sims+2019."""

    freqs: np.ndarray = attrs.field(kw_only=True, eq=attrs.cmp_using(eq=np.array_equal))
    provides = ("sin_spectrum",)

    base_parameters = [
        Parameter("amp", 0, min=0, max=1, latex=r"A_{\rm sin}"),
        Parameter("lambda", 10, min=1, max=30, latex=r"\lambda_{\rm sin}"),
        Parameter("phase", 0, min=-np.pi, max=np.pi, latex=r"\phi_{\rm sin}"),
    ]

    def calculate(self, ctx=None, **p):
        """Compute the DampedSinusoidal model."""
        models = np.array([v for k, v in ctx.items() if k.endswith("spectrum")])
        amp = np.sum(models, axis=0)
        amp *= p["amp"]
        return amp * np.sin(2 * np.pi * self.freqs / p["lambda"] + p["phase"])


class LinPoly(LinLog):
    """Eq. 10 from Hills et al.

    .. note:: The polynomial terms are offset by a prior assumption for beta: -2.5.

    The equation is

    .. math :: T(nu) = (nu/nuc)**-beta * Sum[p_i (nu/nuc)**i]
    """

    def model(self, **p):
        """Compute the LinPoly model."""
        terms = []
        for key, val in p.items():
            if key == "beta":
                continue
            i = int(key[1:])
            terms.append(val * self.f**i)

        return np.sum(terms, axis=0) * self.f ** p["beta"]


@attrs.define
class Bias(Component):
    """A Bias component that can be added to another component."""

    x: np.ndarray = attrs.field(kw_only=True, eq=attrs.cmp_using(eq=np.array_equal))
    centre = attrs.field(default=1, converter=float, kw_only=True)

    poly_order = attrs.field(default=1, converter=int, kw_only=True)
    kind = attrs.field(default="spectrum", kw_only=True)
    log = attrs.field(default=False, kw_only=True)
    additive = attrs.field(default=False, kw_only=True, converter=bool)

    @cached_property
    def base_parameters(self):
        """The base parameters of the Bias model."""
        p = [Parameter("b0", 1, min=-np.inf if self.additive else 0, latex=r"b_0")]

        assert self.poly_order >= 1, "poly_order must be >= 1"

        # First create the parameters.
        p.extend(
            Parameter(f"b{i}", 0, latex=rf"b_{i}") for i in range(1, self.poly_order)
        )
        return tuple(p)

    def evaluate_poly(self, **params):
        """Evaluate the polynomial model."""
        x = self.x / self.centre
        if self.log:
            x = np.log(x)

        return sum(params[f"b{i}"] * x**i for i in range(self.poly_order))

    def calculate(self, ctx, **params):
        """Compute the Bias model."""
        bias = self.evaluate_poly(**params)

        for key in ctx:
            if key.endswith(self.kind):
                if self.additive:
                    ctx[key] += bias
                    break  # only add to one thing, otherwise it's doubling up.
                ctx[key] *= bias


@attrs.define
class LogPoly(Foreground):
    """
    LogPoly model from Sims et.al. 2020 (Equation 18).

    T_{Fg} = 10^sum(d_i*log10(nu/nu_0)^i)_{i=0 to N}

    Parameters
    ----------
    poly_order : int
        The maximum polynomial order will be `poly_order `. There are `poly_order+1`
        total parameters.
    """

    poly_order = attrs.field(default=5, converter=int, kw_only=True)

    @cached_property
    def base_parameters(self):
        """The base parameters of the LogPoly model."""
        p = [
            Parameter("p0", 2, latex=r"p_0"),
        ]
        assert self.poly_order >= 1, "poly_order must be >= 1"

        # First create the parameters.
        p.extend(
            Parameter(f"p{i}", 0, latex=rf"p_{i}")
            for i in range(1, self.poly_order + 1)
        )
        return tuple(p)

    @cached_property
    def logf(self):
        """Compute the logarithm of the frequency array."""
        return np.log10(self.f)

    @cached_property
    def basis(self):
        """Compute the basis functions of the LogPoly model."""
        return np.array([self.logf**i for i in range(self.poly_order + 1)])

    def model(self, **p):
        """Compute the LogPoly model."""
        pp = [p[f"p{i}"] for i in range(self.poly_order + 1)]

        terms = [pp[i] * self.basis[i] for i in range(self.poly_order + 1)]
        return 10 ** np.sum(terms, axis=0)
