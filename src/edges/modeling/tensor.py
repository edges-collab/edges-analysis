"""Tensor-product linear models over two coordinates (e.g. frequency and LST).

A tensor-product model is ``sum_{j,k} c_jk * a_j(x) * b_k(y)``: every function of one
basis times every function of the other. Here the first basis is that of any
:class:`~edges.modeling.FixedLinearModel` in frequency, and the second a polynomial in
the second coordinate (e.g. LST). This is the model used by
:func:`edges.filters.xrfi.xrfi_iterative_sliding_window_2d`.
"""

import numpy as np


def polynomial_basis(
    coords: np.ndarray, use: np.ndarray, n_terms: int
) -> tuple[np.ndarray, int]:
    """Get a polynomial basis on the range of the used coordinates, mapped to [0, 1].

    Parameters
    ----------
    coords
        The coordinate of each row (e.g. the LST of each LST bin).
    use
        Boolean array, True for the rows whose coordinates define the range (e.g. the
        LST bins that have data).
    n_terms
        The number of polynomial terms. It is capped at the number of used rows.

    Returns
    -------
    basis
        The basis, of shape ``(n_terms, len(coords))``: ``x**k`` for ``k < n_terms``,
        where ``x`` maps the used range onto [0, 1].
    n_terms
        The number of terms in ``basis``.
    """
    coords = np.asarray(coords, dtype=float)
    use = np.asarray(use, dtype=bool)
    n_terms = max(1, min(int(n_terms), int(np.sum(use))))
    lo, hi = np.min(coords[use]), np.max(coords[use])
    x = (coords - lo) / (hi - lo) if hi > lo else np.zeros_like(coords)
    return np.array([x**k for k in range(n_terms)]), n_terms


def fit_tensor_product(
    freq_basis: np.ndarray,
    second_basis: np.ndarray,
    data: np.ndarray,
    weights: np.ndarray,
    rows: np.ndarray | None = None,
    method: str = "lstsq",
) -> np.ndarray:
    """Weighted fit of ``sum_{j,k} c_jk * freq_basis[j](f) * second_basis[k](t)``.

    Parameters
    ----------
    freq_basis
        The basis along the last axis of ``data``, of shape ``(n_freq_terms, n_freq)``
        (e.g. ``FixedLinearModel.basis``).
    second_basis
        The basis along the first axis of ``data``, of shape ``(n_terms, n_rows)``
        (e.g. from :func:`polynomial_basis`).
    data
        The data, of shape ``(n_rows, n_freq)``. Values with zero weight are ignored
        (they may be NaN).
    weights
        The weights (inverse variance, e.g. nsamples), the same shape as ``data``.
    rows
        The rows to include in the fit. By default every row with any weight.
    method
        With ``"alan-qrd"`` the normal equations are built as in Alan's C code (summed
        one row at a time, unscaled) and solved with its QR decomposition. Otherwise
        the design matrix of the samples with weight, scaled by ``sqrt(w)`` and with
        normalised columns, is solved with :func:`numpy.linalg.lstsq`, like the
        ``lstsq`` method of :class:`~edges.modeling.ModelFit`.

    Returns
    -------
    model
        The fitted model at every row and frequency, of shape ``(n_rows, n_freq)``.
    """
    freq_basis = np.asarray(freq_basis, dtype=float)
    second_basis = np.asarray(second_basis, dtype=float)
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if rows is None:
        rows = np.where(np.any(weights > 0, axis=1))[0]

    n_freq_terms = freq_basis.shape[0]
    n_second_terms = second_basis.shape[0]

    if method == "alan-qrd":
        from .fitting import _c_qrd

        npar = n_freq_terms * n_second_terms
        aarr = np.zeros((npar, npar))
        bbrr = np.zeros(npar)
        for t in rows:
            w = weights[t]
            y = np.where(w > 0, data[t], 0.0)
            wa = np.multiply(freq_basis, w, order="C")
            gram = np.dot(wa, freq_basis.T)
            proj = np.dot(wa, y)
            lt = second_basis[:, t]
            aarr += np.kron(np.outer(lt, lt), gram)
            bbrr += np.kron(lt, proj)
        _, coeffs = _c_qrd(aarr, bbrr)
    else:
        use = np.zeros(weights.shape, dtype=bool)
        use[rows] = weights[rows] > 0
        t_idx, f_idx = np.nonzero(use)
        sqrt_w = np.sqrt(weights[t_idx, f_idx])
        design = (
            second_basis[:, t_idx][:, None, :] * freq_basis[None, :, f_idx]
        ).reshape(n_second_terms * n_freq_terms, -1).T * sqrt_w[:, None]
        norms = np.linalg.norm(design, axis=0)
        norms[norms == 0] = 1.0
        coeffs = (
            np.linalg.lstsq(design / norms, data[t_idx, f_idx] * sqrt_w, rcond=None)[0]
            / norms
        )

    coeffs = np.asarray(coeffs).reshape(n_second_terms, n_freq_terms)
    return second_basis.T @ (coeffs @ freq_basis)
