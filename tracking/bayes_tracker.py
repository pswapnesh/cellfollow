"""Bayesian tracker: every step is an update of a probability, on each cell's learnt Gaussian shape.

State     Kalman filter on all 5 positions and velocities, measured with the learnt joint error covariance R, and on the
          matrix log of each point's 2 x 2 shape block, so the covariances stay valid.
Match     track i and detection j are the same cell with log odds
              LLR_ij = log(p_i p_D / (1 - p_i p_D)) + log(P_j / lam_new) - KL_ij / T
          KL_ij: symmetric KL of the two aligned shapes, read as a log likelihood. The Hungarian maximises the sum and a
          pair needs LLR > 0, so not matching is always an option and there is no gate.
Existence Bayes' rule: a match adds the detection's log odds, a miss multiplies the odds by 1 - p_D.
Birth     unmatched detections with P_j >= birth start at p = P_j; with masks, one track per mask, none outside.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def to_log(S):
    """2 x 2 covariances (..., 2, 2) -> the 3 parameters (..., 3) of their matrix log."""
    w, V = np.linalg.eigh(S)
    A = (V * np.log(np.maximum(w, 1e-6))[..., None, :]) @ np.swapaxes(V, -1, -2)
    return np.stack([A[..., 0, 0], A[..., 0, 1], A[..., 1, 1]], -1)


def from_log(v):
    """Inverse of to_log: always positive definite."""
    A = np.stack([v[..., 0], v[..., 1], v[..., 1], v[..., 2]], -1).reshape(*v.shape[:-1], 2, 2)
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def logit(p):
    return np.log(p) - np.log1p(-p)


@dataclass
class Track:
    id: int
    y: float
    x: float
    vy: float
    vx: float
    p_exist: float
    cls: int
    cls_prob: np.ndarray
    points: np.ndarray  # (5, 2) centre + 4 eigen points, y, x
    shape: np.ndarray   # (5, 2, 2) covariance of each point's Gaussian


class BayesTracker:
    def __init__(self, Q, nc=4, k=4, T=8.0, lam_new=0.1, p_d=0.9, q_mult=4.0, q_vel=16.0, q_shape=1.0, r_shape=1.0,
                 birth=0.1, p_show=0.5, p_remove=0.1, k_near=8, v0=25.0, a_cls=0.8):
        self.T, self.lam_new, self.p_d, self.k_near = T, lam_new, p_d, k_near
        self.q_shape, self.r_shape, self.birth, self.p_show, self.p_remove = q_shape, r_shape, birth, p_show, p_remove
        self.v0, self.a_cls, self.k, self.nc = v0, a_cls, k, nc
        self.d = d = 2 * (k + 1)
        self.F = np.eye(2 * d)
        self.F[:d, d:] = np.eye(d)
        self.Q = np.zeros((2 * d, 2 * d))
        self.Q[:d, :d] = q_mult * np.asarray(Q)
        self.Q[d:, d:] = q_vel * (np.kron(np.ones((k + 1, k + 1)), np.eye(2)) + 0.25 * np.eye(d))
        self.perms = np.array(list(itertools.permutations(range(k))))
        self.reset()

    def reset(self):
        """Forget all tracks."""
        d, k = self.d, self.k
        self.next_id = 1
        self.ids, self.p, self.lost = np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
        self.X, self.P = np.zeros((0, 2 * d)), np.zeros((0, 2 * d, 2 * d))
        self.S, self.V, self.cls = np.zeros((0, k + 1, 2, 2)), np.zeros((0, k + 1)), np.zeros((0, self.nc))

    def align(self, rows, cols, mu):
        """Order (pairs, k) of the detections' eigen points that best fits each track."""
        ot = (self.X[rows, 2:self.d] - np.tile(self.X[rows, :2], self.k)).reshape(-1, self.k, 2)
        od = (mu[cols, 2:] - np.tile(mu[cols, :2], self.k)).reshape(-1, self.k, 2)
        return self.perms[np.abs(ot[:, None] - od[:, self.perms]).sum((-1, -2)).argmin(1)]

    def order(self, perm):
        return np.concatenate([np.zeros((len(perm), 1), np.int64), perm + 1], 1)

    def reorder(self, mu, R, perm):
        q = self.order(perm)
        idx = np.stack([2 * q, 2 * q + 1], -1).reshape(len(perm), -1)
        r = np.arange(len(perm))[:, None]
        return mu[r, idx], R[r[:, None], idx[:, :, None], idx[:, None, :]]

    def kl(self, rows, cols, mu, shape):
        """Symmetric KL (pairs,) of the track's and the detection's aligned shapes, mean per point."""
        q = self.order(self.align(rows, cols, mu))
        r = np.arange(len(rows))[:, None]
        Sd, St = shape[cols][r, q], self.S[rows]
        e = self.X[rows, :self.d].reshape(len(rows), -1, 2) - mu[cols].reshape(len(rows), -1, 2)[r, q]
        Si, Di = np.linalg.inv(St), np.linalg.inv(Sd)
        tr = (Di * St).sum((-1, -2)) + (Si * Sd).sum((-1, -2))
        return (0.5 * (tr - 4) + 0.5 * (e * ((Si + Di) @ e[..., None])[..., 0]).sum(-1)).mean(1)

    def log_odds(self, mu, shape, prob):
        """LLR (T, M), minus infinity outside each track's k nearest detections."""
        T, M = len(self.ids), len(mu)
        k = min(self.k_near, M)
        rows = np.repeat(np.arange(T), k)
        cols = cKDTree(mu[:, :2]).query(self.X[:, :2], k=k)[1].reshape(T, k).ravel()
        a = np.clip(self.p[rows] * self.p_d, 1e-6, 1 - 1e-6)
        L = np.full((T, M), -np.inf)
        L[rows, cols] = logit(a) + np.log(prob[cols] / self.lam_new) - self.kl(rows, cols, mu, shape) / self.T
        return L

    def correct(self, t, m, mu, R, shape, prob, cls_prob):
        """Update the matched tracks t with their detections m."""
        d, perm = self.d, self.align(t, m, mu)
        mu_a, R_a = self.reorder(mu[m], R[m], perm)
        K = np.linalg.solve(self.P[t, :d, :d] + R_a, self.P[t, :d, :]).transpose(0, 2, 1)
        self.X[t] += (K @ (mu_a - self.X[t, :d])[..., None])[..., 0]
        P = self.P[t] - K @ self.P[t, :d, :]
        self.P[t] = 0.5 * (P + P.transpose(0, 2, 1))
        gain = (self.V[t] / (self.V[t] + self.r_shape))[..., None]
        log_s = to_log(self.S[t])
        self.S[t] = from_log(log_s + gain * (to_log(shape[m][np.arange(len(m))[:, None], self.order(perm)]) - log_s))
        self.V[t] = self.V[t] * (1 - gain[..., 0])
        self.cls[t] = self.a_cls * self.cls[t] + (1 - self.a_cls) * cls_prob[m]
        self.p[t] = np.clip(1 / (1 + np.exp(-(logit(np.clip(self.p[t], 1e-3, 1 - 1e-3)) + logit(prob[m])))), 1e-3, 0.999)

    def births(self, free, prob, mu, labels):
        """Detections that start a track, best first with masks."""
        new = np.nonzero(free & (prob >= self.birth))[0]
        if labels is None or not len(mu):
            return new
        H, W = labels.shape
        lab = labels[np.clip(np.round(mu[:, 0]).astype(int), 0, H - 1), np.clip(np.round(mu[:, 1]).astype(int), 0, W - 1)]
        held, chosen = set(lab[~free].tolist()), []
        for i in new[np.argsort(-prob[new])]:
            if lab[i] > 0 and lab[i] not in held:
                held.add(lab[i])
                chosen.append(i)
        return np.array(chosen, np.int64)

    def update(self, mu, R, shape, prob, cls_prob, labels=None):
        """mu (M, D) points, R (M, D, D) joint error, shape (M, P, 2, 2), prob (M,) P(real), cls_prob (M, nc), labels (H, W)
        optional instance masks for births. Returns the reported tracks."""
        d = self.d
        prob = np.clip(prob, 1e-3, 1 - 1e-3)
        self.X = self.X @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.V = self.V + self.q_shape
        free, hit = np.ones(len(mu), bool), np.zeros(len(self.ids), bool)
        if len(self.ids) and len(mu):
            L = self.log_odds(mu, shape, prob)
            r, c = linear_sum_assignment(-np.maximum(L, 0))       # a pair with LLR <= 0 gains nothing over no match
            ok = L[r, c] > 0
            if ok.any():
                self.correct(r[ok], c[ok], mu, R, shape, prob, cls_prob)
                hit[r[ok]], free[c[ok]] = True, False
        miss = ~hit
        self.p[miss] = self.p[miss] * (1 - self.p_d) / (self.p[miss] * (1 - self.p_d) + 1 - self.p[miss])
        self.lost = np.where(hit, 0, self.lost + 1)
        keep = self.p >= self.p_remove
        new = self.births(free, prob, mu, labels)
        n = len(new)
        X, P = np.zeros((n, 2 * d)), np.zeros((n, 2 * d, 2 * d))
        X[:, :d] = mu[new]
        P[:, :d, :d] = R[new]
        P[:, range(d, 2 * d), range(d, 2 * d)] = self.v0
        self.ids = np.concatenate([self.ids[keep], np.arange(self.next_id, self.next_id + n)])
        self.next_id += n
        self.X, self.P = np.concatenate([self.X[keep], X]), np.concatenate([self.P[keep], P])
        self.S = np.concatenate([self.S[keep], shape[new]])
        self.V = np.concatenate([self.V[keep], np.full((n, self.k + 1), self.r_shape)])
        self.p = np.concatenate([self.p[keep], prob[new]])
        self.lost = np.concatenate([self.lost[keep], np.zeros(n, np.int64)])
        self.cls = np.concatenate([self.cls[keep], cls_prob[new]])
        idx = np.nonzero((self.lost == 0) & (self.p >= self.p_show))[0]
        return [Track(int(self.ids[i]), self.X[i, 0], self.X[i, 1], self.X[i, d], self.X[i, d + 1], float(self.p[i]),
                      int(self.cls[i].argmax()), self.cls[i], self.X[i, :d].reshape(-1, 2), self.S[i]) for i in idx]

    def current(self):
        """(ids, yx, p_exist) of all tracks matched this frame, tentative ones included."""
        i = self.lost == 0
        return self.ids[i], self.X[i, :2], self.p[i]
