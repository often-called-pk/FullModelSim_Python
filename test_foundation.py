"""Validate the foundation helpers against analytic ground truth (no casadi)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from functions.rotatePoint2D import rotatePoint2D
from functions.curv2cart import curv2cart
from functions.cartPath import cartPath
from functions.trackLimits import trackLimits
from functions.simpleMA import simpleMA
from functions.collocation import collocation_points, collocation_coeff
from functions.context import Ctx
from Powertrain import Powertrain

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

print("rotatePoint2D")
ok("identity (alpha=0)", np.allclose(rotatePoint2D(0.0, [3.0, 4.0]), [3.0, 4.0]))
# +90 deg (pi/2): MATLAB R=[[0,-1],[1,0]], P*R = [P0*0+P1*1, -P0*1+P1*0] = [P1, -P0]
ok("rotate +pi/2 -> [y,-x]", np.allclose(rotatePoint2D(np.pi/2, [1.0, 0.0]), [0.0, -1.0]))

print("collocation_coeff (Legendre, d=3)")
d = 3
tau = collocation_points(d, "legendre")
ok("3 points in (0,1)", len(tau) == 3 and all(0 < t < 1 for t in tau))
ok("points ~ Gauss-Legendre", np.allclose(sorted(tau), [0.112701665, 0.5, 0.887298335], atol=1e-6))
C, D, B = collocation_coeff(tau)
ok("C shape (d+1,d)", C.shape == (d + 1, d))
ok("D shape (d+1,1)", D.shape == (d + 1, 1))
ok("B shape (d,1)", B.shape == (d, 1))
# Gauss weights on [0,1] sum to 1 and integrate t^p exactly up to degree 2d-1
ok("B sums to 1", abs(float(B.sum()) - 1.0) < 1e-12)
taua = np.array(tau)
for p in range(2 * d):  # exact up to degree 5
    quad = float((B[:, 0] * taua**p).sum())
    ok(f"  Gauss exact for t^{p}", abs(quad - 1.0 / (p + 1)) < 1e-10)
# Differentiation/continuity on x(t)=t^3 with values at augmented nodes [0, tau]
Z = np.array([[0.0] + [t**3 for t in tau]])             # 1 x (d+1), row of node values
dPi = Z @ C                                             # derivatives at collocation pts (1 x d)
ok("C reproduces d/dt t^3 = 3t^2", np.allclose(dPi.ravel(), 3 * taua**2, atol=1e-10))
ok("D reproduces x(1)=1 for t^3", abs((Z @ D).item() - 1.0) < 1e-10)

print("curv2cart")
# Straight line along x: s = 0..10, k = 0  ->  y stays 0, x increases
s = np.linspace(0, 10, 11)
k = np.zeros_like(s)
x, y = curv2cart(s, k)
ok("straight: y ~ 0", np.allclose(y, 0.0, atol=1e-9))
ok("straight: x monotonic to ~10", abs(x[-1] - 10.0) < 1e-6)
# Circle: constant curvature should close to ~ a circle of radius 1/k
R = 50.0
s = np.linspace(0, 2 * np.pi * R, 2000)
k = (1.0 / R) * np.ones_like(s)
x, y = curv2cart(s, k)
radii = np.sqrt((x - x.mean())**2 + (y - y.mean())**2)
ok("circle: ~constant radius", np.std(radii) / np.mean(radii) < 0.02)

print("cartPath / trackLimits")
# Straight centreline along x; offset n shifts in +y (normal = [-dy,dx]=[0,1])
xc = np.linspace(0, 10, 11); yc = np.zeros_like(xc)
xo, yo = cartPath(xc, yc, np.full_like(xc, 2.0))
ok("cartPath offset +2 in y", np.allclose(yo, 2.0, atol=1e-9))
Xl, Xr = trackLimits(xc, yc, 4.0)            # width 4 -> +/-2
ok("left limit at +2", np.allclose(Xl[:, 1], 2.0, atol=1e-9))
ok("right limit at -2", np.allclose(Xr[:, 1], -2.0, atol=1e-9))
ok("limits length matches centreline", Xl.shape[0] == xc.size and Xr.shape[0] == xc.size)

print("simpleMA")
step = np.concatenate([np.zeros(50), np.ones(50)])
sm = simpleMA(step, 10, 2)
ok("same length", sm.size == step.size)
ok("smoother than step (smaller max gradient)", np.max(np.abs(np.diff(sm))) < np.max(np.abs(np.diff(step))))
ok("approx value preserved on interior flats",
   abs(sm[20:40].mean()) < 1e-6 and abs(sm[60:85].mean() - 1.0) < 1e-6)

print("Powertrain")
ctx = Ctx()
Powertrain(ctx)
ok("Pmax = 450000 W", ctx.pt.Pmax == 450000.0)
ok("Tmax = 429 Nm", ctx.pt.Tmax == 429.0)
ok("OMmax = 25000*pi/30", abs(ctx.pt.OMmax - 25000 * np.pi / 30) < 1e-9)
ok("Vmax = 105.56 m/s", abs(ctx.pt.Vmax - 380 / 3.6) < 1e-9)
ok("Rw = 0.35", ctx.vp.Rw == 0.35)
ok("gear = OMmax*Rw/Vmax", abs(ctx.vp.gear - (ctx.pt.OMmax * ctx.vp.Rw) / ctx.pt.Vmax) < 1e-12)

print("\nALL FOUNDATION TESTS PASSED")
