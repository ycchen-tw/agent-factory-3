"""Stress test: 800 concurrent MCP sessions simulating RL math training.

Each session simulates a rollout: multiple tool calls solving a math problem
with persistent state across calls (like a ReAct agent would).

Usage:
    uv run python mcp_tools/python_tool/stress_test.py
    uv run python mcp_tools/python_tool/stress_test.py --sessions 200 --url http://localhost:8811/mcp
"""

import argparse
import asyncio
import random
import time
import traceback

from fastmcp import Client

# ── Problem templates (multi-step, like real training rollouts) ─────────

PROBLEMS = [
    # 1. Sympy solve + verify
    [
        "from sympy import symbols, solve, Rational\nx = symbols('x')\nroots = solve(x**2 - {a}*x + {b}, x)\nprint('roots:', roots)",
        "print('verify:', all(r**2 - {a}*r + {b} == 0 for r in roots))",
        "print('sum:', sum(roots))",
    ],
    # 2. Numpy linear algebra
    [
        "import numpy as np\nA = np.random.RandomState({seed}).randn(4, 4)\nA = A @ A.T + np.eye(4)\nprint('det:', round(np.linalg.det(A), 4))",
        "eigvals = np.linalg.eigvals(A)\nprint('all_positive:', all(eigvals > 0))",
        "x = np.linalg.solve(A, np.ones(4))\nprint('residual:', round(np.linalg.norm(A @ x - np.ones(4)), 10))",
    ],
    # 3. Z3 constraint solving
    [
        "from z3 import *\nx, y, z = Ints('x y z')\ns = Solver()\ns.add(x + y + z == {target})\ns.add(x > 0, y > 0, z > 0)\ns.add(x < y, y < z)\nresult = s.check()\nprint('sat:', result)",
        "if result == sat:\n    m = s.model()\n    vals = [m[v].as_long() for v in [x,y,z]]\n    print('solution:', vals)\n    print('sum_check:', sum(vals) == {target})",
    ],
    # 4. Scipy optimization
    [
        "import numpy as np\nfrom scipy.optimize import minimize\ndef f(x): return (x[0]-{a})**2 + (x[1]-{b})**2 + (x[0]*x[1]-{c})**2\nres = minimize(f, [0, 0])\nprint('min:', res.fun < 0.01)\nprint('x:', np.round(res.x, 4).tolist())",
        "from scipy.integrate import quad\nval, _ = quad(lambda t: np.exp(-t**2/{a}), -10, 10)\nprint('integral:', round(val, 4))",
    ],
    # 5. Number theory (gmpy2 + sympy)
    [
        "import gmpy2\np = gmpy2.next_prime({big})\nprint('prime:', p)\nprint('is_prime:', gmpy2.is_prime(p))",
        "from sympy import factorint\nn = int(p) - 1\nf = factorint(n)\nprint('factors:', f)\nprint('verify:', eval('*'.join(f'{{k}}**{{v}}' for k,v in f.items())) == n)",
    ],
    # 6. Ortools CP-SAT
    [
        "from ortools.sat.python import cp_model\nm = cp_model.CpModel()\nvars = [m.new_int_var(1, {n}, f'v{{i}}') for i in range({n})]\nm.add_all_different(vars)\nm.add(sum(vars) == {target})\ns = cp_model.CpSolver()\nstatus = s.solve(m)\nprint('feasible:', status in [cp_model.FEASIBLE, cp_model.OPTIMAL])",
        "if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:\n    vals = [s.value(v) for v in vars]\n    print('solution:', vals)\n    print('sum_check:', sum(vals) == {target})\n    print('unique:', len(set(vals)) == len(vals))",
    ],
    # 7. Mpmath high precision
    [
        "from mpmath import mp, mpf\nmp.dps = 50\nresult = mp.sqrt({n}) * mp.pi\nprint('value:', result)",
        "mp.dps = 100\nresult2 = mp.sqrt({n}) * mp.pi\nprint('100digit:', result2)\nprint('consistent:', str(result)[:40] == str(result2)[:40])",
    ],
    # 8. Shapely geometry
    [
        "from shapely.geometry import Point, Polygon\ncircle = Point(0, 0).buffer({r})\nsquare = Polygon([(-1,-1),(1,-1),(1,1),(-1,1)])\ninter = circle.intersection(square)\nprint('area:', round(inter.area, 4))",
        "union = circle.union(square)\nprint('union_area:', round(union.area, 4))\nprint('ratio:', round(inter.area / union.area, 4))",
    ],
    # 9. Python-SAT
    [
        "from pysat.solvers import Glucose3\ng = Glucose3()\nfor i in range(1, {n}+1):\n    g.add_clause([i, i+{n}])\n    g.add_clause([-i, -i-{n}])\nprint('sat:', g.solve())",
        "model = g.get_model()\nprint('model_len:', len(model))\nprint('consistent:', all(not (m > 0 and -m in model) for m in model[:5]))",
    ],
    # 10. Mixed: sympy + numpy verification
    [
        "from sympy import Matrix, Rational\nM = Matrix([[Rational({a},{b}), 1], [0, Rational({c},{d})]])\nprint('det:', M.det())\nprint('eigenvals:', M.eigenvals())",
        "import numpy as np\nMn = np.array([[{a}/{b}, 1], [0, {c}/{d}]])\nprint('np_det:', round(np.linalg.det(Mn), 6))\nprint('match:', abs(float(M.det()) - np.linalg.det(Mn)) < 1e-10)",
    ],
]


def make_problem(idx: int) -> list[str]:
    """Generate a concrete problem instance with random parameters."""
    rng = random.Random(idx)
    template = PROBLEMS[idx % len(PROBLEMS)]

    params = {
        'a': rng.randint(2, 20),
        'b': rng.randint(1, 50),
        'c': rng.randint(1, 10),
        'd': rng.randint(1, 10),
        'n': rng.randint(3, 8),
        'r': round(rng.uniform(0.5, 2.0), 2),
        'seed': rng.randint(0, 10000),
        'target': rng.randint(10, 30),
        'big': rng.randint(10**8, 10**12),
    }

    return [step.format(**params) for step in template]


async def run_session(session_idx: int, url: str, results: dict):
    """Run a single simulated rollout session."""
    t0 = time.monotonic()
    steps_done = 0
    error = None

    try:
        async with Client(url) as client:
            problem = make_problem(session_idx)
            for step_code in problem:
                r = await asyncio.wait_for(
                    client.call_tool('python', {'code': step_code}),
                    timeout=30.0,
                )
                output = r.data
                if '[ERROR]' in output:
                    error = output.split('\n')[0]
                    break
                steps_done += 1
    except asyncio.TimeoutError:
        error = 'timeout'
    except Exception as e:
        error = f'{type(e).__name__}: {e}'

    elapsed = time.monotonic() - t0
    results['total'] += 1
    results['steps'] += steps_done
    results['elapsed'].append(elapsed)
    if error:
        results['errors'].append((session_idx, error))
    else:
        results['success'] += 1


async def main(num_sessions: int, concurrency: int, url: str):
    results = {
        'total': 0,
        'success': 0,
        'steps': 0,
        'errors': [],
        'elapsed': [],
    }

    sem = asyncio.Semaphore(concurrency)

    async def bounded(idx):
        async with sem:
            await run_session(idx, url, results)

    print(f"=== Stress Test: {num_sessions} sessions, concurrency={concurrency} ===")
    print(f"URL: {url}")
    print()

    t_start = time.monotonic()

    # Launch in batches to avoid overwhelming connection setup
    tasks = [asyncio.create_task(bounded(i)) for i in range(num_sessions)]

    # Progress reporting
    last_report = t_start
    while not all(t.done() for t in tasks):
        await asyncio.sleep(2)
        now = time.monotonic()
        if now - last_report >= 5:
            done = results['total']
            errs = len(results['errors'])
            elapsed = now - t_start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] {done}/{num_sessions} done, "
                  f"{results['success']} ok, {errs} errors, "
                  f"{rate:.1f} sessions/s")
            last_report = now

    await asyncio.gather(*tasks)
    t_total = time.monotonic() - t_start

    # Results
    print()
    print(f"=== Results ===")
    print(f"Total:      {results['total']}")
    print(f"Success:    {results['success']}")
    print(f"Errors:     {len(results['errors'])}")
    print(f"Steps done: {results['steps']}")
    print(f"Wall time:  {t_total:.1f}s")
    print(f"Throughput: {results['total'] / t_total:.1f} sessions/s")

    if results['elapsed']:
        el = sorted(results['elapsed'])
        print(f"Latency:    p50={el[len(el)//2]:.2f}s  p90={el[int(len(el)*0.9)]:.2f}s  "
              f"p99={el[int(len(el)*0.99)]:.2f}s  max={el[-1]:.2f}s")

    if results['errors']:
        print()
        # Group errors
        error_types: dict[str, int] = {}
        for idx, err in results['errors']:
            key = err[:80]
            error_types[key] = error_types.get(key, 0) + 1
        print(f"Error breakdown ({len(results['errors'])} total):")
        for err, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  [{count:>4}x] {err}")

    success_rate = results['success'] / max(results['total'], 1) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=800)
    parser.add_argument('--concurrency', type=int, default=100)
    parser.add_argument('--url', default='http://127.0.0.1:8811/mcp')
    args = parser.parse_args()

    asyncio.run(main(args.sessions, args.concurrency, args.url))
