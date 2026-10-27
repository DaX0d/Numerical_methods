import itertools
import math
import re
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "2_2" / "variant_01.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MAX_ITER = 1000
DIVERGENCE = 1e4
GRID_TOTAL = 2000
H = 1e-6

MATH_NAMES = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tg": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "arcsin": math.asin, "arccos": math.acos, "arctg": math.atan,
    "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10, "lg": math.log10,
    "log2": math.log2, "sqrt": math.sqrt, "abs": abs, "pi": math.pi, "e": math.e,
}
EVAL_ERRORS = (ValueError, ZeroDivisionError, OverflowError)


def read_input(path):
    """Читает строки вида "имя = значение" в словарь."""
    params = {}
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"строка {line_no}: ожидается запись вида 'имя = значение'")
            name, value = line.split("=", 1)
            params[name.strip()] = value.strip()
    return params


def numbers(text, count, name):
    try:
        values = [float(t) for t in text.split()]
    except ValueError:
        raise ValueError(f"{name}: не удалось прочитать числа") from None
    if len(values) != count:
        raise ValueError(f"{name}: должно быть чисел: {count}, а записано: {len(values)}")
    return values


def compile_expression(text, name):
    """Строку из файла превращаем в скомпилированное выражение Python (^ означает степень)."""
    try:
        return compile(text.replace("^", "**"), name, "eval")
    except SyntaxError:
        raise ValueError(f"{name}: ошибка в записи выражения '{text}'") from None


class Problem:
    """Система из входного файла: функции F(x), матрица Якоби J(x) и Phi(x) для простой итерации."""

    def __init__(self, params):
        if "eps" not in params or "region" not in params:
            raise ValueError("нужны параметры eps и region")
        self.eps = float(params["eps"])
        if self.eps <= 0:
            raise ValueError("eps должно быть положительным")

        self.n = sum(1 for key in params if re.fullmatch(r"f\d+", key))
        n = self.n
        if n == 0:
            raise ValueError("не заданы уравнения f1, f2, ...")
        self.f_text = [self.require(params, f"f{i}") for i in range(1, n + 1)]
        self.phi_text = [self.require(params, f"phi{i}") for i in range(1, n + 1)]
        self.f_code = [compile_expression(t, f"f{i + 1}") for i, t in enumerate(self.f_text)]
        self.phi_code = [compile_expression(t, f"phi{i + 1}") for i, t in enumerate(self.phi_text)]
        self.J_code = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                key = f"J{i + 1}{j + 1}"
                if key not in params and f"J{i + 1}_{j + 1}" in params:
                    key = f"J{i + 1}_{j + 1}"
                self.J_code[i][j] = compile_expression(self.require(params, key), key)

        bounds = numbers(params["region"], 2 * n, "region")
        self.region = [(bounds[2 * i], bounds[2 * i + 1]) for i in range(n)]
        if any(lo >= hi for lo, hi in self.region):
            raise ValueError("region: в каждой паре левая граница должна быть меньше правой")
        self.plot = None
        if "plot" in params:
            p = numbers(params["plot"], 4, "plot")
            self.plot = [(p[0], p[1]), (p[2], p[3])]

        # Параметры задачи: все остальные строки "имя = число" (например, a = 2)
        self.constants = {}
        for key, value in params.items():
            if key in ("eps", "region", "plot") or re.fullmatch(r"(f|phi)\d+|J\d+_?\d+", key):
                continue
            try:
                self.constants[key] = float(value)
            except ValueError:
                raise ValueError(f"непонятный параметр '{key} = {value}'") from None
        self.env = {"__builtins__": {}, **MATH_NAMES, **self.constants}
        self.phi_uses_f = any(f"f{i + 1}" in code.co_names for code in self.phi_code for i in range(n))

    @staticmethod
    def require(params, key):
        if key not in params:
            raise ValueError(f"не задан параметр '{key}'")
        return params[key]

    def evaluate(self, code, local):
        value = eval(code, self.env, local)
        if isinstance(value, complex):
            raise ValueError("значение вне области определения")
        return value

    def variables(self, x):
        return {f"x{i + 1}": x[i] for i in range(self.n)}

    def F(self, x):
        local = self.variables(x)
        return [self.evaluate(code, local) for code in self.f_code]

    def J(self, x):
        local = self.variables(x)
        return [[self.evaluate(code, local) for code in row] for row in self.J_code]

    def Phi(self, x):
        local = self.variables(x)
        if self.phi_uses_f:
            local.update({f"f{i + 1}": v for i, v in enumerate(self.F(x))})
        return [self.evaluate(code, local) for code in self.phi_code]


def vector_norm(v):
    """||v|| = max |v_i|."""
    return max(abs(t) for t in v)


def matrix_norm(M):
    """||M|| = max_i sum_j |m_ij| — согласована с векторной нормой max |v_i|."""
    return max(sum(abs(t) for t in row) for row in M)


def solve_linear(A, b):
    """Решение A*x = b методом Гаусса с выбором главного элемента по столбцу."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-14:
            raise ZeroDivisionError("матрица Якоби вырождена")
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            l = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= l * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def numerical_jacobian(func, x):
    """Матрица производных d func_i / d x_j центральными разностями (только для проверки условий)."""
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    for j in range(n):
        xp, xm = list(x), list(x)
        xp[j] += H
        xm[j] -= H
        fp, fm = func(xp), func(xm)
        for i in range(n):
            J[i][j] = (fp[i] - fm[i]) / (2 * H)
    return J


def grid_points(region):
    per_dim = max(3, round(GRID_TOTAL ** (1 / len(region))))
    axes = [[lo + (hi - lo) * k / (per_dim - 1) for k in range(per_dim)] for lo, hi in region]
    return [list(p) for p in itertools.product(*axes)]


def check_simple_iteration(problem):
    """
    Достаточные условия сходимости метода простой итерации в области G:
      1) Phi(x) принадлежит G для всех x из G;
      2) q = max ||Phi'(x)|| < 1 на G (Phi'(x) — матрица частных производных d phi_i / d x_j).
    Проверяются по сетке точек. Возвращает q и признак Phi(G) в G.
    """
    q = 0.0
    inside = True
    for x in grid_points(problem.region):
        y = problem.Phi(x)
        if any(not lo <= y[i] <= hi for i, (lo, hi) in enumerate(problem.region)):
            inside = False
        q = max(q, matrix_norm(numerical_jacobian(problem.Phi, x)))
    return q, inside


def simple_iteration(problem, x0, q, eps):
    """
    x(k+1) = Phi(x(k)).
    Критерий окончания:  q/(1-q) * ||x(k+1) - x(k)|| <= eps  (при q < 1),
                         иначе ||x(k+1) - x(k)|| <= eps.
    Возвращает x, history = [(x(k), ||x(k) - x(k-1)||, оценка погрешности)], status.
    """
    coef = q / (1 - q) if q < 1 else 1.0
    x = list(x0)
    history = [(x, None, None)]
    for _ in range(MAX_ITER):
        try:
            x_new = problem.Phi(x)
        except EVAL_ERRORS:
            return x, history, "diverged"
        diff = vector_norm([x_new[i] - x[i] for i in range(len(x))])
        history.append((x_new, diff, coef * diff))
        x = x_new
        if coef * diff <= eps:
            return x, history, "converged"
        if not math.isfinite(diff) or diff > DIVERGENCE * history[1][1]:
            return x, history, "diverged"
    return x, history, "max_iter"


def newton(problem, x0, eps):
    """
    x(k+1) = x(k) + dx(k), где dx(k) — решение линейной системы J(x(k)) * dx = -f(x(k)).
    Критерий окончания: ||x(k+1) - x(k)|| <= eps.
    Возвращает x, history = [(x(k), ||x(k) - x(k-1)||)], status.
    """
    x = list(x0)
    history = [(x, None)]
    for _ in range(MAX_ITER):
        try:
            dx = solve_linear(problem.J(x), [-v for v in problem.F(x)])
        except ZeroDivisionError:
            return x, history, "singular"
        except EVAL_ERRORS:
            return x, history, "diverged"
        x_new = [x[i] + dx[i] for i in range(len(x))]
        diff = vector_norm(dx)
        history.append((x_new, diff))
        x = x_new
        if diff <= eps:
            return x, history, "converged"
        if not math.isfinite(diff) or diff > DIVERGENCE * history[1][1]:
            return x, history, "diverged"
    return x, history, "max_iter"


def fmt_vector(x, prec=8):
    return "(" + "; ".join(f"{t:.{prec}f}" for t in x) + ")"


def print_table(headers, rows):
    widths = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 70 - len(title)))


def status_text(status):
    return {"converged": "точность достигнута",
            "diverged": "метод РАСХОДИТСЯ",
            "max_iter": f"точность не достигнута за {MAX_ITER} итераций",
            "singular": "матрица Якоби вырождена — метод Ньютона неприменим"}[status]


def error_ratio(errors, k, power):
    """e(k) / e(k-1)^power; около машинной точности отношение теряет смысл — тогда '-'."""
    if k == 0 or errors[k] < 1e-13 or errors[k - 1] < 1e-13:
        return "-"
    return f"{errors[k] / errors[k - 1] ** power:.4f}"


def iteration_table(history, errors, power, with_estimate):
    """Строки таблицы итераций: k, x(k), ||x(k) - x(k-1)||, [оценка], ||x(k) - x*||, отношение."""
    n = len(history[0][0])
    headers = ["k"] + ([f"x{i + 1}" for i in range(n)] if n <= 4 else ["x(k)"]) + ["||x(k)-x(k-1)||"]
    headers += (["оценка погр."] if with_estimate else []) + ["||x(k)-x*||"]
    headers += ["e(k)/e(k-1)" if power == 1 else "e(k)/e(k-1)^2"]
    rows = []
    for k, item in enumerate(history):
        x, diff = item[0], item[1]
        row = [str(k)] + ([f"{t:.8f}" for t in x] if n <= 4 else [fmt_vector(x, 6)])
        row.append("-" if diff is None else f"{diff:.2e}")
        if with_estimate:
            row.append("-" if item[2] is None else f"{item[2]:.2e}")
        row += [f"{errors[k]:.2e}" if errors else "-", error_ratio(errors, k, power) if errors else "-"]
        rows.append(row)
    print_table(headers, rows)


def plot_curves(problem, solution, path):
    """График кривых f1(x1, x2) = 0 и f2(x1, x2) = 0 — решение системы в точке их пересечения."""
    import numpy as np
    import matplotlib.pyplot as plt

    (x1_lo, x1_hi), (x2_lo, x2_hi) = problem.plot or [
        (lo - (hi - lo), hi + (hi - lo)) for lo, hi in problem.region]
    X1, X2 = np.meshgrid(np.linspace(x1_lo, x1_hi, 400), np.linspace(x2_lo, x2_hi, 400))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, color in ((0, "tab:blue"), (1, "tab:orange")):
        Z = np.full(X1.shape, np.nan)
        for r in range(X1.shape[0]):
            for c in range(X1.shape[1]):
                try:
                    Z[r, c] = problem.evaluate(problem.f_code[i], {"x1": X1[r, c], "x2": X2[r, c]})
                except EVAL_ERRORS:
                    pass
        ax.contour(X1, X2, Z, levels=[0], colors=color)
        ax.plot([], [], color=color, label=f"f{i + 1} = 0:  {problem.f_text[i]}")
    (g1_lo, g1_hi), (g2_lo, g2_hi) = problem.region
    ax.add_patch(plt.Rectangle((g1_lo, g2_lo), g1_hi - g1_lo, g2_hi - g2_lo,
                               color="green", alpha=0.15, label="область G"))
    if solution is not None:
        ax.plot([solution[0]], [solution[1]], "ro",
                label=f"решение ({solution[0]:.4f}; {solution[1]:.4f})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Графики уравнений системы")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def plot_errors(errors_si, errors_newton, path):
    """Зависимость погрешности ||x(k) - x*|| от номера итерации (логарифмическая шкала)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for errors, label, style in ((errors_si, "метод простой итерации", "o-"),
                                 (errors_newton, "метод Ньютона", "s-")):
        points = [(k, e) for k, e in enumerate(errors) if e > 0]
        if points:
            ax.semilogy([k for k, _ in points], [e for _, e in points], style, label=label)
    ax.set_xlabel("номер итерации k")
    ax.set_ylabel("||x(k) - x*||")
    ax.set_title("Зависимость погрешности от количества итераций")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    args = [a for a in sys.argv[1:] if a != "--no-show"]
    show = "--no-show" not in sys.argv
    path = Path(args[0]) if args else DEFAULT_INPUT
    try:
        problem = Problem(read_input(path))
    except (OSError, ValueError) as e:
        print(f"Ошибка во входном файле {path}: {e}")
        sys.exit(1)
    n, eps = problem.n, problem.eps

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(exist_ok=True)
    curves_path = OUTPUT_DIR / f"2_2_{path.stem}_graph.png"
    errors_path = OUTPUT_DIR / f"2_2_{path.stem}_errors.png"

    print("Задание 2.2. Метод простой итерации и метод Ньютона для системы нелинейных уравнений")
    print(f"Входной файл: {path}")
    if problem.constants:
        print("Параметры: " + ", ".join(f"{k} = {v:g}" for k, v in problem.constants.items()))
    print(f"Система (n = {n}):")
    for i in range(n):
        print(f"  f{i + 1}(x) = {problem.f_text[i]} = 0")
    print(f"Точность: eps = {eps:g}")

    section("Начальное приближение (графически)")
    print("Область G, в которой лежит решение (для n = 2 — найдена по графику): "
          + " x ".join(f"[{lo:g}, {hi:g}]" for lo, hi in problem.region))
    x0 = [(lo + hi) / 2 for lo, hi in problem.region]
    print(f"Начальное приближение x(0) — центр G: {fmt_vector(x0, 4)}")

    section("Метод простой итерации")
    print("Система приведена к виду x = Phi(x):")
    for i in range(n):
        print(f"  x{i + 1} = phi{i + 1}(x) = {problem.phi_text[i]}")
    try:
        q, inside = check_simple_iteration(problem)
    except EVAL_ERRORS:
        print("Phi не определена в части области G — выберите другую Phi или область.")
        sys.exit(1)
    print(f"q = max ||Phi'(x)|| на G = {q:.6f}  ->  " + ("q < 1" if q < 1 else "q >= 1 (!)"))
    print("Phi(G) " + ("лежит в G" if inside else "НЕ лежит в G (!)"))
    if q < 1 and inside:
        print("Достаточные условия сходимости выполнены.")
        print("Критерий окончания: q/(1-q) * ||x(k) - x(k-1)|| <= eps")
    else:
        print("Достаточные условия сходимости НЕ выполнены — сходимость не гарантирована.")
        if q >= 1:
            print("Критерий окончания: ||x(k) - x(k-1)|| <= eps")
    x_si, hist_si, status_si = simple_iteration(problem, x0, q, eps)
    x_n, hist_n, status_n = newton(problem, x0, eps)

    x_star = None
    start = x_n if status_n == "converged" else x_si if status_si == "converged" else None
    if start is not None:
        x_ref, _, st = newton(problem, start, 1e-15)
        if st == "converged" or vector_norm(problem.F(x_ref)) < 1e-12:
            x_star = x_ref

    def errors_of(history):
        if x_star is None:
            return []
        return [vector_norm([item[0][i] - x_star[i] for i in range(n)]) for item in history]

    errors_si, errors_n = errors_of(hist_si), errors_of(hist_n)
    print()
    iteration_table(hist_si, errors_si, 1, True)
    print()
    print(f"Результат: {status_text(status_si)}, число итераций: {len(hist_si) - 1}")
    if status_si != "diverged":
        print(f"x = {fmt_vector(x_si)},  ||f(x)|| = {vector_norm(problem.F(x_si)):.2e}")

    section("Метод Ньютона")
    print("J(x(k)) * dx(k) = -f(x(k))  (решается методом Гаусса),  x(k+1) = x(k) + dx(k)")
    print("Матрица Якоби J_ij = df_i/dx_j:")
    print(f"J(x(0)) = {[[round(v, 4) for v in row] for row in problem.J(x0)]}")
    print("Критерий окончания: ||x(k) - x(k-1)|| <= eps")
    print()
    iteration_table(hist_n, errors_n, 2, False)
    print()
    print(f"Результат: {status_text(status_n)}, число итераций: {len(hist_n) - 1}")
    if status_n not in ("diverged", "singular"):
        print(f"x = {fmt_vector(x_n)},  ||f(x)|| = {vector_norm(problem.F(x_n)):.2e}")

    section("Анализ зависимости погрешности от количества итераций")
    if x_star is None:
        print("Ни один метод не сошёлся — анализ невозможен.")
    else:
        print(f"Точное решение (метод Ньютона до машинной точности): x* = {fmt_vector(x_star, 12)}")
        print()
        print("Число итераций, необходимое для разной точности:")
        rows = []
        for p in range(1, 13):
            e = 10.0 ** (-p)
            _, h1, s1 = simple_iteration(problem, x0, q, e)
            _, h2, s2 = newton(problem, x0, e)
            rows.append([f"1e-{p:02d}",
                         str(len(h1) - 1) if s1 == "converged" else "-",
                         str(len(h2) - 1) if s2 == "converged" else "-"])
        print_table(["eps", "простая итерация", "Ньютон"], rows)
        print()
        dphi = matrix_norm(numerical_jacobian(problem.Phi, x_star))
        print("Метод простой итерации сходится ЛИНЕЙНО: погрешность за итерацию уменьшается в среднем")
        print("в одно и то же число раз — отношение e(k)/e(k-1) держится примерно на одном уровне,")
        print(f"порядка ||Phi'(x*)|| = {dphi:.4f} (у систем оно может колебаться от итерации к итерации).")
        print("Метод Ньютона сходится КВАДРАТИЧНО: e(k) ~ C * e(k-1)^2, отношение e(k)/e(k-1)^2")
        print("почти постоянно; число верных знаков на каждой итерации примерно удваивается.")
        plot_errors(errors_si, errors_n, errors_path)
        print()
        print(f"График погрешности сохранён: {errors_path}")

    if n == 2:
        plot_curves(problem, x_star, curves_path)
        print(f"График кривых сохранён:      {curves_path}")
    else:
        print("График кривых строится только для системы из двух уравнений.")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
