import math
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "2_1" / "variant_01.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MAX_ITER = 1000
DIVERGENCE = 1e4
GRID_POINTS = 1001
H = 1e-6

MATH_NAMES = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tg": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "arcsin": math.asin, "arccos": math.acos, "arctg": math.atan,
    "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10, "lg": math.log10,
    "log2": math.log2, "sqrt": math.sqrt, "abs": abs, "pi": math.pi, "e": math.e,
}


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
    for key in ("eps", "f", "df", "phi", "interval"):
        if key not in params:
            raise ValueError(f"не задан параметр '{key}'")
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


def make_functions(params):
    """Создаёт функции f(x), f'(x) и phi(x) из выражений входного файла."""
    f_code = compile_expression(params["f"], "f")
    df_code = compile_expression(params["df"], "df")
    phi_code = compile_expression(params["phi"], "phi")
    env = {"__builtins__": {}, **MATH_NAMES}

    def evaluate(code, local):
        value = eval(code, env, local)
        if isinstance(value, complex):
            raise ValueError("значение вне области определения")
        return value

    def f(x):
        return evaluate(f_code, {"x": x})

    def df(x):
        return evaluate(df_code, {"x": x})

    def phi(x):
        local = {"x": x}
        if "f" in phi_code.co_names:
            local["f"] = f(x)
        return evaluate(phi_code, local)

    return f, df, phi


def grid(a, b, count=GRID_POINTS):
    return [a + (b - a) * i / (count - 1) for i in range(count)]


def derivative(func, x):
    """Численная производная (центральная разность) — только для проверки условий сходимости."""
    return (func(x + H) - func(x - H)) / (2 * H)


def second_derivative(func, x):
    return (func(x + H * 100) - 2 * func(x) + func(x - H * 100)) / (H * 100) ** 2


def check_simple_iteration(phi, a, b):
    """
    Достаточные условия сходимости метода простой итерации на [a, b]:
      1) phi(x) принадлежит [a, b] для всех x из [a, b];
      2) |phi'(x)| <= q < 1 на [a, b].
    Проверяются по сетке точек. Возвращает q, min phi, max phi.
    """
    q = 0.0
    values = []
    for x in grid(a, b):
        values.append(phi(x))
        q = max(q, abs(derivative(phi, x)))
    return q, min(values), max(values)


def simple_iteration(phi, x0, q, eps):
    """
    x(k+1) = phi(x(k)).
    Критерий окончания:  q/(1-q) * |x(k+1) - x(k)| <= eps  (при q < 1),
                         иначе |x(k+1) - x(k)| <= eps.
    Возвращает x, history = [(x(k), |x(k) - x(k-1)|, оценка погрешности)], status.
    """
    coef = q / (1 - q) if q < 1 else 1.0
    x = x0
    history = [(x0, None, None)]
    for _ in range(MAX_ITER):
        try:
            x_new = phi(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return x, history, "diverged"
        diff = abs(x_new - x)
        history.append((x_new, diff, coef * diff))
        x = x_new
        if coef * diff <= eps:
            return x, history, "converged"
        if not math.isfinite(diff) or diff > DIVERGENCE * history[1][1]:
            return x, history, "diverged"
    return x, history, "max_iter"


def choose_newton_x0(f, a, b):
    """Начальное приближение для Ньютона: тот конец отрезка, где f(x0) * f''(x0) > 0."""
    for x0 in (a, b):
        if f(x0) * second_derivative(f, x0) > 0:
            return x0, True
    return (a + b) / 2, False


def newton(f, df, x0, eps):
    """
    x(k+1) = x(k) - f(x(k)) / f'(x(k)).
    Критерий окончания: |x(k+1) - x(k)| <= eps.
    Возвращает x, history = [(x(k), |x(k) - x(k-1)|)], status.
    """
    x = x0
    history = [(x0, None)]
    for _ in range(MAX_ITER):
        try:
            d = df(x)
            if d == 0:
                return x, history, "zero_derivative"
            x_new = x - f(x) / d
        except (ValueError, ZeroDivisionError, OverflowError):
            return x, history, "diverged"
        diff = abs(x_new - x)
        history.append((x_new, diff))
        x = x_new
        if diff <= eps:
            return x, history, "converged"
        if not math.isfinite(diff) or diff > DIVERGENCE * history[1][1]:
            return x, history, "diverged"
    return x, history, "max_iter"


def print_table(headers, rows):
    widths = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 70 - len(title)))


def error_ratio(errors, k, power):
    """e(k) / e(k-1)^power; около машинной точности отношение теряет смысл — тогда '-'."""
    if k == 0 or errors[k] < 1e-13 or errors[k - 1] < 1e-13:
        return "-"
    return f"{errors[k] / errors[k - 1] ** power:.4f}"


def status_text(status):
    return {"converged": "точность достигнута",
            "diverged": "метод РАСХОДИТСЯ",
            "max_iter": f"точность не достигнута за {MAX_ITER} итераций",
            "zero_derivative": "f'(x) = 0 — метод Ньютона неприменим"}[status]


def plot_graphs(f, phi, plot_range, a, b, root, title, path):
    """График f(x) (для выбора отрезка с корнем) и график y = phi(x), y = x на [a, b]."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xs = grid(plot_range[0], plot_range[1], 2001)
    ys = []
    for x in xs:
        try:
            ys.append(f(x))
        except (ValueError, ZeroDivisionError, OverflowError):
            ys.append(float("nan"))
    finite = sorted(y for y in ys if math.isfinite(y))
    if finite:
        lo, hi = finite[len(finite) // 50], finite[-len(finite) // 50 - 1]
        pad = (hi - lo) * 0.2 + 1e-9
        ax1.set_ylim(lo - pad, hi + pad)
    ax1.plot(xs, ys, label="y = f(x)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axvspan(a, b, color="orange", alpha=0.2, label=f"[a, b] = [{a:g}, {b:g}]")
    if root is not None:
        ax1.plot([root], [0], "ro", label=f"корень x* = {root:.6f}")
    ax1.set_title(title)
    ax1.set_xlabel("x")
    ax1.grid(True)
    ax1.legend()

    xs = grid(a, b, 501)
    ax2.plot(xs, [phi(x) for x in xs], label="y = phi(x)")
    ax2.plot(xs, xs, "--", label="y = x")
    if root is not None:
        ax2.plot([root], [root], "ro", label="x* = phi(x*)")
    ax2.set_title("Метод простой итерации: x = phi(x) на [a, b]")
    ax2.set_xlabel("x")
    ax2.grid(True)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def plot_errors(errors_si, errors_newton, path):
    """Зависимость погрешности |x(k) - x*| от номера итерации (логарифмическая шкала)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for errors, label, style in ((errors_si, "метод простой итерации", "o-"),
                                 (errors_newton, "метод Ньютона", "s-")):
        points = [(k, e) for k, e in enumerate(errors) if e > 0]
        if points:
            ax.semilogy([k for k, _ in points], [e for _, e in points], style, label=label)
    ax.set_xlabel("номер итерации k")
    ax.set_ylabel("|x(k) - x*|")
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
        params = read_input(path)
        eps = float(params["eps"])
        if eps <= 0:
            raise ValueError("eps должно быть положительным")
        a, b = numbers(params["interval"], 2, "interval")
        if a >= b:
            raise ValueError("interval: должно быть a < b")
        plot_range = numbers(params["plot"], 2, "plot") if "plot" in params else [a - 1, b + 1]
        f, df, phi = make_functions(params)
    except (OSError, ValueError) as e:
        print(f"Ошибка во входном файле {path}: {e}")
        sys.exit(1)

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(exist_ok=True)
    graph_path = OUTPUT_DIR / f"2_1_{path.stem}_graph.png"
    errors_path = OUTPUT_DIR / f"2_1_{path.stem}_errors.png"

    print("Задание 2.1. Метод простой итерации и метод Ньютона для уравнения f(x) = 0")
    print(f"Входной файл: {path}")
    print(f"f(x)   = {params['f']}")
    print(f"f'(x)  = {params['df']}")
    print(f"phi(x) = {params['phi']}")
    print(f"Точность: eps = {eps:g}")

    section("Отделение корня (графически)")
    print(f"По графику выбран отрезок [a, b] = [{a:g}, {b:g}]")
    try:
        fa, fb = f(a), f(b)
    except (ValueError, ZeroDivisionError, OverflowError):
        fa = fb = float("nan")
    print(f"f(a) = {fa:.6f},  f(b) = {fb:.6f}")
    if not fa * fb < 0:
        plot_graphs(f, phi, plot_range, a, b, None, f"f(x) = {params['f']}", graph_path)
        print(f"f(a)*f(b) >= 0 — на отрезке нет смены знака. Посмотрите график {graph_path}")
        print("и задайте в файле другой interval.")
        if show:
            plt.show()
        sys.exit(1)
    print("f(a)*f(b) < 0 -> на отрезке есть корень.")

    # ----------------------------------------------------------- метод простой итерации
    section("Метод простой итерации")
    print(f"Уравнение приведено к виду x = phi(x),  phi(x) = {params['phi']}")
    try:
        q, phi_min, phi_max = check_simple_iteration(phi, a, b)
    except (ValueError, ZeroDivisionError, OverflowError):
        print("phi(x) не определена на части отрезка [a, b] — выберите другую phi или отрезок.")
        sys.exit(1)
    inside = a <= phi_min and phi_max <= b
    print(f"q = max |phi'(x)| на [a, b] = {q:.6f}  ->  " + ("q < 1" if q < 1 else "q >= 1 (!)"))
    print(f"phi([a, b]) = [{phi_min:.6f}, {phi_max:.6f}]  ->  "
          + ("лежит в [a, b]" if inside else "НЕ лежит в [a, b] (!)"))
    if q < 1 and inside:
        print("Достаточные условия сходимости выполнены.")
        print("Критерий окончания: q/(1-q) * |x(k) - x(k-1)| <= eps")
    else:
        print("Достаточные условия сходимости НЕ выполнены — сходимость не гарантирована.")
        if q >= 1:
            print("Критерий окончания: |x(k) - x(k-1)| <= eps")
    x0_si = (a + b) / 2
    print(f"Начальное приближение x(0) = (a + b)/2 = {x0_si:g}")
    x_si, hist_si, status_si = simple_iteration(phi, x0_si, q, eps)

    # ----------------------------------------------------------- метод Ньютона (считаем до вывода,
    # чтобы получить точное значение корня x* для анализа погрешности)
    x0_n, fourier = choose_newton_x0(f, a, b)
    x_n, hist_n, status_n = newton(f, df, x0_n, eps)

    # Эталонный корень x*: продолжаем метод Ньютона до машинной точности
    x_star = None
    if status_n == "converged" or status_si == "converged":
        x_ref, _, st = newton(f, df, x_n if status_n == "converged" else x_si, 1e-15)
        if st == "converged" or abs(f(x_ref)) < 1e-12:
            x_star = x_ref

    errors_si = [abs(x - x_star) if x_star is not None else 0.0 for x, _, _ in hist_si]
    rows = []
    for k, (x, diff, est) in enumerate(hist_si):
        rows.append([str(k), f"{x:.10f}", "-" if diff is None else f"{diff:.2e}",
                     "-" if est is None else f"{est:.2e}",
                     f"{errors_si[k]:.2e}" if x_star is not None else "-",
                     error_ratio(errors_si, k, 1) if x_star is not None else "-"])
    print()
    print_table(["k", "x(k)", "|x(k)-x(k-1)|", "оценка погр.", "|x(k)-x*|", "e(k)/e(k-1)"], rows)
    print()
    print(f"Результат: {status_text(status_si)}, число итераций: {len(hist_si) - 1}")
    if status_si != "diverged":
        print(f"x = {x_si:.10f},  f(x) = {f(x_si):.2e}")

    section("Метод Ньютона")
    print("x(k+1) = x(k) - f(x(k)) / f'(x(k)),   критерий окончания: |x(k) - x(k-1)| <= eps")
    if fourier:
        print(f"Начальное приближение x(0) = {x0_n:g}: конец отрезка, где f(x0)*f''(x0) > 0 "
              f"(f(x0) = {f(x0_n):.4f}, f''(x0) = {second_derivative(f, x0_n):.4f})")
    else:
        print(f"Ни на одном конце отрезка f(x0)*f''(x0) > 0 не выполнено, x(0) = (a + b)/2 = {x0_n:g}")
    errors_n = [abs(x - x_star) if x_star is not None else 0.0 for x, _ in hist_n]
    rows = []
    for k, (x, diff) in enumerate(hist_n):
        rows.append([str(k), f"{x:.10f}", "-" if diff is None else f"{diff:.2e}",
                     f"{errors_n[k]:.2e}" if x_star is not None else "-",
                     error_ratio(errors_n, k, 2) if x_star is not None else "-"])
    print()
    print_table(["k", "x(k)", "|x(k)-x(k-1)|", "|x(k)-x*|", "e(k)/e(k-1)^2"], rows)
    print()
    print(f"Результат: {status_text(status_n)}, число итераций: {len(hist_n) - 1}")
    if status_n not in ("diverged", "zero_derivative"):
        print(f"x = {x_n:.10f},  f(x) = {f(x_n):.2e}")

    # ----------------------------------------------------------- анализ погрешности
    section("Анализ зависимости погрешности от количества итераций")
    if x_star is None:
        print("Ни один метод не сошёлся — анализ невозможен.")
    else:
        print(f"Точное значение корня (метод Ньютона до машинной точности): x* = {x_star:.15f}")
        print()
        print("Число итераций, необходимое для разной точности:")
        rows = []
        for p in range(1, 13):
            e = 10.0 ** (-p)
            _, h1, s1 = simple_iteration(phi, x0_si, q, e)
            _, h2, s2 = newton(f, df, x0_n, e)
            rows.append([f"1e-{p:02d}",
                         str(len(h1) - 1) if s1 == "converged" else "-",
                         str(len(h2) - 1) if s2 == "converged" else "-"])
        print_table(["eps", "простая итерация", "Ньютон"], rows)
        print()
        dphi = abs(derivative(phi, x_star))
        c_newton = abs(second_derivative(f, x_star) / (2 * df(x_star)))
        print("Метод простой итерации сходится ЛИНЕЙНО: |x(k) - x*| ~ |phi'(x*)| * |x(k-1) - x*|,")
        print(f"отношение e(k)/e(k-1) стремится к |phi'(x*)| = {dphi:.4f}; на каждый новый верный")
        if 0 < dphi < 1:
            print(f"десятичный знак нужно примерно {1 / -math.log10(dphi):.1f} итераций.")
        else:
            print("десятичный знак нужно несколько итераций.")
        print("Метод Ньютона сходится КВАДРАТИЧНО: |x(k) - x*| ~ C * |x(k-1) - x*|^2,")
        print(f"C = |f''(x*) / (2 f'(x*))| = {c_newton:.4f}; число верных знаков на каждой итерации")
        print("примерно удваивается, поэтому итераций нужно гораздо меньше.")
        plot_errors(errors_si, errors_n, errors_path)
        print()
        print(f"График погрешности сохранён: {errors_path}")

    plot_graphs(f, phi, plot_range, a, b, x_star, f"f(x) = {params['f']}", graph_path)
    print(f"График функции сохранён:     {graph_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
