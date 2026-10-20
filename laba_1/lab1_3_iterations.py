import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "1_3" / "variant_17.txt"

MAX_ITER = 1000
DIVERGENCE = 1e4
PREC = 4
MAX_TABLE_ROWS = 20



def read_system(path):
    """Читает точность eps и расширенную матрицу [A | b] из текстового файла."""
    eps = None
    rows = []
    # utf-8-sig: файл, сохранённый в Блокноте с BOM, тоже прочитается
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.lower().startswith("eps"):
                try:
                    eps = float(line[3:].replace("=", " ").strip())
                except ValueError:
                    raise ValueError(f"строка {line_no}: не удалось прочитать eps") from None
                continue
            try:
                rows.append([float(t) for t in line.replace("|", " ").split()])
            except ValueError:
                raise ValueError(f"строка {line_no}: не удалось прочитать числа") from None

    if eps is None:
        raise ValueError('не задана точность — добавьте строку вида "eps = 0.01"')
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")
    n = len(rows)
    if n == 0:
        raise ValueError("в файле нет ни одного уравнения")
    for i, row in enumerate(rows, start=1):
        if len(row) != n + 1:
            raise ValueError(f"в уравнении {i} количество чисел {len(row)}, а должно быть "
                             f"n + 1 = {n + 1} (n коэффициентов и свободный член)")

    M = np.array(rows)
    return M[:, :n], M[:, n], eps


def vector_norm(v):
    """||v|| = max |v_i|."""
    return max(abs(t) for t in v)


def matrix_norm(M):
    """||M|| = max_i sum_j |m_ij| (максимальная сумма модулей по строкам)."""
    return max(sum(abs(t) for t in row) for row in M)


def to_iteration_form(A, b):
    """
    Приведение A*x = b к виду x = alpha*x + beta: из i-го уравнения выражаем x_i.
        alpha_ij = -a_ij / a_ii (j != i),  alpha_ii = 0,  beta_i = b_i / a_ii.
    """
    n = len(b)
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError(f"a_{i + 1}{i + 1} = 0: из {i + 1}-го уравнения нельзя выразить "
                             f"x_{i + 1}, переставьте уравнения")
        alpha[i] = -A[i] / A[i, i]
        alpha[i, i] = 0.0
        beta[i] = b[i] / A[i, i]
    return alpha, beta


def simple_iteration_step(alpha, beta, x):
    """МПИ: x^(k+1) = beta + alpha*x^(k) — все компоненты считаются по старому вектору x^(k)."""
    return beta + alpha @ x


def seidel_step(alpha, beta, x):
    """
    Метод Зейделя: x_i^(k+1) = beta_i + sum_(j<i) alpha_ij*x_j^(k+1) + sum_(j>i) alpha_ij*x_j^(k).
    Уже найденные на этой итерации компоненты x_1^(k+1) ... x_(i-1)^(k+1) сразу используются.
    """
    n = len(beta)
    x_new = x.copy()
    for i in range(n):
        x_new[i] = beta[i] + alpha[i, :i] @ x_new[:i] + alpha[i, i + 1:] @ x[i + 1:]
    return x_new


def iterate(step, alpha, beta, eps, coef):
    """
    Итерации x^(k+1) = step(x^(k)) начиная с x^(0) = beta, пока не выполнится
    критерий остановки  eps^(k) = coef * ||x^(k) - x^(k-1)|| <= eps.

    Возвращает x, history и status:
      history — список (x^(k), ||x^(k) - x^(k-1)||, eps^(k)) для k = 1, 2, ...
      status  — "converged", "diverged" или "max_iter".
    """
    x = beta.copy()
    history = []
    for _ in range(MAX_ITER):
        x_new = step(alpha, beta, x)
        diff = vector_norm(x_new - x)
        est = coef * diff
        history.append((x_new, diff, est))
        x = x_new
        if est <= eps:
            return x, history, "converged"
        if not math.isfinite(diff) or diff > DIVERGENCE * history[0][1]:
            return x, history, "diverged"
    return x, history, "max_iter"


def a_priori_iterations(norm_alpha, norm_beta, eps):
    """
    Априорная оценка числа итераций МПИ при x^(0) = beta:
        ||x^(k) - x*|| <= ||alpha||^(k+1) / (1 - ||alpha||) * ||beta|| <= eps
        => k + 1 >= (lg eps - lg ||beta|| + lg(1 - ||alpha||)) / lg ||alpha||.
    """
    if not 0 < norm_alpha < 1 or norm_beta == 0:
        return None
    k_plus_1 = (math.log10(eps) - math.log10(norm_beta) + math.log10(1 - norm_alpha)) \
        / math.log10(norm_alpha)
    return max(0, math.ceil(k_plus_1 - 1))



def fmt(v, prec=PREC):
    if abs(v) < 0.5 * 10 ** (-prec):
        v = 0.0
    return f"{v:.{prec}f}"


def print_matrix(name, M, prec=PREC):
    cells = [[fmt(v, prec) for v in row] for row in M]
    width = max(len(c) for row in cells for c in row)
    print(f"{name}:")
    for row in cells:
        print("  [ " + "  ".join(c.rjust(width) for c in row) + " ]")


def print_vector(name, v, prec=PREC):
    print(f"{name} = ( " + "; ".join(fmt(t, prec) for t in v) + " )")


def print_table(headers, rows):
    """Печатает таблицу строк; если строк много — середина заменяется на '...'."""
    widths = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]
    if len(rows) > MAX_TABLE_ROWS:
        rows = rows[:MAX_TABLE_ROWS - 5] + [["..."] * len(headers)] + rows[-5:]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))


def print_iterations(history, beta, x_ref):
    """Таблица итераций: компоненты x^(k) (если n <= 6), ||x^(k)-x^(k-1)||, оценка eps^(k)
    и фактическая погрешность ||x^(k) - x*|| относительно эталонного решения."""
    n = len(beta)
    show_x = n <= 6
    headers = ["k"] + ([f"x{i + 1}" for i in range(n)] if show_x else []) \
        + ["||x(k)-x(k-1)||", "eps(k)"] + (["||x(k)-x*||"] if x_ref is not None else [])

    def row(k, x, diff, est):
        r = [str(k)] + ([fmt(t) for t in x] if show_x else []) + [diff, est]
        if x_ref is not None:
            r.append(f"{vector_norm(x - x_ref):.2e}")
        return r

    rows = [row(0, beta, "-", "-")]
    for k, (x, diff, est) in enumerate(history, start=1):
        rows.append(row(k, x, f"{diff:.2e}", f"{est:.2e}"))
    print_table(headers, rows)


def describe_result(name, x, history, status, eps, x_ref):
    print()
    if status == "converged":
        print(f"{name}: точность eps = {eps:g} достигнута. Число итераций: {len(history)}")
    elif status == "diverged":
        print(f"{name}: метод РАСХОДИТСЯ — шаг ||x(k) - x(k-1)|| вырос более чем в {DIVERGENCE:g} раз, "
              f"остановлено на итерации {len(history)}.")
    else:
        print(f"{name}: точность не достигнута, выполнено итераций: {MAX_ITER}")
    if status != "diverged":
        print("Решение:")
        for i, t in enumerate(x):
            print(f"  x{i + 1} = {fmt(t, 6)}")
        if x_ref is not None:
            print(f"Фактическая погрешность ||x - x*|| = {vector_norm(x - x_ref):.2e}")


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 70 - len(title)))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    try:
        A, b, eps = read_system(path)
        alpha, beta = to_iteration_form(A, b)
    except (OSError, ValueError) as e:
        print(f"Ошибка во входном файле {path}: {e}")
        sys.exit(1)
    n = len(b)

    print("Задание 1.3. Метод простых итераций и метод Зейделя")
    print(f"Входной файл: {path}")
    print(f"Размерность системы: n = {n}")
    print(f"Заданная точность: eps = {eps:g}")
    print()
    print_matrix("Матрица A", A)
    print_vector("b", b)

    
    try:
        x_ref = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x_ref = None

    section("Приведение к виду x = alpha*x + beta")
    print("alpha_ij = -a_ij / a_ii (j != i),  alpha_ii = 0,  beta_i = b_i / a_ii")
    print_matrix("alpha", alpha)
    print_vector("beta", beta)

    norm_alpha = matrix_norm(alpha)
    norm_C = matrix_norm(np.triu(alpha, 1))
    print()
    print(f"||alpha|| = max_i sum_j |alpha_ij| = {norm_alpha:.4f}")
    print(f"||C|| = {norm_C:.4f}  (C — часть alpha выше диагонали, нужна для оценки метода Зейделя)")

    if norm_alpha < 1:
        print("||alpha|| < 1 -> достаточное условие сходимости выполнено (диагональное преобладание).")
        coef_simple = norm_alpha / (1 - norm_alpha)
        coef_seidel = norm_C / (1 - norm_alpha)
        print("Критерий остановки:")
        print(f"  МПИ:     eps(k) = ||alpha|| / (1 - ||alpha||) * ||x(k) - x(k-1)|| "
              f"= {coef_simple:.4f} * ||x(k) - x(k-1)|| <= eps")
        print(f"  Зейдель: eps(k) = ||C|| / (1 - ||alpha||) * ||x(k) - x(k-1)|| "
              f"= {coef_seidel:.4f} * ||x(k) - x(k-1)|| <= eps")
        k_prior = a_priori_iterations(norm_alpha, vector_norm(beta), eps)
        if k_prior is not None:
            print(f"Априорная оценка: для МПИ заведомо хватит k = {k_prior} итераций.")
    else:
        print("||alpha|| >= 1 -> достаточное условие сходимости НЕ выполнено:")
        print("сходимость не гарантирована (но возможна), оценку погрешности построить нельзя.")
        coef_simple = coef_seidel = 1.0
        print("Критерий остановки для обоих методов: eps(k) = ||x(k) - x(k-1)|| <= eps")

    section("Метод простых итераций")
    print("x(0) = beta,   x(k+1) = beta + alpha * x(k)")
    print()
    x_s, hist_s, status_s = iterate(simple_iteration_step, alpha, beta, eps, coef_simple)
    print_iterations(hist_s, beta, x_ref)
    describe_result("МПИ", x_s, hist_s, status_s, eps, x_ref)

    section("Метод Зейделя")
    print("x(0) = beta,   x_i(k+1) = beta_i + sum_(j<i) alpha_ij*x_j(k+1) + sum_(j>i) alpha_ij*x_j(k)")
    print()
    x_z, hist_z, status_z = iterate(seidel_step, alpha, beta, eps, coef_seidel)
    print_iterations(hist_z, beta, x_ref)
    describe_result("Зейдель", x_z, hist_z, status_z, eps, x_ref)

    section("Эталонное решение (numpy.linalg.solve, только для проверки)")
    if x_ref is None:
        print("Матрица вырождена — эталонного решения нет.")
    else:
        for i, t in enumerate(x_ref):
            print(f"  x*{i + 1} = {fmt(t, 6)}")

    section("Сравнение методов по скорости сходимости")
    print("Число итераций для разной точности:")
    rows = []
    for p in range(5):
        e = eps * 10 ** (-p)
        cells = [f"{e:g}"]
        for step, coef in ((simple_iteration_step, coef_simple), (seidel_step, coef_seidel)):
            _, hist, status = iterate(step, alpha, beta, e, coef)
            cells.append(str(len(hist)) if status == "converged"
                         else "расходится" if status == "diverged" else f">{MAX_ITER}")
        rows.append(cells)
    print_table(["eps", "МПИ", "Зейдель"], rows)

    print()
    if status_s == "converged" and status_z == "converged":
        k_s, k_z = len(hist_s), len(hist_z)
        print(f"При eps = {eps:g} число итераций: МПИ — {k_s}, Зейдель — {k_z}.")
        if k_z < k_s:
            print(f"Метод Зейделя сошёлся быстрее в {k_s / k_z:.2f} раза: на каждой итерации он сразу")
            print("использует уже уточнённые компоненты x_j(k+1), j < i.")
        elif k_z > k_s:
            print(f"В этой системе МПИ сошёлся быстрее в {k_z / k_s:.2f} раза.")
        else:
            print("Оба метода потребовали одинаковое число итераций.")
    else:
        for name, status in (("МПИ", status_s), ("Зейдель", status_z)):
            if status != "converged":
                print(f"{name} не достиг заданной точности — сравнение по числу итераций невозможно.")


if __name__ == "__main__":
    main()
