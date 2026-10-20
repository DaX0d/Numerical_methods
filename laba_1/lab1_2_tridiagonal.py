import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "1_2" / "variant_17.txt"

EPS = 1e-12
CHECK_TOL = 1e-9
PREC = 6


def read_tridiagonal(path):
    """Читает диагонали a, b, c и правую часть d (каждая — список длины n)."""
    a, b, c, d = [], [], [], []
    # utf-8-sig: файл, сохранённый в Блокноте с BOM, тоже прочитается
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].replace("|", " ").strip()
            if not line:
                continue
            try:
                nums = [float(t) for t in line.split()]
            except ValueError:
                raise ValueError(f"строка {line_no}: не удалось прочитать числа") from None
            if len(nums) != 4:
                raise ValueError(f"строка {line_no}: ожидалось 4 числа (a_i b_i c_i | d_i), "
                                 f"получено {len(nums)}")
            a.append(nums[0])
            b.append(nums[1])
            c.append(nums[2])
            d.append(nums[3])

    if not d:
        raise ValueError("в файле нет ни одного уравнения")
    if a[0] != 0:
        raise ValueError("a_1 должно быть равно 0: в первом уравнении нет x_0")
    if c[-1] != 0:
        raise ValueError("c_n должно быть равно 0: в последнем уравнении нет x_(n+1)")
    return a, b, c, d


def check_diagonal_dominance(a, b, c):
    """
    Достаточное условие корректности и устойчивости прогонки:
    |b_i| >= |a_i| + |c_i| для всех i, причём хотя бы для одного i неравенство строгое.
    Возвращает список номеров строк, где условие нарушено, и признак строгого неравенства.
    """
    n = len(b)
    bad_rows = [i + 1 for i in range(n) if abs(b[i]) < abs(a[i]) + abs(c[i])]
    has_strict = any(abs(b[i]) > abs(a[i]) + abs(c[i]) for i in range(n))
    return bad_rows, has_strict


def tridiagonal_solve(a, b, c, d):
    """
    Метод прогонки. Решение ищется в виде x_i = P_i*x_(i+1) + Q_i.

    Прямой ход — прогоночные коэффициенты:
        P_i = -c_i / (b_i + a_i*P_(i-1)),   Q_i = (d_i - a_i*Q_(i-1)) / (b_i + a_i*P_(i-1)),
        при i = 1 (a_1 = 0):  P_1 = -c_1 / b_1,  Q_1 = d_1 / b_1.
    Обратный ход:
        x_n = Q_n  (т.к. c_n = 0 => P_n = 0),   x_i = P_i*x_(i+1) + Q_i,  i = n-1, ..., 1.

    Возвращает x, P, Q и знаменатели b_i + a_i*P_(i-1) (для вывода).
    """
    n = len(d)
    P = [0.0] * n
    Q = [0.0] * n
    denoms = [0.0] * n
    tol = EPS * max(abs(v) for v in b)

    # Прямой ход. P_prev, Q_prev — это P_(i-1), Q_(i-1); для i = 1 берём 0,
    # тогда общая формула сама превращается в P_1 = -c_1/b_1, Q_1 = d_1/b_1.
    P_prev, Q_prev = 0.0, 0.0
    for i in range(n):
        denom = b[i] + a[i] * P_prev
        if abs(denom) <= tol:
            raise ZeroDivisionError(f"знаменатель в уравнении {i + 1} равен нулю, "
                                    f"метод прогонки неприменим")
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q_prev) / denom
        denoms[i] = denom
        P_prev, Q_prev = P[i], Q[i]

    # Обратный ход
    x = [0.0] * n
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x, P, Q, denoms


def residual(a, b, c, d, x):
    """Невязка r_i = a_i*x_(i-1) + b_i*x_i + c_i*x_(i+1) - d_i (считается тоже только по диагоналям)."""
    n = len(d)
    r = []
    for i in range(n):
        s = b[i] * x[i]
        if i > 0:
            s += a[i] * x[i - 1]
        if i < n - 1:
            s += c[i] * x[i + 1]
        r.append(s - d[i])
    return r


# --------------------------------------------------------------------------- вывод

def fmt(v, prec=PREC):
    """prec=None — компактный вид (для входных данных: 5, -2.5), иначе prec знаков после запятой."""
    if prec is None:
        return f"{v:g}"
    if abs(v) < 0.5 * 10 ** (-prec):
        v = 0.0
    return f"{v:.{prec}f}"


def print_table(headers, rows, prec=PREC):
    """Печатает таблицу: первый столбец — номер i, остальные — числа."""
    cells = [[str(i)] + [fmt(v, prec) for v in row] for i, row in rows]
    widths = [max(len(h), *(len(r[j]) for r in cells)) for j, h in enumerate(headers)]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in cells:
        print("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 70 - len(title)))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    try:
        a, b, c, d = read_tridiagonal(path)
    except (OSError, ValueError) as e:
        print(f"Ошибка чтения входного файла {path}: {e}")
        sys.exit(1)
    n = len(d)

    print("Задание 1.2. Метод прогонки")
    print(f"Входной файл: {path}")
    print(f"Размерность системы: n = {n}")
    print()
    print("Коэффициенты системы  a_i*x_(i-1) + b_i*x_i + c_i*x_(i+1) = d_i:")
    print_table(["i", "a_i", "b_i", "c_i", "d_i"],
                [(i + 1, (a[i], b[i], c[i], d[i])) for i in range(n)], prec=None)

    section("Достаточное условие устойчивости: |b_i| >= |a_i| + |c_i|")
    print_table(["i", "|b_i|", "|a_i|+|c_i|"],
                [(i + 1, (abs(b[i]), abs(a[i]) + abs(c[i]))) for i in range(n)], prec=None)
    bad_rows, has_strict = check_diagonal_dominance(a, b, c)
    if not bad_rows and has_strict:
        print("Условие выполнено (и хотя бы в одной строке неравенство строгое) -> |P_i| <= 1,")
        print("метод корректен и устойчив.")
    else:
        if bad_rows:
            print(f"Условие нарушено в уравнениях: {bad_rows}.")
        if not has_strict:
            print("Нет ни одной строки со строгим неравенством.")
        print("Прогонка может оказаться неустойчивой или неприменимой, но попробуем её выполнить.")

    try:
        x, P, Q, denoms = tridiagonal_solve(a, b, c, d)
    except ZeroDivisionError as e:
        print()
        print(f"Ошибка: {e}")
        sys.exit(1)

    section("Прямой ход: прогоночные коэффициенты")
    print("P_1 = -c_1 / b_1,   Q_1 = d_1 / b_1")
    print("P_i = -c_i / (b_i + a_i*P_(i-1)),   Q_i = (d_i - a_i*Q_(i-1)) / (b_i + a_i*P_(i-1))")
    print()
    print_table(["i", "b_i + a_i*P_(i-1)", "P_i", "Q_i"],
                [(i + 1, (denoms[i], P[i], Q[i])) for i in range(n)])
    print(f"max |P_i| = {fmt(max(abs(p) for p in P))}")

    section("Обратный ход: решение системы")
    print("x_n = Q_n,   x_i = P_i*x_(i+1) + Q_i,  i = n-1, ..., 1")
    print()
    width = max(len(fmt(v)) for v in x)
    for i in range(n):
        name = f"x{i + 1}".ljust(len(f"x{n}"))
        print(f"  {name} = {fmt(x[i]).rjust(width)}")

    section("Проверка: невязка r_i = a_i*x_(i-1) + b_i*x_i + c_i*x_(i+1) - d_i")
    err = max(abs(v) for v in residual(a, b, c, d, x))
    verdict = "решение верное" if err < CHECK_TOL else "решение НЕ сходится с системой"
    print(f"max |r_i| = {err:.2e}  ->  {verdict} (допуск {CHECK_TOL:g})")


if __name__ == "__main__":
    main()
