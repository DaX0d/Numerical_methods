import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "1_4" / "variant_17.txt"

MAX_ITER = 10000
CHECK_TOL = 1e-9
PREC = 4


def read_matrix(path):
    """Читает точность eps и квадратную симметричную матрицу A из текстового файла."""
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
                rows.append([float(t) for t in line.split()])
            except ValueError:
                raise ValueError(f"строка {line_no}: не удалось прочитать числа") from None

    if eps is None:
        raise ValueError('не задана точность — добавьте строку вида "eps = 0.01"')
    if eps <= 0:
        raise ValueError("точность eps должна быть положительной")
    n = len(rows)
    if n == 0:
        raise ValueError("в файле нет матрицы")
    for i, row in enumerate(rows, start=1):
        if len(row) != n:
            raise ValueError(f"матрица должна быть квадратной: в строке {i} количество чисел "
                             f"{len(row)}, а строк {n}")

    A = np.array(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != A[j, i]:
                raise ValueError(f"матрица не симметрична (a_{i + 1}{j + 1} != a_{j + 1}{i + 1}), "
                                 f"метод вращений применим только к симметричным матрицам")
    return A, eps


def off_diagonal_norm(A):
    """t(A) = sqrt( sum_(l<m) a_lm^2 ) — величина внедиагональной части, критерий остановки."""
    n = len(A)
    s = 0.0
    for l in range(n):
        for m in range(l + 1, n):
            s += A[l, m] ** 2
    return math.sqrt(s)


def find_max_off_diagonal(A):
    """Индексы (i, j), i < j, максимального по модулю внедиагонального элемента."""
    n = len(A)
    i_max, j_max = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > abs(A[i_max, j_max]):
                i_max, j_max = i, j
    return i_max, j_max


def rotation_angle(A, i, j):
    """
    Угол phi, при котором элемент (i, j) матрицы U^T * A * U становится нулевым:
        tg(2*phi) = 2*a_ij / (a_ii - a_jj),   при a_ii = a_jj  phi = pi/4.
    """
    if A[i, i] == A[j, j]:
        return math.pi / 4
    return 0.5 * math.atan(2 * A[i, j] / (A[i, i] - A[j, j]))


def rotation_matrix(n, i, j, phi):
    """Матрица вращения U: единичная, кроме u_ii = u_jj = cos(phi), u_ij = -sin(phi), u_ji = sin(phi)."""
    U = np.eye(n)
    c, s = math.cos(phi), math.sin(phi)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -s
    U[j, i] = s
    return U


def jacobi_rotations(A, eps):
    """
    Метод вращений Якоби.
    Возвращает:
      eigenvalues — собственные значения (диагональ итоговой матрицы),
      V           — матрица собственных векторов (собственный вектор h_i — i-й столбец V),
      Ak          — итоговая почти диагональная матрица,
      history     — список (i, j, a_ij, phi, t(A(k+1))) по итерациям.
    """
    n = len(A)
    Ak = A.astype(float)
    V = np.eye(n)
    history = []

    t = off_diagonal_norm(Ak)
    while t >= eps:
        if len(history) == MAX_ITER:
            raise RuntimeError(f"точность не достигнута за {MAX_ITER} итераций")
        # 1) максимальный по модулю внедиагональный элемент
        i, j = find_max_off_diagonal(Ak)
        a_ij = Ak[i, j]
        # 2) угол и матрица вращения, обнуляющего a_ij
        phi = rotation_angle(Ak, i, j)
        U = rotation_matrix(n, i, j, phi)
        # 3) преобразование подобия и накопление собственных векторов
        Ak = U.T @ Ak @ U
        V = V @ U
        # 4) критерий остановки
        t = off_diagonal_norm(Ak)
        history.append((i, j, a_ij, phi, t))

    eigenvalues = [Ak[i, i] for i in range(n)]
    return eigenvalues, V, Ak, history


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


def print_table(headers, rows):
    widths = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 70 - len(title)))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    try:
        A, eps = read_matrix(path)
    except (OSError, ValueError) as e:
        print(f"Ошибка во входном файле {path}: {e}")
        sys.exit(1)
    n = len(A)

    print("Задание 1.4. Метод вращений Якоби")
    print(f"Входной файл: {path}")
    print(f"Размерность матрицы: n = {n}")
    print(f"Заданная точность: eps = {eps:g}")
    print()
    print_matrix("Матрица A (симметричная)", A)

    eigenvalues, V, Ak, history = jacobi_rotations(A, eps)

    section("Итерации метода вращений")
    print("(i, j) — максимальный по модулю внедиагональный элемент a_ij матрицы A(k),")
    print("phi    — угол вращения: tg(2*phi) = 2*a_ij / (a_ii - a_jj), после вращения a_ij = 0,")
    print("t      — критерий остановки t(A(k+1)) = sqrt(sum_(l<m) a_lm^2) < eps.")
    print()
    rows = [["0", "-", "-", "-", f"{off_diagonal_norm(A):.6f}"]]
    for k, (i, j, a_ij, phi, t) in enumerate(history, start=1):
        rows.append([str(k), f"({i + 1}, {j + 1})", fmt(a_ij), f"{phi:.6f}", f"{t:.6f}"])
    print_table(["k", "(i, j)", "a_ij", "phi", "t(A(k))"], rows)
    print()
    print(f"Точность eps = {eps:g} достигнута. Число итераций (вращений): {len(history)}")
    print_matrix(f"A({len(history)}) — почти диагональная", Ak)

    section("Собственные значения (диагональ итоговой матрицы)")
    for i, lam in enumerate(eigenvalues):
        print(f"  lambda{i + 1} = {fmt(lam, 6)}")

    section("Собственные векторы")
    print_matrix("Матрица собственных векторов V = U(0) * U(1) * ... (столбцы — h1, ..., hn)", V, 6)
    print()
    for i in range(n):
        print(f"  h{i + 1} = ( " + "; ".join(fmt(V[r, i], 6) for r in range(n)) + " )")

    section("Проверка A*V = V*Lambda")
    Lam = np.zeros((n, n))
    for i in range(n):
        Lam[i, i] = eigenvalues[i]
    AV, VL = A @ V, V @ Lam
    print_matrix("A*V", AV)
    print_matrix("V*Lambda", VL)
    err = np.max(np.abs(AV - VL))
    print(f"max |A*V - V*Lambda| = {err:.2e}")
    print("A*V - V*Lambda = V*(A(k) - Lambda): это остаток недиагональных элементов, поэтому")
    if err <= eps:
        print(f"разница порядка t(A(k)) и равенство выполняется с заданной точностью eps = {eps:g}.")
    else:
        print(f"разница порядка t(A(k)) ~ eps = {eps:g}.")
    err_ort = np.max(np.abs(V.T @ V - np.eye(n)))
    verdict = "ортонормированы" if err_ort < CHECK_TOL else "НЕ ортонормированы"
    print(f"max |V^T*V - E| = {err_ort:.2e}  ->  собственные векторы {verdict}")
    print(f"Сумма собственных значений = {fmt(sum(eigenvalues), 6)},  след A = {fmt(np.trace(A), 6)}")

    section("Сравнение с NumPy (numpy.linalg.eigh, только для проверки)")
    lam_np = np.linalg.eigh(A)[0]
    lam_my = sorted(eigenvalues)
    print("Собственные значения по возрастанию:")
    for i in range(n):
        print(f"  {fmt(lam_my[i], 6):>14}   numpy: {fmt(lam_np[i], 6):>14}")
    print(f"max |lambda - lambda_numpy| = {max(abs(lam_my[i] - lam_np[i]) for i in range(n)):.2e}")


if __name__ == "__main__":
    main()
