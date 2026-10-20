import sys
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "1_1" / "variant_17.txt"

EPS = 1e-12
CHECK_TOL = 1e-9
PREC = 4


def read_system(path):
    """Читает расширенную матрицу [A | b] из текстового файла, возвращает A и b."""
    rows = []
    # utf-8-sig: файл, сохранённый в Блокноте с BOM, тоже прочитается
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].replace("|", " ").strip()
            if not line:
                continue
            try:
                rows.append([float(t) for t in line.split()])
            except ValueError:
                raise ValueError(f"строка {line_no}: не удалось прочитать числа") from None

    n = len(rows)
    if n == 0:
        raise ValueError("в файле нет ни одного уравнения")
    for i, row in enumerate(rows, start=1):
        if len(row) != n + 1:
            raise ValueError(f"в уравнении {i} количество чисел {len(row)}, а должно быть "
                             f"n + 1 = {n + 1} (n коэффициентов и свободный член)")

    M = np.array(rows)
    return M[:, :n], M[:, n]


def lu_decompose(A):
    """
    LU-разложение с выбором главного элемента по столбцу: P*A = L*U.

    Возвращает:
      L     — нижняя треугольная матрица с единицами на диагонали (множители исключения),
      U     — верхняя треугольная матрица (то, во что A превращается после прямого хода),
      perm  — перестановка строк: i-я строка P*A — это строка perm[i] матрицы A,
      swaps — число перестановок строк (определяет знак определителя).
    """
    n = A.shape[0]
    U = A.astype(float)
    L = np.zeros((n, n))
    perm = list(range(n))
    swaps = 0
    tol = EPS * np.max(np.abs(A))

    for k in range(n):
        # 1) Выбор главного элемента: максимальный по модулю в k-м столбце среди строк k..n-1
        p = k + int(np.argmax(np.abs(U[k:, k])))

        if abs(U[p, k]) <= tol:
            # Весь столбец начиная с диагонали нулевой -> det A = 0.
            # Исключать нечего, обнуляем остатки погрешности и идём к следующему столбцу.
            U[k:, k] = 0.0
            continue

        # 2) Перестановка строк k и p: в U, в уже найденных столбцах L и в perm
        if p != k:
            U[[k, p]] = U[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
            perm[k], perm[p] = perm[p], perm[k]
            swaps += 1

        # 3) Исключение x_k из строк ниже k-й, множители l_ik сохраняем в L
        for i in range(k + 1, n):
            l_ik = U[i, k] / U[k, k]
            L[i, k] = l_ik
            U[i, k] = 0.0
            U[i, k + 1:] -= l_ik * U[k, k + 1:]

    for i in range(n):
        L[i, i] = 1.0
    return L, U, perm, swaps


def permutation_matrix(perm):
    """Матрица перестановок P: в i-й строке единица стоит в столбце perm[i]."""
    n = len(perm)
    P = np.zeros((n, n))
    for i, j in enumerate(perm):
        P[i, j] = 1.0
    return P


def forward_substitution(L, b):
    """Прямая подстановка: решение L*y = b (L — нижняя треугольная, l_ii = 1)."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def backward_substitution(U, y):
    """Обратная подстановка: решение U*x = y (U — верхняя треугольная)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu_solve(L, U, perm, b):
    """Решение A*x = b по готовому разложению: L*y = P*b, затем U*x = y."""
    y = forward_substitution(L, b[perm])   # b[perm] — это P*b
    return backward_substitution(U, y)


def determinant(U, swaps):
    """det A = (-1)^swaps * u_11 * u_22 * ... * u_nn, т.к. det P = (-1)^swaps, det L = 1."""
    det = (-1.0) ** swaps
    for i in range(U.shape[0]):
        det *= U[i, i]
    return det


def inverse_matrix(L, U, perm):
    """A^-1 по столбцам: j-й столбец — решение A*x = e_j (разложение одно на все n систем)."""
    n = L.shape[0]
    A_inv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A_inv[:, j] = lu_solve(L, U, perm, e)
    return A_inv


def fmt(v, prec):
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


def print_check(name, diff):
    err = np.max(np.abs(diff))
    verdict = "выполняется" if err < CHECK_TOL else "НЕ выполняется"
    print(f"max |{name}| = {err:.2e}  ->  равенство {verdict} (допуск {CHECK_TOL:g})")


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 70 - len(title)))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    try:
        A, b = read_system(path)
    except (OSError, ValueError) as e:
        print(f"Ошибка чтения входного файла {path}: {e}")
        sys.exit(1)
    n = len(b)

    print("Задание 1.1. LU-разложение с выбором главного элемента по столбцу")
    print(f"Входной файл: {path}")
    print(f"Размерность системы: n = {n}")
    print()
    print_matrix("Матрица A", A)
    print_vector("b", b)

    L, U, perm, swaps = lu_decompose(A)
    P = permutation_matrix(perm)
    det = determinant(U, swaps)
    singular = any(U[i, i] == 0.0 for i in range(n))

    section("1. Матрицы L и U  (P*A = L*U)")
    print("Порядок строк A в P*A:", [j + 1 for j in perm])
    print(f"Число перестановок строк: {swaps}")
    print_matrix("P", P, prec=0)
    print_matrix("L", L)
    print_matrix("U", U)

    section("2. Произведение L*U")
    print_matrix("L*U", L @ U)

    section("3. Решение системы A*x = b")
    if singular:
        print("Матрица A вырождена (на диагонали U есть ноль) -> единственного решения нет.")
    else:
        y = forward_substitution(L, b[perm])
        x = backward_substitution(U, y)
        print_vector("P*b", b[perm])
        print_vector("y  (L*y = P*b)", y, prec=6)
        print("x  (U*x = y):")
        for i in range(n):
            print(f"  x{i + 1} = {fmt(x[i], 6)}")
        print_check("A*x - b", A @ x - b)

    section("4. Обратная матрица A^-1 (через LU-разложение)")
    if singular:
        print("Матрица A вырождена -> обратной матрицы не существует.")
    else:
        A_inv = inverse_matrix(L, U, perm)
        print_matrix("A^-1", A_inv, prec=6)

    section("5. Определитель (через LU-разложение)")
    print("det A = (-1)^s * u_11 * ... * u_nn, s — число перестановок строк")
    print(f"det A = (-1)^{swaps} * " + " * ".join(f"({fmt(U[i, i], PREC)})" for i in range(n)))
    print(f"det A = {fmt(det, 6)}")

    section("6. Проверка A*A^-1 = E")
    if singular:
        print("Обратной матрицы нет — проверка не выполняется.")
    else:
        AA_inv = A @ A_inv
        print_matrix("A*A^-1", AA_inv)
        print_check("A*A^-1 - E", AA_inv - np.eye(n))

    section("7. Проверка L*U = P*A")
    print_matrix("P*A", P @ A)
    print_check("L*U - P*A", L @ U - P @ A)

    if not singular:
        section("Сравнение с NumPy")
        print(f"max |x - numpy.linalg.solve(A, b)| = {np.max(np.abs(x - np.linalg.solve(A, b))):.2e}")
        print(f"|det - numpy.linalg.det(A)|        = {abs(det - np.linalg.det(A)):.2e}")
        print(f"max |A^-1 - numpy.linalg.inv(A)|   = {np.max(np.abs(A_inv - np.linalg.inv(A))):.2e}")


if __name__ == "__main__":
    main()
