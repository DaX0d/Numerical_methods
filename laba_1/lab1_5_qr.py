import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT = Path(__file__).resolve().parent / "input" / "1_5" / "variant_17.txt"

MAX_ITER = 1000
CHECK_TOL = 1e-9
PREC = 4
MAX_TABLE_ROWS = 20


def read_matrix(path):
    """Читает точность eps и квадратную матрицу A из текстового файла."""
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
    return np.array(rows), eps


def householder_qr(A):
    """
    QR-разложение A = Q*R методом Хаусхолдера.

    На шаге k матрица отражения H = E - 2*v*v^T / (v^T*v) обнуляет элементы k-го столбца
    ниже диагонали. Вектор v:
        v_i = 0 (i < k),   v_k = a_kk + sign(a_kk) * sqrt( sum_(i>=k) a_ik^2 ),   v_i = a_ik (i > k).
    Знак sign(a_kk) выбран так, чтобы в v_k не вычитались близкие числа.
    R = H_(n-1) * ... * H_1 * A — верхняя треугольная,
    Q = H_1 * H_2 * ... * H_(n-1) — ортогональная (каждая H симметрична и ортогональна).
    """
    n = len(A)
    R = A.astype(float)
    Q = np.eye(n)
    for k in range(n - 1):
        norm = math.sqrt(sum(R[i, k] ** 2 for i in range(k, n)))
        if norm == 0:
            continue
        v = np.zeros(n)
        v[k] = R[k, k] + (norm if R[k, k] >= 0 else -norm)
        v[k + 1:] = R[k + 1:, k]
        H = np.eye(n) - 2 * np.outer(v, v) / (v @ v)
        R = H @ R
        R[k + 1:, k] = 0.0
        Q = Q @ H
    return Q, R


def norm_below(A, col, row_from):
    """sqrt( sum_(i >= row_from) a_(i,col)^2 ) — норма части столбца col начиная со строки row_from."""
    return math.sqrt(sum(A[i, col] ** 2 for i in range(row_from, len(A))))


def block_eigenvalues(A, j):
    """
    Собственные значения блока 2x2 в строках и столбцах j, j+1 — корни уравнения
        (a_jj - l) * (a_(j+1)(j+1) - l) - a_j(j+1) * a_(j+1)j = 0,
    т.е. l^2 - p*l + q = 0, где p — след блока, q — его определитель.
    При отрицательном дискриминанте корни комплексно-сопряжённые.
    """
    a, b = A[j, j], A[j, j + 1]
    c, d = A[j + 1, j], A[j + 1, j + 1]
    p = a + d
    q = a * d - b * c
    disc = p * p - 4 * q
    if disc >= 0:
        s = math.sqrt(disc)
        return complex((p + s) / 2), complex((p - s) / 2)
    s = math.sqrt(-disc)
    return complex(p / 2, s / 2), complex(p / 2, -s / 2)


def split_into_blocks(A, eps):
    """
    Разбор матрицы A(k) по столбцам слева направо. Возвращает список блоков (j, size, values):
      size = 1 — вещественное собственное значение a_jj (под ним всё <= eps);
      size = 2 — блок 2x2 в строках j, j+1 (ниже блока всё <= eps), values — его собственные значения;
      size = 0 — столбец j ещё не отделился, values = [a_jj] — пока только приближение.
    """
    n = len(A)
    blocks = []
    j = 0
    while j < n:
        if norm_below(A, j, j + 1) <= eps:
            blocks.append((j, 1, [complex(A[j, j])]))
            j += 1
        elif j + 1 < n and norm_below(A, j, j + 2) <= eps and norm_below(A, j + 1, j + 2) <= eps:
            blocks.append((j, 2, list(block_eigenvalues(A, j))))
            j += 2
        else:
            blocks.append((j, 0, [complex(A[j, j])]))
            j += 1
    return blocks


def block_change(block, prev_blocks):
    """|lambda(k) - lambda(k-1)| для блока 2x2; None, если на прошлой итерации такого блока не было."""
    j, _, values = block
    for pj, psize, pvalues in prev_blocks:
        if pj == j and psize == 2:
            return max(abs(values[0] - pvalues[0]), abs(values[1] - pvalues[1]))
    return None


def is_converged(blocks, prev_blocks, eps):
    """Все собственные значения найдены: вещественные отделились, у блоков 2x2 корни перестали меняться."""
    for block in blocks:
        if block[1] == 0:
            return False
        if block[1] == 2:
            change = block_change(block, prev_blocks)
            if change is None or change > eps:
                return False
    return True


def qr_algorithm(A, eps):
    """
    QR-алгоритм: A(k) = Q(k)*R(k),  A(k+1) = R(k)*Q(k).
    Возвращает итоговую матрицу A(k), history — список блоков по итерациям, и признак сходимости.
    """
    Ak = A.astype(float)
    history = []
    prev_blocks = []
    for _ in range(MAX_ITER):
        Q, R = householder_qr(Ak)
        Ak = R @ Q
        blocks = split_into_blocks(Ak, eps)
        history.append(blocks)
        if is_converged(blocks, prev_blocks, eps):
            return Ak, history, True
        prev_blocks = blocks
    return Ak, history, False


def fmt(v, prec=PREC):
    if abs(v) < 0.5 * 10 ** (-prec):
        v = 0.0
    return f"{v:.{prec}f}"


def fmt_complex(z, prec=PREC):
    if abs(z.imag) < 0.5 * 10 ** (-prec):
        return fmt(z.real, prec)
    sign = "+" if z.imag > 0 else "-"
    return f"{fmt(z.real, prec)} {sign} {fmt(abs(z.imag), prec)}i"


def print_matrix(name, M, prec=PREC):
    cells = [[fmt(v, prec) for v in row] for row in M]
    width = max(len(c) for row in cells for c in row)
    print(f"{name}:")
    for row in cells:
        print("  [ " + "  ".join(c.rjust(width) for c in row) + " ]")


def print_table(headers, rows):
    """Печатает таблицу строк; если строк много — середина заменяется на '...'."""
    widths = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]
    if len(rows) > MAX_TABLE_ROWS:
        rows = rows[:MAX_TABLE_ROWS - 5] + [["..."] * len(headers)] + rows[-5:]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(s.rjust(w) for s, w in zip(r, widths)))


def check(name, diff):
    err = np.max(np.abs(diff))
    verdict = "выполняется" if err < CHECK_TOL else "НЕ выполняется"
    print(f"max |{name}| = {err:.2e}  ->  равенство {verdict}")


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

    print("Задание 1.5. QR-алгоритм нахождения собственных значений")
    print(f"Входной файл: {path}")
    print(f"Размерность матрицы: n = {n}")
    print(f"Заданная точность: eps = {eps:g}")
    print()
    print_matrix("Исходная матрица A", A)

    section("QR-разложение исходной матрицы (метод Хаусхолдера)")
    Q, R = householder_qr(A)
    print_matrix("Q (ортогональная)", Q)
    print_matrix("R (верхняя треугольная)", R)
    print_matrix("Q*R", Q @ R)
    check("Q*R - A", Q @ R - A)
    check("Q^T*Q - E", Q.T @ Q - np.eye(n))

    section("Итерации QR-алгоритма")
    print("A(0) = A,  A(k) = Q(k)*R(k),  A(k+1) = R(k)*Q(k)")
    print("Критерии окончания:")
    print("  вещественное lambda_j = a_jj:        sqrt( sum_(i>j) a_ij^2 ) <= eps")
    print("  пара из блока 2x2 (строки j, j+1):   ниже блока всё <= eps и |lambda(k) - lambda(k-1)| <= eps")
    print()
    Ak, history, converged = qr_algorithm(A, eps)
    rows = []
    for k, blocks in enumerate(history, start=1):
        rows.append([str(k)] + [fmt_complex(z) for _, _, values in blocks for z in values])
    print_table(["k"] + [f"lambda{i + 1}" for i in range(n)], rows)
    print()
    if converged:
        print(f"Точность eps = {eps:g} достигнута. Число итераций: {len(history)}")
    else:
        print(f"Точность не достигнута, выполнено итераций: {MAX_ITER}. Ниже — последние приближения.")
    print_matrix(f"A({len(history)}) — квазитреугольная", Ak)

    section("Собственные значения")
    blocks = history[-1]
    prev_blocks = history[-2] if len(history) > 1 else []
    eigenvalues = []
    for block in blocks:
        j, size, values = block
        eigenvalues += values
        if size == 1:
            print(f"  lambda{j + 1} = {fmt_complex(values[0], 6)}")
            print(f"      вещественное: sqrt( sum_(i>{j + 1}) a_i{j + 1}^2 ) = {norm_below(Ak, j, j + 1):.2e}")
        elif size == 2:
            change = block_change(block, prev_blocks)
            change_text = f"{change:.2e}" if change is not None else "-"
            if values[0].imag != 0:
                z = values[0]
                print(f"  lambda{j + 1},{j + 2} = {fmt(z.real, 6)} +- {fmt(abs(z.imag), 6)}i")
                print(f"      комплексно-сопряжённая пара из блока 2x2 (строки {j + 1}, {j + 2}),")
            else:
                print(f"  lambda{j + 1} = {fmt_complex(values[0], 6)},  lambda{j + 2} = {fmt_complex(values[1], 6)}")
                print(f"      вещественная пара из блока 2x2 (строки {j + 1}, {j + 2}),")
            print(f"      |lambda(k) - lambda(k-1)| = {change_text}")
        else:
            print(f"  lambda{j + 1} ~ {fmt_complex(values[0], 6)}   (не отделилось)")

    section("Сравнение с NumPy (numpy.linalg.eigvals, только для проверки)")
    lam_np = list(np.linalg.eigvals(A))
    for z in eigenvalues:
        nearest = min(lam_np, key=lambda w: abs(w - z))
        print(f"  {fmt_complex(z, 6):>24}   numpy: {fmt_complex(nearest, 6):>24}   |разница| = {abs(z - nearest):.2e}")
    print(f"Сумма собственных значений = {fmt_complex(sum(eigenvalues), 6)},  след A = {fmt(np.trace(A), 6)}")


if __name__ == "__main__":
    main()
