import statistics
import sys
import time

import numpy as np


def generate_matrix(n):
    return np.random.rand(n, n).astype(np.float64)


def dgemm_numpy(a, b):
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    c = np.dot(a, b)

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    return c, end_wall - start_wall, end_cpu - start_cpu


def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")


def main():
    if len(sys.argv) < 2:
        print("Uso: python numpy_dgemm.py <tamanho_da_matriz>")
        return

    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Informe um tamanho de matriz inteiro e positivo.")
        return

    num_tests = 5
    wall_times = []
    cpu_times = []
    log_path = f"numpy_dgemm_log_{n}.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_message(f"Executando {num_tests} testes de DGEMM com NumPy", log_file)
        log_message(f"Tamanho da matriz: {n} x {n}", log_file)
        log_message("Tipo de dado: float64", log_file)
        log_message("-" * 40, log_file)

        for test_index in range(1, num_tests + 1):
            log_message(f"Teste {test_index}/{num_tests}: gerando matrizes...", log_file)
            a = generate_matrix(n)
            b = generate_matrix(n)

            log_message(f"Teste {test_index}/{num_tests}: iniciando multiplicacao...", log_file)
            _, total_wall_time, total_cpu_time = dgemm_numpy(a, b)

            wall_times.append(total_wall_time)
            cpu_times.append(total_cpu_time)

            log_message(
                f"Resultado teste {test_index}: "
                f"wall = {total_wall_time:.6f} s | "
                f"cpu = {total_cpu_time:.6f} s",
                log_file,
            )
            log_message("-" * 40, log_file)

        avg_wall = statistics.mean(wall_times)
        avg_cpu = statistics.mean(cpu_times)
        min_wall = min(wall_times)
        max_wall = max(wall_times)
        min_cpu = min(cpu_times)
        max_cpu = max(cpu_times)
        std_wall = statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
        std_cpu = statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0

        log_message("", log_file)
        log_message("========== RESUMO FINAL ==========", log_file)
        log_message(f"Tamanho da matriz: {n} x {n}", log_file)
        log_message(f"Numero de testes: {num_tests}", log_file)
        log_message(f"Wall time medio: {avg_wall:.6f} s", log_file)
        log_message(f"Wall time minimo: {min_wall:.6f} s", log_file)
        log_message(f"Wall time maximo: {max_wall:.6f} s", log_file)
        log_message(f"Desvio padrao wall: {std_wall:.6f} s", log_file)
        log_message(f"CPU time medio: {avg_cpu:.6f} s", log_file)
        log_message(f"CPU time minimo: {min_cpu:.6f} s", log_file)
        log_message(f"CPU time maximo: {max_cpu:.6f} s", log_file)
        log_message(f"Desvio padrao cpu: {std_cpu:.6f} s", log_file)
        log_message("==================================", log_file)

    print(f"\nLog salvo em: {log_path}")


if __name__ == "__main__":
    main()
