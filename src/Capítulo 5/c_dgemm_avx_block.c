#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_TESTS 5
#define UNROLL 4
#define AVX_WIDTH 4
#define BLOCKSIZE 32
#define ALIGNMENT 32

static double get_wall_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void log_message(FILE *log_file, const char *format, ...)
{
    va_list args;

    va_start(args, format);
    vprintf(format, args);
    printf("\n");
    va_end(args);

    if (log_file != NULL)
    {
        va_start(args, format);
        vfprintf(log_file, format, args);
        fprintf(log_file, "\n");
        va_end(args);
    }
}

static double calculate_mean(const double *values, int count)
{
    double sum = 0.0;

    for (int i = 0; i < count; i++)
    {
        sum += values[i];
    }

    return sum / count;
}

static double calculate_stddev(const double *values, int count, double mean)
{
    if (count < 2)
    {
        return 0.0;
    }

    double sum = 0.0;

    for (int i = 0; i < count; i++)
    {
        double diff = values[i] - mean;
        sum += diff * diff;
    }

    return sqrt(sum / (count - 1));
}

static bool allocate_aligned_matrix(size_t n, double **matrix)
{
    return posix_memalign((void **)matrix, ALIGNMENT, n * n * sizeof(double)) == 0;
}

static void create_random_matrix(size_t n, double *matrix)
{
    size_t total = n * n;

    for (size_t i = 0; i < total; i++)
    {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

static void initialize_matrix_with_zeros(size_t n, double *matrix)
{
    size_t total = n * n;

    for (size_t i = 0; i < total; i++)
    {
        matrix[i] = 0.0;
    }
}

static void do_block(size_t n, size_t si, size_t sj, size_t sk, const double *A, const double *B, double *C)
{
    for (size_t i = si; i < si + BLOCKSIZE; i += UNROLL * AVX_WIDTH)
    {
        for (size_t j = sj; j < sj + BLOCKSIZE; j++)
        {
            __m256d c[UNROLL];

            for (int r = 0; r < UNROLL; r++)
            {
                c[r] = _mm256_load_pd(&C[i + r * AVX_WIDTH + j * n]);
            }

            for (size_t k = sk; k < sk + BLOCKSIZE; k++)
            {
                __m256d b0 = _mm256_broadcast_sd(&B[k + j * n]);

                for (int r = 0; r < UNROLL; r++)
                {
                    __m256d a0 = _mm256_load_pd(&A[i + r * AVX_WIDTH + k * n]);
                    c[r] = _mm256_fmadd_pd(a0, b0, c[r]);
                }
            }

            for (int r = 0; r < UNROLL; r++)
            {
                _mm256_store_pd(&C[i + r * AVX_WIDTH + j * n], c[r]);
            }
        }
    }
}

static void dgemm_c_avx_block(size_t n, const double *A, const double *B, double *C)
{
    for (size_t sj = 0; sj < n; sj += BLOCKSIZE)
    {
        for (size_t si = 0; si < n; si += BLOCKSIZE)
        {
            for (size_t sk = 0; sk < n; sk += BLOCKSIZE)
            {
                do_block(n, si, sj, sk, A, B, C);
            }
        }
    }
}

static bool run_single_test(size_t n, double *wall_time, double *cpu_time)
{
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    if (!allocate_aligned_matrix(n, &A) || !allocate_aligned_matrix(n, &B) || !allocate_aligned_matrix(n, &C))
    {
        free(A);
        free(B);
        free(C);
        return false;
    }

    create_random_matrix(n, A);
    create_random_matrix(n, B);
    initialize_matrix_with_zeros(n, C);

    double start_wall = get_wall_time_seconds();
    clock_t start_cpu = clock();

    dgemm_c_avx_block(n, A, B, C);

    clock_t end_cpu = clock();
    double end_wall = get_wall_time_seconds();

    *wall_time = end_wall - start_wall;
    *cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    free(A);
    free(B);
    free(C);

    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Uso: ./c_dgemm_avx_block <tamanho_da_matriz>\n");
        return 1;
    }

    char *endptr = NULL;
    errno = 0;

    unsigned long parsed_n = strtoul(argv[1], &endptr, 10);

    if (errno != 0 || endptr == argv[1] || *endptr != '\0' || parsed_n == 0)
    {
        printf("Informe um tamanho de matriz inteiro e positivo.\n");
        return 1;
    }

    size_t n = (size_t)parsed_n;

    if (n % BLOCKSIZE != 0)
    {
        printf("Para manter a estrutura do capitulo 5, informe um tamanho multiplo de %d.\n", BLOCKSIZE);
        return 1;
    }

    double wall_times[NUM_TESTS];
    double cpu_times[NUM_TESTS];
    char log_path[128];

    snprintf(log_path, sizeof(log_path), "c_dgemm_avx_block_log_%zu.txt", n);

    FILE *log_file = fopen(log_path, "w");

    if (log_file == NULL)
    {
        printf("Falha ao criar arquivo de log: %s\n", log_path);
        return 1;
    }

    srand((unsigned int)time(NULL));

    log_message(log_file, "Executando %d testes de DGEMM em C (Otimizacao 3 - AVX com unrolling e blocking)", NUM_TESTS);
    log_message(log_file, "Tamanho da matriz: %zu x %zu", n, n);
    log_message(log_file, "Representacao das matrizes: arrays unidimensionais em ordem coluna-major");
    log_message(log_file, "Implementacao: intrinsics AVX com FMA, unrolling %d e blocos %dx%d", UNROLL, BLOCKSIZE, BLOCKSIZE);
    log_message(log_file, "Alocacao: memoria alinhada em %d bytes para loads/stores AVX alinhados", ALIGNMENT);
    log_message(log_file, "----------------------------------------");

    for (int test_index = 0; test_index < NUM_TESTS; test_index++)
    {
        log_message(log_file, "Teste %d/%d: gerando matrizes...", test_index + 1, NUM_TESTS);
        log_message(log_file, "Teste %d/%d: iniciando multiplicacao...", test_index + 1, NUM_TESTS);

        if (!run_single_test(n, &wall_times[test_index], &cpu_times[test_index]))
        {
            log_message(log_file, "Falha ao alocar memoria para o teste %d.", test_index + 1);
            fclose(log_file);
            return 1;
        }

        log_message(
            log_file,
            "Resultado teste %d: wall = %.6f s | cpu = %.6f s",
            test_index + 1,
            wall_times[test_index],
            cpu_times[test_index]);
        log_message(log_file, "----------------------------------------");
    }

    double avg_wall = calculate_mean(wall_times, NUM_TESTS);
    double avg_cpu = calculate_mean(cpu_times, NUM_TESTS);
    double std_wall = calculate_stddev(wall_times, NUM_TESTS, avg_wall);
    double std_cpu = calculate_stddev(cpu_times, NUM_TESTS, avg_cpu);
    double min_wall = wall_times[0];
    double max_wall = wall_times[0];
    double min_cpu = cpu_times[0];
    double max_cpu = cpu_times[0];

    for (int i_index = 1; i_index < NUM_TESTS; i_index++)
    {
        if (wall_times[i_index] < min_wall)
        {
            min_wall = wall_times[i_index];
        }

        if (wall_times[i_index] > max_wall)
        {
            max_wall = wall_times[i_index];
        }

        if (cpu_times[i_index] < min_cpu)
        {
            min_cpu = cpu_times[i_index];
        }

        if (cpu_times[i_index] > max_cpu)
        {
            max_cpu = cpu_times[i_index];
        }
    }

    log_message(log_file, "");
    log_message(log_file, "========== RESUMO FINAL ==========");
    log_message(log_file, "Tamanho da matriz: %zu x %zu", n, n);
    log_message(log_file, "Numero de testes: %d", NUM_TESTS);
    log_message(log_file, "Wall time medio: %.6f s", avg_wall);
    log_message(log_file, "Wall time minimo: %.6f s", min_wall);
    log_message(log_file, "Wall time maximo: %.6f s", max_wall);
    log_message(log_file, "Desvio padrao wall: %.6f s", std_wall);
    log_message(log_file, "CPU time medio: %.6f s", avg_cpu);
    log_message(log_file, "CPU time minimo: %.6f s", min_cpu);
    log_message(log_file, "CPU time maximo: %.6f s", max_cpu);
    log_message(log_file, "Desvio padrao cpu: %.6f s", std_cpu);
    log_message(log_file, "Arquivo de log: %s", log_path);

    fclose(log_file);
    return 0;
}

/*
Compilacao:
gcc -O3 -mavx -mfma -Wall -Wextra -o c_dgemm_avx_block 'src/Capítulo 5/c_dgemm_avx_block.c' -lm

Execucao:
./c_dgemm_avx_block 512
*/
