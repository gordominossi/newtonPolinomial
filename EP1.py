import sys
import numpy as np

MAX_ITER = 10
EPSILON = 1e-8

# Calcula uma raiz de um polinomio p usando o método de Newton
# usando um número complexo x0 como aproximação inicial.
# Recebe um polinômio p(x) do qual se quer saber um raiz
# e uma aproximação inicial x0 da raiz.
# Retorna uma raiz do polinômio p(x)
# g = x - f(x)/df(x)
def NewtonComAproximaçãoInicial(p, x0):
    x = x0
    # Calcula a derivada dp do polinômio p
    dp = np.polyder(p)
    n_iter = 0
    while abs(np.polyval(p, x)) > EPSILON and n_iter < MAX_ITER:
        # xk+1 = xk - f(xk)/df(xk)
        x = x - np.polyval(p, x) / np.polyval(dp, x)
        n_iter += 1
    return x


# Calcula alguma raiz zl1 com o método de Newton
# usando um número complexo aleatório como aproximação inicial.
# Recebe um polinômio p(x) do qual se quer saber a raiz
# e um polinômio q(x) de fatores já encontrados do polinômio p(x).
# Retorna uma raiz do polinômio deflacionado f(x) = p(x)/q(x)
def NewtonComDeflação(p, q):
    dp = np.polyder(p)
    dq = np.polyder(q)
    # Encontrar uma raiz != NaN
    zl1 = np.nan
    while np.isnan(zl1):
        # x0 é um complexo aleatório de módulo entre 0 e 1
        x = complex(np.random.rand(), np.random.rand())

        # Para se |p(x)| > EPSILON ou se n_iter >= MAX_ITER
        n_iter = 0
        while abs(np.polyval(p, x)) >= EPSILON and n_iter < MAX_ITER:
            # xk1 = xk - 1/(dp/p - dq/q)
            x = x - 1/(np.polyval(dp, x)/np.polyval(p, x) - np.polyval(dq, x)/np.polyval(q, x))
            n_iter += 1

        # Se n_iter > MAX_ITER, ainda não achou uma raiz de f = p/q
        if n_iter > MAX_ITER:
            zl1 = np.nan
        else:
            zl1 = x
    return zl1


# Retorna um vetor de aproximações do polinômio p(x)
# Recebe o vetor dos coeficientes a do polinômio p(x)
def polyzeros(a):
    p = np.poly1d(a)
    zeros = []

    # Calcula a aproximação da primeira raiz com o método de Newton
    # usando um número complexo aleatório como aproximação inicial
    raiz = NewtonComAproximaçãoInicial(p, complex(np.random.rand(), np.random.rand()))
    zeros.append(raiz)

    # q = polinômio de fatores do polinômio p encontrados até agora
    q = np.poly1d(zeros, True)

    # Encontra as outras raízes
    # Adiciona raizes ao vetor de zeros enquanto ele for menor que
    # a lista de coeficientes de a (que tem a - 1 raizes)
    while(len(zeros) < len(a) - 1):
        # Calcula alguma raiz zl1 != NaN com o método de Newton
        # do polinômio deflacionado f(x) = p(x)/q(x)
        zl1 = NewtonComDeflação(p, q)

        # Calcula uma raiz com o método de Newton
        # usando zl1 como aproximação inicial
        zero = NewtonComAproximaçãoInicial(p, zl1)
        zeros.append(zero)

        # Recalcula o polinômio q(x) com a nova raiz encontrada
        q = np.poly1d(zeros, True)

    return zeros


def parseStringsToComplexes(strings):
    complexes = []
    for string in strings:
        complexes.append(complex(string))
    return complexes


def main():

    try:
        f = open("entrada.txt", "r")
    except:
        print("Erro ao ler o arquivo de entrada.")
        print("Certifique-se de que esteja no mesmo diretório e " +
        "de que seja chamado entrada.txt")
        print(sys.exc_info()[1], "aconteceu.")
        return 1

    try:
        coeficientes = parseStringsToComplexes(f.readline().strip("[]\n").split(', '))
    except:
        print("Erro ao ler os coeficientes.")
        print("Certifique-se de que estão no formato a + bj e " +
        "de que estejam na primeira linha de entrada.txt")
        print(sys.exc_info()[1], "aconteceu.")
        f.close()
        return 1
    try:
        global EPSILON
        EPSILON = float(f.readline().strip())
    except:
        print("Erro ao ler os epsilon.")
        print("Certifique-se de que seja um ponto flutuante e " +
        "de que esteja na segunda linha de entrada.txt")
        print(sys.exc_info()[1], "aconteceu.")
        f.close()
        return 1

    try:
        global MAX_ITER
        MAX_ITER = int(f.readline().strip())
    except:
        print("Erro ao ler os epsilon.")
        print("Certifique-se de que seja um inteiro e" +
        "de que esteja na terceira linha de entrada.txt")
        print(sys.exc_info()[1], "aconteceu.")
        f.close()
        return 1

    print("zeros: ", polyzeros(coeficientes))
    f.close()


main()
