import sys
import numpy as np
import matplotlib.pyplot as plt

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
# e um um vetor de zeros já encontrados do polinômio p(x).
# Retorna uma raiz do polinômio deflacionado f(x) = p(x)/q(x), onde 
# q(x) = polinômio dos fatores de p(x) determinados pelo vetor de zeros.
def NewtonComDeflação(p, zeros):
    # dp(x), derivada do polinômio p(x)
    dp = np.polyder(p)

    # Encontra uma raiz != NaN
    zl1 = np.nan
    while np.isnan(zl1):
        # x0 é um complexo aleatório de módulo entre 0 e 1
        x = complex(np.random.rand(), np.random.rand())

        # Para se |p(x)| > EPSILON ou se n_iter >= MAX_ITER
        n_iter = 0
        while abs(np.polyval(p, x)) >= EPSILON and n_iter < MAX_ITER:
            # Calcula dq(x)/q(x)
            dqq = 0
            for zero in zeros:
                dqq += 1 / np.polyval(np.poly1d([zero], True), x)

            # xk1 = xk - 1/(dp/p - dq/q)
            x = x - 1/(np.polyval(dp, x)/np.polyval(p, x) - dqq)
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
    zero = NewtonComAproximaçãoInicial(p, complex(np.random.rand(), np.random.rand()))
    zeros.append(zero)

    # Encontra as outras raízes
    # Adiciona raizes ao vetor de zeros enquanto ele for menor que
    # a lista de coeficientes de a (que tem a - 1 raizes)
    while(len(zeros) < len(a) - 1):
        # Calcula alguma raiz zl1 != NaN com o método de Newton
        # do polinômio deflacionado f(x) = p(x)/q(x), onde q(x) é
        # o polinômio determinado pelos fatores de p(x) dados pelo vetor de zeros
        zl1 = NewtonComDeflação(p, zeros)

        # Calcula uma raiz com o método de Newton
        # usando zl1 como aproximação inicial
        zero = NewtonComAproximaçãoInicial(p, zl1)
        zeros.append(zero)

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

    try:
        global MAX_ITER
        MAX_ITER = int(f.readline().strip())
    except:
        print("Erro ao ler os epsilon.")
        print("Certifique-se de que seja um inteiro e" +
        "de que esteja na terceira linha de entrada.txt")
        print(sys.exc_info()[1], "aconteceu.")

    # Calcula os zeros do polinômio correspondente aos coeficientes
    zeros = polyzeros(coeficientes)

    # Plota os gráficos dos zeros encontrados
    reais = []
    imaginários = []
    for zero in zeros:
        reais.append(zero.real)
        imaginários.append(zero.imag)
    plt.plot(reais, imaginários, "r*")
    plt.title(str(np.poly1d(coeficientes)), loc='left', family='monospace')
    plt.xlabel("Eixo real", family='monospace')
    plt.ylabel("Eixo imaginário", family='monospace')
    plt.legend(["polyzeros(a)"])
    plt.savefig("polyzeros(" + str(coeficientes) + ")")
    plt.show()
    
    roots = np.roots(coeficientes)
    reais = []
    imaginários = []
    for root in roots:
        reais.append(root.real)
        imaginários.append(root.imag)
    plt.plot(reais, imaginários, "go")
    plt.title(str(np.poly1d(coeficientes)), loc='left', family='monospace')
    plt.xlabel("Eixo real", family='monospace')
    plt.ylabel("Eixo imaginário", family='monospace')
    plt.legend(["numpy.roots(a)"])
    plt.savefig("numpy_roots(" + str(coeficientes) + ")")
    plt.show()

    print("polyzeros: ", zeros)
    print("numpy.roots: ", roots)
    f.close()

main()
