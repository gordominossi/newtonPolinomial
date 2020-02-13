import numpy as np
from matplotlib import pyplot as plt

MAX_ITER = 10
EPSILON = 1e-16

# Calcula uma raiz do polinômio p(x) usando o Método do Ponto fixo para xk+1 = g(xk)
# Recebe p(x), polinômio usado para determinar o critério de parada, g(x),
# função para qual os sucessivos x's convergem e x0, uma aproximação inicial de x
# Retorna uma arpoximação x para uma raiz de p(x)
def pontoFixo(p, g, x0):
    x = x0

    n_iter = 0
    while abs(np.polyval(p, x)) > EPSILON and n_iter < MAX_ITER:
        x = g(x)
        n_iter += 1
    return x

# Método de Newton para um polinômio p(x)
# Recebe p(x) e x0 e retorna uma raiz de p(x)
def newtonClassico(p, x0):
    dp = np.polyder(p)

    # g(x) = x - p(x) / p'(x)
    g = lambda x : x - np.polyval(p, x) / np.polyval(dp, x)

    return pontoFixo(p, g, x0)


# Calcula usa o Método de Newton para a função f(x) = p(x) / q(x), em que 
# q(x) é o polinốmio de fatores já encontrados de p(x) dado pelo vetor de raizes
# Recebe o polinômio p(x), um vetor das raizes já encontradas e
# Retorna uma aproximação de uma raiz nova de p(x)
def newtonPolinomial(p, raizes):
    dp = np.polyder(p)

    # x0 é um complexo aleatório de módulo entre 0 e 1
    x0 = complex(np.random.rand(), np.random.rand())

    # dqq(x) = q'(x) / q(x) = soma das frações 1 / (x - raiz) para cada raiz já encontrada
    dqq = lambda x : sum(list(map(lambda raiz : 1 / np.polyval(np.poly1d([raiz], True), x), raizes)))

    # f(x) = p(x) / q(x)
    # g(x) = x - f(x) / f'(x) = 1 / (p'(x) / p(x) - q'(x) / q(x))
    g = lambda x : x - 1 / (np.polyval(dp, x) / np.polyval(p, x) - dqq(x))

    return pontoFixo(p, g, x0)



# Retorna o vetor de raizes do polinômio p(x) de coeficientes dados pelo vetor a
def polyzeros(a):
    p = np.poly1d(a)

    # Calcula a aproximação da primeira raiz com o método de Newton
    # usando um número complexo aleatório como aproximação inicial
    raiz = newtonClassico(p, complex(np.random.rand(), np.random.rand()))
    raizes = [raiz]


    # Encontra as outras raízes
    # Adiciona raizes ao vetor de raizes [z1, ..., zn] enquanto ele for menor que
    # a lista de coeficientes de a [a0, ..., an]
    while(len(raizes) < len(a) - 1):
        # Encontra uma raiz usando o método de Newton para o polinômio p(x)/q(x)
        zl1 = newtonPolinomial(p, raizes)

        # Refina a aproximação zl1 da raiz
        raiz = newtonClassico(p, zl1)
        raizes.append(raiz)

    return raizes

def plot(raizes, formatação, nome, coeficientes):

    reais = list(map(lambda raiz : raiz.real, raizes))
    imaginários = list(map(lambda raiz : raiz.imag, raizes))

    plt.plot(reais, imaginários, formatação)

    plt.title(str(np.poly1d(coeficientes)), loc='left', family='monospace')
    plt.xlabel("Eixo real", family='monospace')
    plt.ylabel("Eixo imaginário", family='monospace')
    plt.legend([nome])

    plt.savefig(nome + str(coeficientes) + ".png")
    plt.show()

def main():
    coeficientes = list(map(lambda c : complex(c), input(
        "Entre os coeficientes a de p(x) " +
        "da maior ordem para a menor ordem " +
        "separados por ', ': ").split(', ')))
    global EPSILON
    EPSILON = float(input("Entre Epsilon: "))
    global MAX_ITER
    MAX_ITER = int(input("Entre o numero máximo de iterações: "))

    raizes = np.roots(coeficientes)
    plot(raizes, "rs", "numpy.roots", coeficientes)

    print("raizes numpy.roots: ", raizes)

    raizes = polyzeros(coeficientes)
    plot(raizes, "g*", "polyzeros", coeficientes)

    print("raizes polyzeros: ", raizes)

main()
