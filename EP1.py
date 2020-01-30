import numpy as np

MAX_ITER = 20
EPSILON = 1e-10

# Calcula uma raiz de um polinomio p usando o método de Newton
# g = x - f(x)/df(x)
def firstFixedPoint(p, x0):
    x = x0
    dp = np.polyder(p)
    n_iter = 0
    while abs(np.polyval(p, x)) > EPSILON and n_iter < MAX_ITER:
        x = x - np.polyval(p, x) / np.polyval(dp, x)
        n_iter += 1
    return raiz

# Calcula alguma raiz zl1 != NaN com o método de Newton
# usando um número aleatório como aproximação inicial
# g = x - 1/(dp(x)/p(x) - dq(x)/q(x))
def generalFixedPoint(p, q):
    zl1 = np.nan
    dp = np.polyder(p)
    dq = np.polyder(q)
    while np.isnan(zl1):
        # xk1 = xk - 1/(Dp/p - Dq/q)
        x = complex(np.random.rand(), np.random.rand())
        n_iter = 0
        while abs(np.polyval(p, x)) > epsilon and n_iter < max_iter:
            print("x: ", x)
            # xk1 = xk - g(xk)
            x = x - 1/(np.polyval(dp, x)/np.polyval(p, x) - np.polyval(dq, x)/np.polyval(q, x))
            n_iter += 1

        if n_iter > max_iter:
             zli = np.nan
        else:
            print("z: ", x)
            zli = 1/x

def polyzeros(a):
    p = np.poly1d(a)
    print('p: ', p)
    zeros = []

    # Calcula a aproximação da primeira raiz
    raiz = firstFixedPoint(p, complex(np.random.rand(), np.random.rand())
    zeros.append(raiz)
    q = np.poly1d(zeros, True)

    # Adiciona raizes ao vetor de zeros enquanto ele for menor que
    # a lista de coeficientes de a (que tem a - 1 raizes)
    while(len(zeros) < len(a) - 1):
        # g = x - 1/(dp/p - dq/q)
        g = x - 1/((dp/p)[0] - (dq/q)[0])

        # Calcula alguma raiz zl1 != NaN com o método de Newton
        # usando um número aleatório como aproximação inicial
        zl1 = np.nan
        while np.isnan(zl1):
            # xk1 = xk - 1/(Dp/p - Dq/q)
            x0 = complex(np.random.rand(), np.random.rand())
            zl1 = fixed_point(p, g, x0)

        # Calcula uma raiz com o método de Newton
        # usando zl1 como aproximação inicial
        g = x - (p/dp)[0]
        print("g: ", g)
        zero = fixed_point(p, g, zl1)
        print("zero: ", zero)
        zeros.append(zero)
        q = np.poly1d(zeros, True)
        dq = np.polyder(q)
        print("q: ", q)

    return zeros

def main():

    print("zeros: ", polyzeros([1.5, 2, 3, 4, 8, -3, -6]))

main()