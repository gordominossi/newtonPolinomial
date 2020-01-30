import numpy as np

MAX_ITER = 20
EPSILON = 1e-10

## Calcula uma raiz usando o método do ponto fixo para uma função g polinomial
#  recebe os polinomios p e g e uma aproximação inicial da raiz x0
#  retorna uma aproximação de raiz quando n_iter < max_iter ou |p(x)| < epsilon
def fixed_point(p, g, x0):
    epsilon = EPSILON
    max_iter = MAX_ITER

    x = x0

    n_iter = 0
    while abs(np.polyval(p, x)) > epsilon and n_iter < max_iter:
        print("x: ", x)
        # xk1 = xk - g(xk)
        x = np.polyval(g, x)
        n_iter += 1
    
    if n_iter > max_iter:
         return np.nan
    else:
        print("z: ", x)
        return 1/x

def polyzeros(a):
    p = np.poly1d(a)
    x = np.poly1d([0], True)
    zeros = []
    # g = x - f/df
    g = x - (p/np.polyder(p))[0]
    print("g: ", g)

    # Calcula a aproximação da primeira raiz
    zero = np.nan
    while np.isnan(zero):
        x0 = complex(np.random.rand(), np.random.rand())
        zero = fixed_point(p, g, x0)
    zeros.append(zero)
    
    q = np.poly1d(zeros, True)
    print("q: ", q)

    # Adiciona raizes ao vetor de zeros enquanto ele for menor que
    # a lista de coeficientes de a - 1
    while(len(zeros) < len(a) - 1):
        # g = x - 1/(dp/p - dq/q)
        dp = np.polyder(p)
        dq = np.polyder(q)
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
        print("q: ", q)

    return zeros

def main():
    a = [1.5, 2, 3, 4, 8, -3, -6]
    # print("zeros: ", polyzeros([1.5, 2, 3, 4, 8, -3, -6]))

    # print("zeros: ", polyzeros([1, 2, 3]))
    p = np.poly1d(a)
    dp = np.polyder(p)
    print (p/dp)
main()