import numpy as np

MAX_ITER = 10
EPSILON = 1e-8

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
    if n_iter >= MAX_ITER:
        return np.nan
    else: 
        return x

# Retorna o vetor de raizes do polinômio p(x) de coeficientes dados pelo vetor a
def polyzeros(a):
    p = np.poly1d(a, 1)
    dp = np.polyder(p)

    # g(x) = x - p(x) / dp(x)
    g = lambda x : x - np.polyval(p, x) / np.polyval(dp, x)
    
    # Calcula a aproximação da primeira raiz com o método de Newton
    # usando um número complexo aleatório como aproximação inicial
    raiz = pontoFixo(p, g, complex(np.random.rand(), np.random.rand()))
    
    raizes = [raiz]

    # q = polinômio de fatores do polinômio p encontrados até agora
    q = np.poly1d(raizes, True)
    dq = np.polyder(q)

    # Encontra as outras raízes
    # Adiciona raizes ao vetor de raizes enquanto ele for menor que
    # a lista de coeficientes de a (que tem a - 1 raizes)
    while(len(raizes) < len(a)):
        
        # Encontrar uma raiz != NaN usando o método de Newton para o polinômio p(x)/q(x)
        zl1 = np.nan
        while np.isnan(zl1):
            # x0 é um complexo aleatório de módulo entre 0 e 1
            x0 = complex(np.random.rand(), np.random.rand())
            
            # g(x) = p(x) / q(x)
            g = lambda x : x - 1 / (np.polyval(dp, x) / np.polyval(p, x) -
                                    np.polyval(dq, x) / np.polyval(q, x))
            zl1 = pontoFixo(p, g, x0)

        g = lambda x : x - np.polyval(p, x) / np.polyval(dp, x)
        # Calcula uma raiz com o método de Newton
        # usando zl1 como aproximação inicial
        raiz = pontoFixo(p, g, zl1)
        raizes.append(raiz)

        # Recalcula o polinômio q(x) com a nova raiz encontrada
        q = np.poly1d(raizes, True)
        dq = np.polyder(q)

    return raizes

def main():
    coeficientes = list(map(lambda c : complex(c), input(
        "Entre os coeficientes a de p(x) " + 
        "da maior ordem para a menor ordem").split(', ')))
    global EPSILON
    EPSILON = float(input("Entre Epsilon: "))
    global MAX_ITER
    MAX_ITER = int(input("Entre o numero máximo de iterações: "))

    print("raizes: ", polyzeros(coeficientes))
main()