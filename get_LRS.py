import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg
from scipy.fftpack import fft2, ifft2
import os
import copy
import matplotlib.pyplot as plt
cv2.setNumThreads(0)

def im2doubleAux(I):
    if (len(I.shape) == 3):
        n, m, d = I.shape
    elif (len(I.shape) == 2):
        n, m = I.shape

    maximo = np.max(I)
    minimo = np.min(I)
    salida = np.empty(I.shape).astype('float64')
    for linea in range(n):
        for col in range(m):
            if (len(I.shape) == 3):
                for dim in range(d):
                    salida[linea, col, dim] = ((I[linea, col, dim]) - minimo) / (maximo - minimo)
            elif (len(I.shape) == 2):
                salida[linea, col] = ((I[linea, col]) - minimo) / (maximo - minimo)
    return salida

def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: se ha dado tamaño nulo o negativo")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: tamaño de salida menor que el de entrada")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: distinta paridad de tamaños "
                             "de entrada y salida.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape, testFormaPSF = False):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')
    if testFormaPSF:
        print(psf)
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    otf = np.fft.fft2(psf)

    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf
def LIME(L, para):
    T_ini = np.max(L, axis=2) + 0.02
    wx, wy = computeTextureWeights(T_ini, para['sigma'])
    T_ref = solveLinearEquation(T_ini, wx, wy, para['lambda'])

    T_ref = np.clip(T_ref ** (1 / para['gamma']), 0, 1)

    I = np.zeros_like(L)
    I[:, :, 0] = L[:, :, 0] / T_ref
    I[:, :, 1] = L[:, :, 1] / T_ref
    I[:, :, 2] = L[:, :, 2] / T_ref

    return I, T_ini, T_ref

#2D
def computeTextureWeights(fin, sigma):
    fx = np.diff(fin, axis=1)
    fx = np.pad(fx, ((0, 0), (0, 1)), 'constant')

    fy = np.diff(fin, axis=0)
    fy = np.pad(fy, ((0, 1), (0, 0)), 'constant')

    vareps_s = 0.02
    vareps = 0.001

    wto = np.maximum(np.sqrt(fx**2 + fy**2).sum(axis=(0, 1)) / fin.shape[0], vareps_s)**(-1)
    fbin = lpfilter(fin, sigma)

    gfx = np.diff(fbin, axis=1)
    gfx = np.pad(gfx, ((0, 0), (0, 1)), 'constant')

    gfy = np.diff(fbin, axis=0)
    gfy = np.pad(gfy, ((0, 1), (0, 0)), 'constant')

    wtbx = np.maximum(np.abs(gfx).sum(axis=(0, 1)) / fin.shape[0], vareps)**(-1)
    wtby = np.maximum(np.abs(gfy).sum(axis=(0, 1)) / fin.shape[0], vareps)**(-1)

    retx = wtbx * wto
    rety = wtby * wto

    return retx, rety

def lpfilter(FImg, sigma):
    FBImg = np.copy(FImg)
    if FImg.ndim == 2:
        FBImg = conv2_sep(FImg, sigma)
    else:
        for ic in range(FBImg.shape[2]):
            FBImg[:, :, ic] = conv2_sep(FImg[:, :, ic], sigma)
    return FBImg

def conv2_sep(im, sigma):
    ksize = max(3, int(5 * sigma) | 1)  # 确保ksize是奇数
    g = cv2.getGaussianKernel(ksize, sigma).reshape(1, -1)
    ret = convolve2d(im, g)
    ret = convolve2d(ret, g.T)
    return ret

def solveLinearEquation(IN, wx, wy, lambda_):
    r, c = IN.shape
    k = r * c

    if np.isscalar(wx):
        wx = np.full_like(IN, wx)
    if np.isscalar(wy):
        wy = np.full_like(IN, wy)

    dx = -lambda_ * wx.flatten()
    dy = -lambda_ * wy.flatten()

    B = np.vstack([dx, dy])

    d = np.array([-r, -1])

    A = spdiags(B, d, k, k)

    e = dx
    w = np.pad(dx, (r, 0))[:-r]
    s = dy
    n = np.pad(dy, (1, 0))[:-1]

    D = 1 - (e + w + s + n)

    A = A + A.T + spdiags(D, 0, k, k)

    OUT = np.copy(IN)

    if len(IN.shape) == 2:
        tin = IN.flatten()
        tout, _ = cg(A, tin, tol=0.1, maxiter=100)

        OUT = tout.reshape((r, c))

    elif len(IN.shape) == 3:
        for ii in range(IN.shape[2]):
            tin = IN[:, :, ii].flatten()
            tout, _ = cg(A, tin, tol=0.1, maxiter=100)
            OUT[:, :, ii] = tout.reshape((r, c))

    return OUT


def septRelSmo(I, lamb, lb, hb, ite=3, thr=0.05, L1_0=None, prints=False):

    if len(I.shape) == 2:
        I = np.dstack([I] * 1)
        lb = np.dstack([lb] * 1)
        hb = np.dstack([hb] * 1)

    if L1_0 == None:
        L1_0 = copy.deepcopy(I)

    n, m, d = I.shape

    f1 = np.array([[1, -1]])
    f2 = np.array([[1],
                   [-1]])
    f3 = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

    sizeI2D = (n, m)
    otfFx = psf2otf(f1, sizeI2D)
    otfFy = psf2otf(f2, sizeI2D)
    otfL = psf2otf(f3, sizeI2D)

    fft2I = np.empty(I.shape).astype('complex128')
    for c in range(d):
        fft2I[:, :, c] = np.fft.fft2(I[:, :, c])

    Normin1 = np.power(np.dstack([np.absolute(otfL)] * d), 2) * fft2I

    Denormin1 = np.power(np.absolute(otfL), 2)

    Denormin2 = np.power(np.absolute(otfFx), 2) + np.power(np.absolute(otfFy), 2)

    if d > 1:
        Denormin1 = np.dstack([Denormin1] * d)
        Denormin2 = np.dstack([Denormin2] * d)

    eps = 10 ** (-16)

    L1 = L1_0

    for i in range(1, (ite + 1)):
        beta = 2 ** (i - 1) / thr
        if prints:
            print("Iteración con beta = " + str(beta))
        Denormin = lamb * Denormin1 + beta * Denormin2
        if len(Denormin.shape) == 2:
            Denormin = np.dstack([Denormin] * 1)

        if prints:
            print("Actualizando g, beta = " + str(beta))

        gFx = -cv2.filter2D(L1, -1, f1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_REFLECT_101)
        gFy = -cv2.filter2D(L1, -1, f2, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_REFLECT_101)

        if len(gFx.shape) == 2:
            gFx = np.dstack([gFx] * 1)
            gFy = np.dstack([gFy] * 1)


        for fila in range(n):
            for columna in range(m):
                t_x = 0
                t_y = 0
                for canal in range(d):
                    t_x += abs(gFx[fila, columna, canal])
                    t_y += abs(gFy[fila, columna, canal])
                    # gFx(t) = 0;
                if t_x < 1 / beta:
                    for canal in range(d):
                        gFx[fila, columna, canal] = 0
                if t_y < 1 / beta:
                    for canal in range(d):
                        gFy[fila, columna, canal] = 0

        if prints:
            print("Calculando L1, beta = " + str(beta))


        temp = gFx[:, m - 1, :] - gFx[:, 0, :]
        diff = -np.diff(gFx, 1, 1)
        Normin2 = np.empty((n, m, d))

        temporalAux = np.empty((n, 1, d))
        for c in range(d):
            temporalAux[:, 0, c] = temp[:, c]

        for dim in range(0, d):
            Normin2[:, :, dim] = np.column_stack((temporalAux[:, 0, dim], diff[:, :, dim]))


        temp = gFy[n - 1, :, :] - gFy[0, :, :]
        diff = -np.diff(gFy, 1, axis=0)
        Normin2_aux = np.empty((n, m, d))

        temporalAux = np.empty((1, m, d))
        for c in range(d):
            temporalAux[0, :, c] = temp[:, c]
        for dim in range(0, d):
            Normin2_aux[:, :, dim] = np.vstack((temporalAux[0, :, dim], diff[:, :, dim]))

        Normin2 = Normin2 + Normin2_aux

        Normin2 = Normin2.astype('complex128')
        for c in range(d):
            Normin2[:, :, c] = np.fft.fft2(Normin2[:, :, c])

        FL1 = (lamb * Normin1 + beta * Normin2) / (Denormin + eps)
        for c in range(d):
            L1[:, :, c] = np.real(np.fft.ifft2(FL1[:, :, c]))

        if prints:
            print("Aplicando normalización con gradiente descendente, beta = " + str(beta))

        for c in range(0, d):
            L1t = L1[:, :, c]
            for k in range(0, 500):
                dt = 0
                dt = dt + np.sum(L1t[L1t < lb[:, :, c]])
                dt = dt + np.sum(L1t[L1t > hb[:, :, c]])
                dt = dt * 2 / L1t.size
                L1t = L1t - dt
                if np.absolute(dt) < 1 / L1t.size:
                    break
            L1[:, :, c] = L1t

    t = L1 < lb
    L1[t] = lb[t]
    t = L1 > hb
    L1[t] = hb[t]

    L2 = I - L1
    return L1, L2


def get_lrs(input_i):
    input_i = im2doubleAux(input_i)
    lbLibro = np.zeros(input_i.shape)
    hbLibro = input_i
    J, R = septRelSmo(input_i, 50, lbLibro, hbLibro, ite=3, thr=0.15)
    j, G = septRelSmo(input_i, 25000, lbLibro, hbLibro, ite=3, thr=0.05)


    post = True
    if post:
        para = {'lambda': 0.15, 'sigma': 2, 'gamma': 0.8, 'solver': 1, 'strategy': 3}
        I, _, L = LIME(J * 2, para)
        L = np.concatenate((L[:, :, np.newaxis], L[:, :, np.newaxis], L[:, :, np.newaxis]), axis=2)

    return G, J, L