import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import cos, exp
from math import pi


# 正弦波クラス
#   - A: 振幅（-1〜1）
#   - theta: 初期位相（-π〜π）
#   - omega: 角周波数（周波数 f が指定されている場合はそちらが優先される）
#   - f: 周波数 [Hz]
class sin_wave:

    def __init__(self, A, theta, omega=0, f=None):
        self.A = A  # 振幅（-1〜1）
        self.theta = theta  # 初期位相（-π〜π）
        if f is None:
            self.f = omega / (2 * pi)
            self.omega = omega
        else:
            self.f = f  # 周波数 [Hz]
            self.omega = 2 * pi * f  # 角周波数 [rad/秒]

    def __call__(self, t):
        return self.A * cos(self.omega * t + self.theta)  # 時刻 t における信号値


# 複素指数信号クラス
#   - A: 振幅（-1〜1）
#   - theta: 初期位相（-π〜π）
#   - omega: 角周波数（周波数 f が指定されている場合はそちらが優先される）
#   - f: 周波数 [Hz]
class complex_exp:

    def __init__(self, A, theta, omega=0, f=None):
        self.A = A  # 振幅（-1〜1）
        self.theta = theta  # 初期位相（-π〜π）
        if f is None:
            self.f = omega / (2 * pi)
            self.omega = omega
        else:
            self.f = f  # 周波数 [Hz]
            self.omega = 2 * pi * f  # 角周波数 [rad/秒]

    def __call__(self, t):
        return self.A * exp(1j * (self.omega * t + self.theta))  # 時刻 t における信号値（1j は python における虚数単位の表記）


# 単位ステップ信号
def u(t):
    '''
    # 時刻 t における信号値を次式で決定
    if t < 0:
        return 0
    else:
        return 1
    '''
    return np.where(t < 0, 0, 1)  # 上の処理を一行で表現したもの


# サンプリング
#   - x: 関数（連続時間信号 x(t) と解釈して用いる）
#   - nrange: 出力の離散時間信号がカバーする区間（nrange=[a, b] のとき，n=a, a+1, ... , b-2, b-1 と考え，x[a]〜x[b-1] までをサンプリングする）
#   - fs: サンプリング周波数 [Hz]（サンプリング周期 T が指定されている場合はそちらが優先される）
#   - T: サンプリング周期 [秒]
def sampling(x, nrange, fs=44100, T=0):
    if T != 0:
        fs = 1 / T
    n = np.arange(nrange[0], nrange[1])
    x_ = x(n / fs)  # サンプリング
    if nrange[0] != 0:
        tmp = x_
        x_ = {}
        for n in range(len(tmp)):
            x_[n + nrange[0]] = tmp[n]
    return x_


# 信号 x を離散時間信号と解釈してグラフ表示する
#   - x: 表示対象の信号（配列または辞書）
#   - nrange: 横軸の表示範囲（nrange=[a, b] のとき，x[a] から x[b-1] までを表示）
#   - vrange: 縦軸の表示範囲（デフォルトでは -1〜1）
#   - title: グラフタイトル
#   - s: 何秒後にグラフを閉じるか（s<=0 のときは手動で閉じる）
def show_discrete_signal(x, nrange, vrange=[-1.0, 1.0], title='Sample Signal', s=0):

    # 縦軸の目盛りの定義
    vticks = [0.0] * 5
    for i in range(len(vticks)):
        vticks[i] = vrange[0] + (vrange[1] - vrange[0]) * i / (len(vticks) - 1)

    # グラフを作成
    n = np.arange(nrange[0], nrange[1])
    if type(x) == dict:
        x_ = np.asarray([(x[n] if n in x.keys() else 0) for n in range(nrange[0], nrange[1])])
    else:
        x_ = np.asarray([(x[n] if 0 <= n and n < len(x) else 0) for n in range(nrange[0], nrange[1])])
    if x_.dtype == np.complex128:
        # 複素信号の場合
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel('n')
        ax.set_ylabel('signal value (real)')
        ax.set_zlabel('signal value (imag)')
        ax.set_yticks(vticks)
        ax.set_zticks(vticks)
        ax.set_ylim(vrange)
        ax.set_zlim(vrange)
        ax.grid(True)
        for i in range(len(n)):
            plt.plot([n[i], n[i]], [0, x_[i].real], [0, x_[i].imag], color='tab:red')
        plt.plot(n, x_.real, x_.imag, linewidth=0.5, linestyle='dashed')
    else:
        # 実信号の場合
        plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.xlabel('n')
        plt.ylabel('signal value')
        plt.yticks(vticks)
        plt.ylim(vrange)
        plt.grid()
        for i in range(len(n)):
            plt.plot([n[i], n[i]], [0, x_[i]], color='tab:red')
        plt.plot(n, x_, linewidth=0.5, linestyle='dashed')

    # グラフを表示
    if s > 0:
        plt.pause(s)
        plt.close()
    else:
        plt.show()


# 信号 x を連続時間信号と解釈してグラフ表示する
#   - x: 表示対象の信号（関数）
#   - trange: 横軸の表示範囲（trange=[a, b] のとき，a 秒目から b 秒目までを表示）
#   - vrange: 縦軸の表示範囲（デフォルトでは -1〜1）
#   - title: グラフタイトル
#   - fs: サンプリング周波数 [Hz]
#   - s: 何秒後にグラフを閉じるか（s<=0 のときは手動で閉じる）
def show_continuous_signal(x, trange, vrange=[-1.0, 1.0], title='Sample Signal', fs=44100, s=0):

    # 縦軸の目盛りの定義
    vticks = [0.0] * 5
    for i in range(len(vticks)):
        vticks[i] = vrange[0] + (vrange[1] - vrange[0]) * i / (len(vticks) - 1)

    # グラフを作成
    t = np.arange(trange[0], trange[1], 1/fs)
    data = x(t)
    if data.dtype == np.complex128:
        # 複素信号の場合
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel('time [sec.]')
        ax.set_ylabel('signal value (real)')
        ax.set_zlabel('signal value (imag)')
        ax.set_yticks(vticks)
        ax.set_zticks(vticks)
        ax.set_ylim(vrange)
        ax.set_zlim(vrange)
        ax.grid(True)
        ax.plot(t, data.real, data.imag)
    else:
        # 実信号の場合
        plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.xlabel('time [sec.]')
        plt.ylabel('signal value')
        plt.yticks(vticks)
        plt.ylim(vrange)
        plt.grid()
        plt.plot(t, data)

    # グラフを表示
    if s > 0:
        plt.pause(s)
        plt.close()
    else:
        plt.show()


# 信号 x を離散時間信号と解釈し，その振幅スペクトルをグラフ表示する
#   - x: 表示対象の信号（配列）
#   - fs: サンプリング周波数 [Hz]
#   - frange: 横軸の表示範囲（-frange [Hz] から frange [Hz] までを表示）
#   - vrange: 縦軸の表示範囲（デフォルトでは 0〜1）
#   - title: グラフタイトル
#   - s: 何秒後にグラフを閉じるか（s<=0 のときは手動で閉じる）
def show_discrete_amplitude_spectrum(x, fs, frange, vrange=[0.0, 1.0], title='Amplitude Spectrum', s=0):

    # 縦軸の目盛りの定義
    vticks = [0.0] * 5
    for i in range(len(vticks)):
        vticks[i] = vrange[0] + (vrange[1] - vrange[0]) * i / (len(vticks) - 1)

    # 振幅スペクトルを計算
    N = len(x)  # 信号長
    c = np.abs(np.fft(x)) / N # フーリエ係数を求め，その絶対値を取得
    Nf = (frange * N) // fs
    if Nf < N // 2:
        s = np.concatenate((c[-Nf : ], c[ : Nf]), axis=0)
    else:
        s = np.zeros(2 * Nf)
        s[ : N // 2] = c[ : N // 2]
        s[-N // 2 : ] = c[-N // 2 : ]
        s = np.concatenate((s[-Nf : ], s[ : Nf]), axis=0)

    # グラフを作成
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.yticks(vticks)
    plt.ylim(vrange)
    plt.grid()
    plt.plot(np.arange(-frange, frange, fs / N), s)

    # グラフを表示
    if s > 0:
        plt.pause(s)
        plt.close()
    else:
        plt.show()
