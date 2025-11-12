import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from sympy import *
import matplotlib.pyplot as plt
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import *
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


n = 0
f = []
Rab = []
a = 0
b = 0

Rx = []
Ry = []
fx = []
xa = []
xb = []
x = []
cur = [[]]
NUM_POINTS = 50
t1 = []
ans = []
f_i = 0
R_i = 0

f_text = ['', '', '', '', '', '']
R_text = ['', '', '', '', '', '']
a_text = ''
b_text = ''
ts_text = ''
p0_text = ''
J_f_text = ''
ax1 = ''
ax2 = ''


class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.help_window()

    def help_window(self):
        self.setWindowTitle("Help")
        self.setFixedSize(QSize(515, 400))
        lbl1 = QLabel("\n  При введении задачи в систему необходимо придерживаться\n\n  следующих правил:\n\n  1) "
                      "Переменные "
                      "задаются как x1, x2 и т.д;\n\n  2) Все уравнения необходимо вводить вручную, без пробелов,\n\n  "
                      "так как это бы потребовалось ввести в python(например, степень - **);\n\n  3) Краевые условия "
                      "необходимо вводить в виде: xa1, xb2\n\n  (здесь a - левая граница t, b - правая);\n\n  4) При "
                      "вводе "
                      "функционала необходимо указать\n\n  подынтегральную функцию f0.\n\n\n  Выполнила: студентка 313 "
                      "группы "
                      "Асанова Илона.", self)

        self.show()


class MyWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        global f_text, R_text, a_text, b_text, ts_text, p0_text, J_f_text
        self.setWindowTitle('Метод продолжения по параметру для решения краевой задачи')
        # self.setFixedSize(QSize(515, 400))
        widget = QWidget()
        page = QVBoxLayout()

        diffInputLayout = QHBoxLayout()

        l1 = QVBoxLayout()
        diffLayout = QFormLayout()
        lbl1 = QLabel("Введите систему дифференциальных уравнений:")
        difflbl = ["dx1/dt = ", "dx2/dt = ", "dx3/dt = ", "dx4/dt = ", "dx5/dt = ", "dx6/dt = "]

        diffline = QLineEdit()
        diffLayout.addRow(difflbl[0], diffline)
        diffline.textEdited.connect(self.save_f1)
        diffline = QLineEdit()
        diffLayout.addRow(difflbl[1], diffline)
        diffline.textEdited.connect(self.save_f2)
        diffline = QLineEdit()
        diffLayout.addRow(difflbl[2], diffline)
        diffline.textEdited.connect(self.save_f3)
        diffline = QLineEdit()
        diffLayout.addRow(difflbl[3], diffline)
        diffline.textEdited.connect(self.save_f4)
        diffline = QLineEdit()
        diffLayout.addRow(difflbl[4], diffline)
        diffline.textEdited.connect(self.save_f5)
        diffline = QLineEdit()
        diffLayout.addRow(difflbl[5], diffline)
        diffline.textEdited.connect(self.save_f6)
        l1.addWidget(lbl1)
        l1.addLayout(diffLayout)

        l2 = QVBoxLayout()
        boundLayout = QFormLayout()
        lbl2 = QLabel("Введите систему краевых условий:")
        boundlbl = ["R1(a, b) = ", "R2(a, b) = ", "R3(a, b) = ", "R4(a, b) = ", "R5(a, b) = ", "R6(a, b)"]

        boundline = QLineEdit()
        boundLayout.addRow(boundlbl[0], boundline)
        boundline.textEdited.connect(self.save_R1)
        boundline = QLineEdit()
        boundLayout.addRow(boundlbl[1], boundline)
        boundline.textEdited.connect(self.save_R2)
        boundline = QLineEdit()
        boundLayout.addRow(boundlbl[2], boundline)
        boundline.textEdited.connect(self.save_R3)
        boundline = QLineEdit()
        boundLayout.addRow(boundlbl[3], boundline)
        boundline.textEdited.connect(self.save_R4)
        boundline = QLineEdit()
        boundLayout.addRow(boundlbl[4], boundline)
        boundline.textEdited.connect(self.save_R5)
        boundline = QLineEdit()
        boundLayout.addRow(boundlbl[5], boundline)
        boundline.textEdited.connect(self.save_R6)
        l2.addWidget(lbl2)
        l2.addLayout(boundLayout)

        paramLayout = QFormLayout()
        lbl3 = QLabel("Введите начальное время      ")
        t_start = QLineEdit()
        paramLayout.addRow(lbl3, t_start)
        t_start.textEdited.connect(self.save_a)
        lbl4 = QLabel("Введите конечное время       ")
        t_end = QLineEdit()
        paramLayout.addRow(lbl4, t_end)
        t_end.textEdited.connect(self.save_b)
        lbl5 = QLabel("Введите t*                   ")
        tsr = QLineEdit()
        paramLayout.addRow(lbl5, tsr)
        tsr.textEdited.connect(self.save_ts)
        lbl6 = QLabel("Введите начальное приближение")
        pl = QLineEdit()
        paramLayout.addRow(lbl6, pl)
        pl.textEdited.connect(self.save_p0)

        diffInputLayout.addLayout(l1)
        diffInputLayout.setSpacing(10)
        diffInputLayout.addLayout(l2)
        diffInputLayout.setSpacing(20)
        diffInputLayout.addLayout(paramLayout)
        diffInputLayout.setSpacing(20)

        do_btn = QPushButton("Вычислить")
        do_btn.clicked.connect(self.do_btn_func)

        self.pbar = QProgressBar(self)

        jLayout = QHBoxLayout()

        JForm = QFormLayout()
        jj = QLineEdit()
        JForm.addRow("Введите подынтегральную функцию f0", jj)
        jj.textEdited.connect(self.save_J)
        J_btn = QPushButton("Вычислить")
        J_btn.clicked.connect(self.J_btn_func)
        jLayout.addLayout(JForm)
        jLayout.addWidget(J_btn)

        btnLayout = QVBoxLayout()
        graph_btn = QPushButton("Построить график")
        graph_btn.clicked.connect(self.graph_btn_func)
        axesLayout = QHBoxLayout()
        self.axes_choise1 = QComboBox(self)
        self.axes_choise2 = QComboBox(self)
        self.axes_choise1.addItem('t')
        self.axes_choise1.addItem('x1')
        self.axes_choise1.addItem('x2')
        self.axes_choise1.addItem('x3')
        self.axes_choise1.addItem('x4')
        self.axes_choise1.addItem('x5')
        self.axes_choise1.addItem('x6')
        self.axes_choise2.addItem('t')
        self.axes_choise2.addItem('x1')
        self.axes_choise2.addItem('x2')
        self.axes_choise2.addItem('x3')
        self.axes_choise2.addItem('x4')
        self.axes_choise2.addItem('x5')
        self.axes_choise2.addItem('x6')

        axesLayout.addWidget(self.axes_choise1)
        axesLayout.addWidget(self.axes_choise2)
        btnLayout.addLayout(axesLayout)
        btnLayout.addWidget(graph_btn)

        page.addLayout(diffInputLayout)
        page.addWidget(do_btn)
        page.addWidget(self.pbar)
        page.addLayout(jLayout)
        page.addLayout(btnLayout)
        widget.setLayout(page)
        self.setCentralWidget(widget)

    def graph_btn_func(self):
        global ax1, ax2
        ax1 = self.axes_choise1.currentText()
        ax2 = self.axes_choise2.currentText()
        x1 = 0
        x2 = 0
        if ax1 == 't':
            x1 = t1
        elif ax2 == 't':
            x2 = t1
        for i in range(6):
            if ax1 == ('x' + str(i + 1)):
                x1 = ans[:, i]
            if ax2 == ('x' + str(i + 1)):
                x2 = ans[:, i]
        plt.figure()
        plt.grid()
        plt.plot(x1, x2)
        plt.show()

    def J_btn_func(self):
        count_J()

    def do_btn_func(self):
        solution()

    def save_f1(self, text):
        global f_text
        f_text[0] = text

    def save_f2(self, text):
        global f_text
        f_text[1] = text

    def save_f3(self, text):
        global f_text
        f_text[2] = text

    def save_f4(self, text):
        global f_text
        f_text[3] = text

    def save_f5(self, text):
        global f_text
        f_text[4] = text

    def save_f6(self, text):
        global f_text
        f_text[5] = text

    def save_R1(self, text):
        global R_text
        R_text[0] = text

    def save_R2(self, text):
        global R_text
        R_text[1] = text

    def save_R3(self, text):
        global R_text
        R_text[2] = text

    def save_R4(self, text):
        global R_text
        R_text[3] = text

    def save_R5(self, text):
        global R_text
        R_text[4] = text

    def save_R6(self, text):
        global R_text
        R_text[5] = text

    def save_a(self, text):
        global a_text
        a_text = text

    def save_b(self, text):
        global b_text
        b_text = text

    def save_ts(self, text):
        global ts_text
        ts_text = text

    def save_p0(self, text):
        global p0_text
        p0_text = text

    def save_J(self, text):
        global J_f_text
        J_f_text = text


def func_subs(y, t):
    tt = Symbol("t")
    x = []
    for i in range(1, 7):
        x.append(Symbol("x" + str(i)))
    global n
    res = np.zeros(n)
    for i in range(n):
        tmp_f = f[i].subs(tt, t)
        for j in range(n):
            tmp_f = tmp_f.subs(x[j], y[j])
        res[i] = tmp_f.evalf()
    return res


def count_J():
    fj = parse_expr(J_f_text)
    x = []
    for i in range(1, n + 1):
        x.append(Symbol("x" + str(i)))
    m = len(ans)
    tt = Symbol("t")
    f0 = np.zeros(m)
    for i in range(m):
        tmp_f = fj.subs(tt, t1[i])
        for j in range(n):
            tmp_f = tmp_f.subs(x[j], ans[i][j])
        f0[i] = tmp_f.evalf()
    h = (b - a) / m
    res = h * (sum(f0) - f0[0])
    print(res)
    return res


def Fi(p0, t0):
    t = np.linspace(t0, a)
    res_ode = odeint(func_subs, p0, t)
    x_a_p0 = res_ode[-1]
    t = np.linspace(t0, b)
    res_ode = odeint(func_subs, p0, t)
    x_b_p0 = res_ode[-1]

    xa = []
    xb = []
    for i in range(1, 7):
        xa.append(Symbol("xa" + str(i)))
        xb.append(Symbol("xb" + str(i)))
    res = np.zeros(n)
    for i in range(n):
        tmp_func = Rab[i]
        for j in range(n):
            tmp_func = tmp_func.subs(xa[j], x_a_p0[j])
            tmp_func = tmp_func.subs(xb[j], x_b_p0[j])
        res[i] = tmp_func.evalf()
    return res


def diff_R(p, t0):
    x_a_p = cur[0]
    x_b_p = cur[-1]
    dRdx = np.zeros((n, n))
    dRdy = np.zeros((n, n))
    xa = []
    xb = []
    for i in range(n):
        xa.append(Symbol("xa"+str(i + 1)))
        xb.append(Symbol("xb" + str(i + 1)))
    for i in range(n):
        for k in range(n):
            tmp_fun = Rx[k][i]
            for j in range(n):
                tmp_fun = tmp_fun.subs(xa[j], x_a_p[j])
                tmp_fun = tmp_fun.subs(xb[j], x_b_p[j])
            dRdx[k][i] = tmp_fun.evalf()
    for i in range(n):
        for k in range(n):
            tmp_fun = Ry[k][i]
            for j in range(n):
                tmp_fun = tmp_fun.subs(xa[j], x_a_p[j])
                tmp_fun = tmp_fun.subs(xb[j], x_b_p[j])
            dRdy[k][i] = tmp_fun.evalf()
    return [dRdx, dRdy]


def Xrhs(tt, y):
    t = Symbol("t")
    x = []
    x_c = cur[ind(tt)]
    for i in range(1, 7):
        x.append(Symbol("x" + str(i)))
    dfdxp = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            tmp_fun = fx[k][i]
            tmp_fun.subs(t, tt)
            for j in range(n):
                tmp_fun = tmp_fun.subs(x[j], x_c[j])
            dfdxp[k][i] = tmp_fun.evalf()
    return np.matmul(dfdxp, np.array(y))


def X(p, a, b):
    res = []
    t = np.linspace(a, b, num=5)
    for i in range(n):
        init = np.zeros(n)
        init[i] = 1
        res.append(solve_ivp(Xrhs, [a, b], init).y[:, -1])
    return np.array(res).transpose()


def ind(t):
    res = int(np.floor((t - a) * (NUM_POINTS - 1) / (b - a)))
    if res < 0:
        res = 0
    if res > NUM_POINTS - 1:
        res = NUM_POINTS - 1
    return res


def diff_Fi(p, t0):
    t = np.linspace(a, b, num=NUM_POINTS)
    global cur
    cur = odeint(func_subs, p, np.flip(t[0:(ind(t0) + 1)]))
    cur = np.flip(cur, 0)
    cur = np.append(cur, odeint(func_subs, p, t[ind(t0):])[1:], axis=0)

    R = diff_R(p, t0)
    Xa = X(p, t0, a)
    Xb = X(p, t0, b)
    res = np.matmul(R[0], Xa) + np.matmul(R[1], Xb)
    return res


def prhs(mu, p, t0, fi):
    print(p, " ", mu)
    if window.pbar.value() < floor(mu * 100):
        window.pbar.setValue(floor(mu * 100))
    f_inv = np.linalg.inv(diff_Fi(p, t0))
    return -np.matmul(f_inv, fi)


def solution():
    global f_text, R_text, a_text, b_text, ts_text, p0_text
    global a, b
    global ans, t1
    global f, Rab
    f = []
    Rab = []
    for i in range(6):
        if f_text[i] != "":
            f.append(parse_expr(f_text[i], evaluate=True))
    for i in range(6):
        if R_text[i] != "":
            Rab.append(parse_expr(R_text[i], evaluate=True))
    global n

    n = len(f)

    a = eval(a_text)
    b = eval(b_text)
    p0 = np.array(eval(p0_text))
    ts = eval(ts_text)

    # test()
    # p0 = [2, 0, 2*3.1415926, 2]
    # ts = 0

    global xa, xb, x, Rx, Ry, fx
    xa = []
    xb = []
    x = []
    for i in range(1, n + 1):
        xa.append(Symbol("xa" + str(i)))
        xb.append(Symbol("xb" + str(i)))
        x.append(Symbol("x" + str(i)))
    for i in range(n):
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for k in range(n):
            tmp1.append(diff(Rab[i], xa[k]))
            tmp2.append(diff(Rab[i], xb[k]))
            tmp3.append(diff(f[i], x[k]))
        Rx.append(tmp1)
        Ry.append(tmp2)
        fx.append(tmp3)

    fi = Fi(p0, ts)
    rhs = lambda tt, y: prhs(tt, y, ts, fi)
    mu = np.linspace(0, 1, num=3)
    p_res = solve_ivp(rhs, [0,1], p0, method='RK23').y[:, -1]
    t1 = np.linspace(ts, a)
    ans = odeint(func_subs, p_res, t1)
    ans = np.flip(ans, 0)
    t = np.linspace(ts, b)
    ans = np.append(ans, odeint(func_subs, p_res, t), axis=0)
    t1 = np.flip(t1)
    t1 = np.append(t1, t)
    print(p_res)


app1 = QApplication(sys.argv)
helpwindow = HelpWindow()
helpwindow.show()
app1.exec()

app2 = QApplication(sys.argv)
window = MyWindow()
window.show()
app2.exec()
