from scipy import stats
from scipy.stats import norm
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf as acf_sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Загрузка и подготовка данных
df = pd.read_csv("FAOSTAT_data_en_12-12-2025 (3).csv")
df = df[["Year", "Value"]].copy()
df = df.sort_values("Year").reset_index(drop=True)
df["Year"] = pd.to_numeric(df["Year"])
df["Value"] = pd.to_numeric(df["Value"])

print("Данные загружены. Период:", df["Year"].min(), "-", df["Year"].max())
print("Количество наблюдений:", len(df))

# Таблица: начало и конец
print("\n--- Таблица данных (начало и конец) ---")
head_tail = pd.concat([df.head(5), df.tail(5)])
print(head_tail.to_string(index=False))

# Визуализация временного ряда
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Value"], marker='o', markersize=3, linestyle='-', color='green')
plt.title("Производство кукурузы в США (1961–2023)", fontsize=14)
plt.xlabel("Год")
plt.ylabel("Производство, тонны")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Гистограмма распределения производства кукурузы
plt.figure(figsize=(10, 6))
plt.hist(df["Value"], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Гистограмма распределения годового производства кукурузы в США (1961–2023)")
plt.xlabel("Производство, тонны")
plt.ylabel("Частота (количество лет)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# Проверка стационарности и коррелограмма

fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(df["Value"], lags=15, ax=ax)
ax.set_title("Коррелограмма (ACF)")
plt.tight_layout()
plt.show()

y = df["Value"].values
n = len(y)
y_mean = np.mean(y)

max_lag = 15


acf_library = acf_sm(y, nlags=max_lag, fft=False)

print("Лаг ACF (statsmodels)")
print("-" * 45)
for k in range(max_lag + 1):
    print(f"{k:3d} {acf_library[k]:14.6f}")


# Сезонность и цикличность
# ------------------------------------------------------------
print("\n--- Сезонная и циклическая составляющая ---")
df["Trend"] = df["Value"].rolling(window=5, center=True).mean()

df["Cycle_Plus_Noise"] = df["Value"] - df["Trend"]

df["Cycle"] = df["Cycle_Plus_Noise"].rolling(window=3, center=True).mean()

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Cycle"], color='purple', label='Циклическая компонента')
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Циклическая составляющая производства кукурузы в США")
plt.xlabel("Год")
plt.ylabel("Отклонение от тренда, тонны")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Аномальные уровни — метод Ирвина
y = df["Value"].values
n = len(y)
y_mean = np.mean(y)
S = np.std(y, ddof=1)

# Максимальное отклонение
max_dev = np.max(np.abs(y - y_mean))
lambda_stat = max_dev / S

# Критическое значение для n = 63 (уровень значимости 0.05)
# Согласно таблицам: для n=50 λкр=1.37, для n=100 λкр=1.31 → для n=63 ≈ 1.36
lambda_crit = 1.36

print("\n--- Метод Ирвина: поиск аномальных уровней ---")
print(f"Среднее значение: {y_mean:,.0f}")
print(f"СКО (S): {S:,.0f}")
print(f"Максимальное отклонение: {max_dev:,.0f}")
print(f"λ = {lambda_stat:.3f}")
print(f"λ_кр (n={n}) ≈ {lambda_crit:.2f}")

if lambda_stat > lambda_crit:
    print("→ Есть аномальные уровни (λ > λ_кр)")
    threshold = lambda_crit * S
    anomalies_irwin = df[np.abs(df["Value"] - y_mean) > threshold].copy()
    print("\nАномальные годы (метод Ирвина):")
    print(anomalies_irwin[["Year", "Value"]].to_string(index=False))

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(df["Year"], df["Value"], color='green', label='Производство')
    plt.scatter(anomalies_irwin["Year"], anomalies_irwin["Value"],
                color='red', label='Аномалии (Ирвин)', zorder=5)
    plt.title("Аномальные уровни (метод Ирвина)")
    plt.xlabel("Год")
    plt.ylabel("Производство, тонны")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("→ Аномальных уровней не обнаружено (λ ≤ λ_кр)")
    anomalies_irwin = pd.DataFrame()


# Метод Фостера–Стюарта для проверки наличия тренда
def foster_stuart_test(y, alpha=0.05):
    n = len(y)
    if n < 3:
        raise ValueError("Недостаточно данных (n < 3)")

    p = np.zeros(n)
    q = np.zeros(n)

    # Первый элемент не участвует
    for t in range(1, n):
        y_prev = y[:t]
        if y[t] > np.max(y_prev):
            p[t] = 1
        if y[t] < np.min(y_prev):
            q[t] = 1

    P = np.sum(p)
    Q = np.sum(q)
    S = P + Q

    # Приближённые значения из статистических таблиц
    mu_S = 2.27 * np.log(n) - 0.67
    sigma_S = np.sqrt(2.80 * np.log(n) - 2.22)

    t_stat = (S - mu_S) / sigma_S

    # Критическое значение (двусторонний t-критерий, ~нормальное приближение)
    from scipy.stats import norm
    t_crit = norm.ppf(1 - alpha / 2)

    return {
        'n': n,
        'P': P,
        'Q': Q,
        'S': S,
        'mu_S': mu_S,
        'sigma_S': sigma_S,
        't_stat': t_stat,
        't_crit': t_crit,
        'has_trend': abs(t_stat) > t_crit
    }


result = foster_stuart_test(df["Value"].values)

print("\n--- Метод Фостера–Стюарта ---")
print(f"n = {result['n']}")
print(f"P = {result['P']:.0f} (число новых максимумов)")
print(f"Q = {result['Q']:.0f} (число новых минимумов)")
print(f"S = {result['S']:.0f}")
print(f"μ_S = {result['mu_S']:.2f}, σ_S = {result['sigma_S']:.2f}")
print(f"t = {result['t_stat']:.3f}")
print(f"t_кр (α={0.05}) = ±{result['t_crit']:.2f}")
if result['has_trend']:
    print("→ Есть статистически значимый тренд (|t| > t_кр)")
else:
    print("→ Тренд не обнаружен (|t| ≤ t_кр)")

# Сглаживание (скользящая средняя)
df["MA3"] = df["Value"].rolling(window=3, center=True).mean()
df["MA5"] = df["Value"].rolling(window=5, center=True).mean()
df['EMA_5'] = df['Value'].ewm(span=5, adjust=False).mean()
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Value"], alpha=0.5, label='Исходный ряд', color='gray')
plt.plot(df["Year"], df["MA3"], label='MA(3)', color='blue')
plt.plot(df["Year"], df["MA5"], label='MA(5)', color='red')
plt.plot(df["Year"], df["EMA_5"], label='EMA (Экспоненциальная, 5 лет)', linewidth=2, linestyle='--')
plt.title("Сглаживание временного ряда")
plt.xlabel("Год")
plt.ylabel("Производство, тонны")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


df["Smooth_Trend"] = df["Value"].rolling(window=5, center=True).mean()

y_smooth = df["Smooth_Trend"].dropna()

# Вычисление приростов
# Приросты 1-го порядка: delta y = y_t - y_{t-1}
diff_1 = y_smooth.diff().dropna()

# Приросты 2-го порядка: delta^2 y = delta y_t - delta y_{t-1}
diff_2 = diff_1.diff().dropna()

# Расчет характеристик устойчивости (Коэффициент вариации CV = sigma / |mean|)
# Чем меньше CV, тем "константнее" величина
cv_1 = diff_1.std() / abs(diff_1.mean())
cv_2 = diff_2.std() / abs(diff_2.mean())

print(f"Средний прирост 1-го порядка: {diff_1.mean():.2f} (CV = {cv_1:.2f})")
print(f"Средний прирост 2-го порядка: {diff_2.mean():.2f} (CV = {cv_2:.2f})")

# Логика выбора
selected_degree = 1
model_name = "Линейная"

if cv_1 < cv_2:
    print("\n>>> ВЫВОД: Приросты 1-го порядка более устойчивы (CV1 < CV2).")
    print(">>> Характер тренда: ЛИНЕЙНЫЙ (y = a + bt)")
    selected_degree = 1
    model_name = "Линейная"
else:
    print("\n>>> ВЫВОД: Приросты 2-го порядка более устойчивы (CV2 < CV1).")
    print(">>> Характер тренда: ПАРАБОЛИЧЕСКИЙ (y = a + bt + ct^2)")
    selected_degree = 2
    model_name = "Параболическая"


# Построение модели (МНК)

# Центрирование t
df["t"] = df["Year"] - df["Year"].iloc[0]
X = df["t"].values
y = df["Value"].values

coeffs = np.polyfit(X, y, deg=selected_degree)
df["Trend_Model"] = np.polyval(coeffs, X)
df["Residuals"] = df["Value"] - df["Trend_Model"]

# Расчет R^2
ss_res = np.sum((y - df["Trend_Model"]) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"\n--- Параметры модели (R^2 = {r2:.4f}) ---")
if selected_degree == 1:
    print(f"y = {coeffs[0]:.2f}*t + {coeffs[1]:.2f}")
else:
    print(f"y = {coeffs[0]:.4f}*t^2 + {coeffs[1]:.2f}*t + {coeffs[2]:.2f}")

# Комплексная оценка остаточной компоненты
residuals = df["Residuals"].values
n = len(residuals)
mean_res = np.mean(residuals)
std_res = np.std(residuals, ddof=1)

print("\n" + "="*50)
print(" ПОДРОБНЫЙ АНАЛИЗ ОСТАТКОВ (ПУНКТЫ 2.1 - 2.3)")
print("="*50)

# Проверка на случайность: Критерий серий (Runs Test)
signs = np.sign(residuals)
signs = signs[signs != 0]
runs = 1
for i in range(1, len(signs)):
    if signs[i] != signs[i-1]:
        runs += 1

n_pos = np.sum(signs > 0)
n_neg = np.sum(signs < 0)
exp_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
std_runs = np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / ((n_pos + n_neg)**2 * (n_pos + n_neg - 1)))
z_runs = (runs - exp_runs) / std_runs

print(f"\n1. ПРОВЕРКА НА СЛУЧАЙНОСТЬ (Критерий серий):")
print(f"   Фактически серий: {runs}")
print(f"   Ожидалось (E): {exp_runs:.1f}")
print(f"   Z-статистика: {z_runs:.2f}")
if abs(z_runs) < 1.96:
    print("   [OK] Остатки случайны (нет автокорреляции знаков).")
else:
    print("   [!] Остатки НЕ случайны (присутствует зависимость).")

# 4.2. Проверка на нормальность: RS-критерий
R_range = np.max(residuals) - np.min(residuals) # Размах
RS_stat = R_range / std_res

# Табличные границы для альфа=0.05 и n=60 (примерно)
rs_lower, rs_upper = 3.0, 4.5

print(f"\n2. ПРОВЕРКА НА НОРМАЛЬНОСТЬ (RS-критерий):")
print(f"   Размах (R): {R_range:,.0f}")
print(f"   Станд. откл (S): {std_res:,.0f}")
print(f"   RS = R/S: {RS_stat:.2f}")
print(f"   Ориентировочные границы для n={n}: [{rs_lower} ... {rs_upper}]")

if rs_lower <= RS_stat <= rs_upper:
    print("   [OK] Значение RS внутри нормального диапазона.")
else:
    print("   [!] Значение RS выходит за границы (есть выбросы или аномальное распределение).")

# Дополнительно Shapiro-Wilk (более точный)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"   (Контроль) Shapiro-Wilk p-value: {shapiro_p:.4f}")

# 4.3. Проверка мат. ожидания остатков (t-критерий Стьюдента)
# residuals, n, std_res (S) уже определены из предыдущих блоков

mean_res = np.mean(residuals)
n_res = len(residuals)

# Формула: t = (E_bar - 0) / (S / sqrt(n))
# Важно: используем несмещенное стандартное отклонение std_res (ddof=1)
t_stat_mean = mean_res / (std_res / np.sqrt(n_res))

# Критическое значение t (для двустороннего теста, alpha=0.05)
t_crit = stats.t.ppf(0.975, df=n_res - 1)

print(f"\n3. ПРОВЕРКА МАТ. ОЖИДАНИЯ (t-критерий Стьюдента):")
print(f"   Среднее остатков: {mean_res:,.2f}")
print(f"   t-статистика: {t_stat_mean:.3f}")
print(f"   Критическое t (α=0.05, df={n_res-1}): ±{t_crit:.2f}")

if abs(t_stat_mean) < t_crit:
    print("   [OK] Гипотеза E[E] = 0 принимается (систематическая ошибка отсутствует).")
else:
    print("   [!] Гипотеза E[E] = 0 отвергается (модель имеет систематическое смещение).")

# 4.4. Критерий Дарбина-Уотсона
diff_res = np.diff(residuals)
dw_stat = np.sum(diff_res**2) / np.sum(residuals**2)
print(f"\n4. АВТОКОРРЕЛЯЦИЯ (Дарбин-Уотсон):")
print(f"   DW = {dw_stat:.2f}")
if 1.5 <= dw_stat <= 2.5:
    print("   [OK] Автокорреляция отсутствует.")
else:
    print("   [!] Есть автокорреляция.")

# ВИЗУАЛИЗАЦИЯ (Остаточная компонента)

plt.figure(figsize=(10, 6))
plt.plot(df["Year"], df["Residuals"], color='purple', marker='o', markersize=4)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title('График остатков (E_t)')
plt.ylabel('Ошибка (тонны)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black')
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
p = norm.pdf(x_axis, mean_res, std_res)
plt.plot(x_axis, p, 'k', linewidth=2, label='Норм. распр.')
plt.title('Гистограмма распределения остатков')
plt.legend()
plt.grid(True)
plt.show()


# Расчет дополнительных метрик точности РАСЧЕТ ДОПОЛНИТЕЛЬНЫХ МЕТРИК ТОЧНОСТИ

residuals = df["Residuals"].values
y_true = df["Value"].values
y_pred = df["Trend_Model"].values
n = len(residuals)

# MAE (Mean Absolute Error)
mae = np.mean(np.abs(residuals))

# MSE (Mean Squared Error)
mse = np.mean(residuals**2)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# MAPE (Mean Absolute Percentage Error)
# Защита от деления на ноль (хотя в вашем ряду это маловероятно)
mape = np.mean(np.abs(residuals / y_true)) * 100

# SMAPE (Symmetric Mean Absolute Percentage Error)
# Формула: (1/n) * sum(|Et| / (|yt| + |y_hat_t|) * 200)
smape_numerator = np.abs(y_pred - y_true)
smape_denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
# Обработка случая, когда y_true и y_pred одновременно равны нулю (что тут невозможно)
smape = np.mean(smape_numerator / smape_denominator) * 100


print("\n" + "="*50)
print("Результаты: ")
print("="*50)
print(f"1. MAE (Средняя абс. ошибка): {mae:,.0f} тонн")
print(f"2. RMSE (Корень из MSE): {rmse:,.0f} тонн")
print(f"3. MAPE (Абс. ошибка в %): {mape:.2f}%")
print(f"4. SMAPE (Симм. ошибка в %): {smape:.2f}%")
print("="*50)
