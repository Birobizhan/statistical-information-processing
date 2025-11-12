import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_test, levene

# Настройки
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
np.set_printoptions(precision=2, suppress=True)

# Загрузка данных
df = pd.read_excel("СОИ_2.xlsx", header=None)

# Присвоение заголовков
df.columns = [
    "Level1", "Area", "Level2", "Industry_group_name",
    "Level3", "Industry_code",
    "Total_enterprises", "Total_sales",
    "Companies_enterprises", "Companies_sales",
    "Others_enterprises", "Others_sales"
]

# Фильтрация: только отрасли 5-го уровня
df6 = df[df['Level3'] == 5].copy()

# Приведение к числовому типу и замена недостающих значений
num_cols = ["Companies_enterprises", "Companies_sales", "Others_enterprises", "Others_sales"]
df6[num_cols] = df6[num_cols].apply(pd.to_numeric, errors='coerce')

# Расчёт эффективности: выручка на предприятие
companies_eff = df6["Companies_sales"] / df6["Companies_enterprises"]
others_eff = df6["Others_sales"] / df6["Others_enterprises"]

# Удаляем бесконечности и NaN
companies_eff = companies_eff.replace([np.inf, -np.inf], np.nan).dropna()
others_eff = others_eff.replace([np.inf, -np.inf], np.nan).dropna()

# Оставляем только общие индексы
common_mask = (~companies_eff.isna()) & (~others_eff.isna())
companies_common = companies_eff[common_mask].values
others_common = others_eff[common_mask].values
print(df6.head())
print(df6.tail())
print(f"Количество отраслей уровня 5: {len(df6)}")
print(f"Корректных значений АО: {len(companies_common)}")
print(f"Корректных значений Остальных компаний: {len(others_common)}")


# 1. Распределения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(companies_common, bins=5, kde=True, color="steelblue")
plt.title("АО: выручка на предприятие")
plt.xlabel("Млн иен")
plt.ylabel('Количество компаний')

plt.subplot(1, 2, 2)
sns.histplot(others_common, bins=5, kde=True, color="orange")
plt.title("Остальные компании: выручка на предприятие")
plt.xlabel("Млн иен")
plt.ylabel('Количество компаний')
plt.tight_layout()
plt.show()

# 2. Boxplot (в логарифмической шкале из-за сильного разброса)
plt.figure()
data = [companies_common, others_common]
plt.boxplot(data, tick_labels=["АО", "Остальные компании"], patch_artist=True,
            boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red"))
plt.ylabel("Выручка на предприятие, млн иен")
plt.title("Сравнение эффективности")
plt.yscale("log")
plt.show()

results = {}

# z-критерий (большой объём, произвольные совокупности)
print("\nz-критерий:")
mean1, mean2 = np.mean(companies_common), np.mean(others_common)
std1, std2 = np.std(companies_common, ddof=1), np.std(others_common, ddof=1)
n1, n2 = len(companies_common), len(others_common)

# Стандартная ошибка разности средних
se = np.sqrt(std1**2 / n1 + std2**2 / n2)
z_stat = (mean1 - mean2) / se
p_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
results["z-test"] = {"stat": z_stat, "p": p_z}
print(f"z = {z_stat:.3f}, p = {p_z:.4f} → {'значимо' if p_z < 0.05 else 'не значимо'}")

# Спирмен — связь между профилями эффективности
print("\nКорреляция Спирмена:")
rho, p_spearman = stats.spearmanr(companies_common, others_common)
results["Spearman"] = {"rho": rho, "p": p_spearman}
print(f"ρ = {rho:.3f}, p = {p_spearman:.4f} → {'связь значима' if p_spearman < 0.05 else 'связи нет'}")

stat, p, med, tbl = median_test(companies_eff, others_eff)
print(f"\nМедианный критерий: \np = {p:.4f}, общая медиана = {med:.0f}")

# Манна-Уитни
print("\nМанна-Уитни:")
u_stat, p_u = stats.mannwhitneyu(companies_common, others_common, alternative='two-sided')
results["Mann-Whitney U"] = {"stat": u_stat, "p": p_u}
print(f"U = {u_stat:.0f}, p = {p_u:.4f} → {'значимо' if p_u < 0.05 else 'не значимо'}")

# Критерий Левена
stat, p = levene(companies_eff, others_eff, center='median')
print(f"\nКритерий Левена: \nstat={stat:.3f}, p={p:.4f}")

# Коэффициент вариации
cv_companies = np.std(companies_common, ddof=1) / np.mean(companies_common)
cv_others = np.std(others_common, ddof=1) / np.mean(others_common)
results["CV"] = {"АО": cv_companies, "Остальные компании": cv_others}
print(f"\nКоэффициент вариации (CV = std/mean):")
print(f"АО: CV = {cv_companies:.3f}")
print(f"Остальные компании: CV = {cv_others:.3f}")
