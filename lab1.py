import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

# Установка стиля для графиков
sns.set_theme(style="whitegrid")


def get_histogram(data, year_label):
    """
    Строит гистограмму распределения данных.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=7, color='skyblue')
    plt.title(f'Гистограмма распределения ОПЖ по регионам РФ в {year_label}', fontsize=16)
    plt.xlabel('Ожидаемая продолжительность жизни, лет', fontsize=12)
    plt.ylabel('Частота (количество регионов)', fontsize=12)
    plt.show()


def get_calc(data, year_label):
    """
    Вычисляет и выводит основные числовые характеристики выборки.
    """
    print(f"\nЧисловые характеристики в {year_label}")
    # 1.1 Среднее выборочное
    sample_mean = np.mean(data)
    print(f"1.1 Среднее выборочное: {sample_mean:.3f}")

    # 1.2 Выборочная дисперсия (смещенная и несмещенная)
    biased_variance = np.var(data, ddof=0)  # ddof=0 для смещенной
    unbiased_variance = np.var(data, ddof=1)  # ddof=1 для несмещенной
    print(f"1.2 Выборочная дисперсия (смещенная): {biased_variance:.3f}, несмещенная: {unbiased_variance:.3f}")

    # 1.3 Выборочное среднее квадратичное отклонение
    sample_std = np.std(data, ddof=1)
    print(f"1.3 Выборочное среднее квадратичное отклонение: {sample_std:.3f}")

    # 1.4 Медиана
    median = np.median(data)
    print(f"1.4 Медиана: {median:.3f}")

    # 1.5 Выборочное абсолютное отклонение
    mad_from_mean = np.mean(np.abs(data - sample_mean))
    print(f"1.5 Выборочное абсолютное отклонение: {mad_from_mean:.3f}")

    # 1.6 Квартили
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    print(f"1.6 Квартили: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")

    # 1.7 Интерквартильная широта
    iqr = q3 - q1
    print(f"1.7 Интерквартильная широта: {iqr:.3f}")

    # 1.8 Полусумма выборочных квартилей
    semi_sum_quartiles = (q1 + q3) / 2
    print(f"1.8 Полусумма выборочных квартилей: {semi_sum_quartiles:.3f}")

    # 1.9 Экстремальные элементы
    min_val = np.min(data)
    max_val = np.max(data)
    print(f"1.9 Экстремальные элементы: Min={min_val:.3f}, Max={max_val:.3f}")

    # 1.10 Размах выборки
    sample_range = max_val - min_val
    print(f"1.10 Размах выборки: {sample_range:.3f}")

    # 1.11 Полусумма экстремальных элементов
    semi_sum_extremes = (min_val + max_val) / 2
    print(f"1.11 Полусумма экстремальных элементов: {semi_sum_extremes:.3f}")

    # 1.12 Выборочная оценка асимметрии
    sample_skewness = stats.skew(data)
    print(f"1.12 Выборочная оценка асимметрии: {sample_skewness:.3f}")

    # 1.13 Выборочная оценка эксцесса
    sample_kurtosis = stats.kurtosis(data)  # Fisher (по умолчанию)
    print(f"1.13 Выборочная оценка эксцесса: {sample_kurtosis:.3f}")

    # Абсолютное отклонение выборочного среднего от медианы
    abs_dev_mean_median = abs(sample_mean - median)
    print(f"Абсолютное отклонение выборочного среднего от медианы: {abs_dev_mean_median:.3f}")

    # Абсолютное отклонение выборочного среднего квадратичного отклонения от половины интерквартильной широты
    abs_dev_std_half_iqr = abs(sample_std - (iqr / 2))
    print(
        f"Абсолютное отклонение выборочного среднего квадратичного отклонения от половины интерквартильной широты: {abs_dev_std_half_iqr:.3f}")

    return sample_mean, median, sample_std, q1, q3, min_val, max_val, semi_sum_quartiles, semi_sum_extremes


def plot_density_and_estimates(data, year_label, theoretical_pdf_func):
    """
    Строит график эмпирической и теоретической плотности, а также наносит точечные оценки среднего.
    """
    sample_mean = np.mean(data)
    median = np.median(data)
    sample_std = np.std(data, ddof=1)
    q1, _, q3 = np.percentile(data, [25, 50, 75])
    min_val = np.min(data)
    max_val = np.max(data)
    semi_sum_quartiles = (q1 + q3) / 2
    semi_sum_extremes = (min_val + max_val) / 2

    print(f"\n Оценка плотности и точечные оценки в {year_label}")

    # Оценивает отклонение эмпирической плотности распределения от теоретической плотности
    plt.figure(figsize=(10, 6))
    kde = stats.gaussian_kde(data)
    x_vals = np.linspace(min_val - sample_std, max_val + sample_std, 500)

    plt.plot(x_vals, kde(x_vals), color='red', linestyle='-', linewidth=2, label='Эмпирическая плотность')

    p0 = [sample_mean, sample_std]
    if p0[1] <= 0:
        p0[1] = 0.01
    bounds = ([-np.inf, 0.001], [np.inf, np.inf])

    fitted_params, pcov = curve_fit(theoretical_pdf_func, x_vals, kde(x_vals), p0=p0, bounds=bounds)
    plt.plot(x_vals, theoretical_pdf_func(x_vals, *fitted_params), color='blue',
             linestyle='--', linewidth=2, label=f'Теоретическая плотность')
    print(f"Параметры нормального распределения, подобранные методом наименьших квадратов: "
          f"mu={fitted_params[0]:.3f}, sigma={fitted_params[1]:.3f}")

    plt.title(f'Сравнение эмпирической и теоретической плотности в {year_label}', fontsize=16)
    plt.xlabel('Ожидаемая продолжительность жизни, лет', fontsize=12)
    plt.ylabel('Плотность', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    # Нанесение на линию значение различных точечных оценок среднего
    plt.figure(figsize=(12, 3))
    y_pos = 0.5

    plt.plot([min_val - 1, max_val + 1], [y_pos, y_pos], 'k-', linewidth=1, alpha=0.7)

    # Отметка различных оценок среднего
    plt.plot(sample_mean, y_pos, 'o', color='blue', markersize=10, label=f'Среднее ({sample_mean:.2f})')
    plt.plot(median, y_pos, 'X', color='red', markersize=10, label=f'Медиана ({median:.2f})')
    plt.plot(semi_sum_quartiles, y_pos, '^', color='green', markersize=10,
             label=f'Полусумма квартилей ({semi_sum_quartiles:.2f})')
    plt.plot(semi_sum_extremes, y_pos, 's', color='purple', markersize=10,
             label=f'Полусумма экстремумов ({semi_sum_extremes:.2f})')

    plt.title(f'Точечные оценки среднего в {year_label}', fontsize=16)
    plt.xlabel('Ожидаемая продолжительность жизни, лет', fontsize=12)
    plt.yticks([])  # Убираем метки на оси Y
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Теоретическая плотность распределения для нормального распределения
def normal_pdf(x, loc, scale):
    return stats.norm.pdf(x, loc=loc, scale=scale)


if __name__ == '__main__':
    df = pd.read_excel('СОИ_1.xls')

    # Обработка первого столбца для того, чтобы не было лишних пробелов
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

    condition1 = df.iloc[:, 1] == 'Оба пола'
    condition2 = df.iloc[:, 2] == 'все население'

    # Применение фильтров к исходному датафрейму
    filtered_df = df[condition1 & condition2]

    print("Просмотр отфильтрованных данных")
    print(filtered_df.head())
    print(filtered_df.tail())

    values_1990 = filtered_df.iloc[:, 3].dropna().values
    values_2022 = filtered_df.iloc[:, -1].dropna().values

    values_1990 = np.array(values_1990, dtype=float)
    values_2022 = np.array(values_2022, dtype=float)

    # Анализ для 1990 года
    print("Анализ данных за 1990 год")

    # Наглядное представление данных в виде гистограммы
    get_histogram(values_1990, '1990 году')

    # Числовые оценки числовых характеристик данных
    sample_mean_1990, median_1990, sample_std_1990, q1_1990, q3_1990, min_val_1990, max_val_1990, semi_sum_quartiles_1990, semi_sum_extremes_1990 = get_calc(
        values_1990, '1990 году')

    # Оценка плотности распределения и точечных оценок
    plot_density_and_estimates(values_1990, "1990 году", normal_pdf)

    # Проверка подтверждения гипотезы о нормальном характере распределения
    print("\n Проверка гипотезы о нормальном распределении (1990 год)")
    alpha_level = 0.05

    # Критерий Шапиро-Уилка
    shapiro_test_1990 = stats.shapiro(values_1990)
    print(f"Критерий Шапиро-Уилка p-значение={shapiro_test_1990.pvalue:.4f}")
    if shapiro_test_1990.pvalue < alpha_level:
        print(f"p < {alpha_level}: данные не распределены нормально")
    else:
        print(f"p >= {alpha_level}: данные могут быть распределены нормально")

    # Интервальные оценки числовых характеристик данных
    print("\n Интервальные оценки (Доверительные интервалы) для 1990 года")
    confidence_level = 0.95
    alpha = 1 - confidence_level
    n_1990 = len(values_1990)

    # Доверительный интервал для среднего
    se_1990 = sample_std_1990 / np.sqrt(n_1990)
    t_critical_1990 = stats.t.ppf(1 - alpha / 2, n_1990 - 1)
    ci_mean_1990 = (sample_mean_1990 - t_critical_1990 * se_1990,
                    sample_mean_1990 + t_critical_1990 * se_1990)
    print(f"{confidence_level * 100}% ДИ для среднего: ({ci_mean_1990[0]:.3f}, {ci_mean_1990[1]:.3f})")

    # Доверительный интервал для дисперсии и стандартного отклонения
    var_1990 = np.var(values_1990, ddof=1)
    chi2_lower_1990 = stats.chi2.ppf(alpha / 2, n_1990 - 1)
    chi2_upper_1990 = stats.chi2.ppf(1 - alpha / 2, n_1990 - 1)
    ci_var_1990 = ((n_1990 - 1) * var_1990 / chi2_upper_1990,
                   (n_1990 - 1) * var_1990 / chi2_lower_1990)
    print(f"{confidence_level * 100}% ДИ для дисперсии: ({ci_var_1990[0]:.3f}, {ci_var_1990[1]:.3f})")
    print(
        f"{confidence_level * 100}% ДИ для стандартного отклонения: ({np.sqrt(ci_var_1990[0]):.3f}, {np.sqrt(ci_var_1990[1]):.3f})")

    # Анализ для 2022 года
    print("Анализ данных за 2022 год")

    # Наглядное представление данных в виде гистограммы
    get_histogram(values_2022, "2022 году")

    # Числовые оценки числовых характеристик данных
    sample_mean_2022, median_2022, sample_std_2022, q1_2022, q3_2022, min_val_2022, max_val_2022, semi_sum_quartiles_2022, semi_sum_extremes_2022 = get_calc(
        values_2022, '2022 году')

    plot_density_and_estimates(values_2022, "2022 году", normal_pdf)

    # Проверка подтверждения гипотезы о характере распределения (Нормальное)
    print("\nПроверка гипотезы о нормальном распределении (2022 год)")

    # Критерий Шапиро-Уилка
    shapiro_test_2022 = stats.shapiro(values_2022)
    print(
        f"Критерий Шапиро-Уилка p-значение={shapiro_test_2022.pvalue:.4f}")
    if shapiro_test_2022.pvalue < alpha_level:
        print(f"p < {alpha_level}: данные не распределены нормально")
    else:
        print(f"p >= {alpha_level}: данные могут быть распределены нормально")

    # Интервальные оценки числовых характеристик данных
    print("\n Интервальные оценки (Доверительные интервалы) для 2022 года")
    n_2022 = len(values_2022)

    # Доверительный интервал для среднего
    se_2022 = sample_std_2022 / np.sqrt(n_2022)  # Стандартная ошибка среднего
    t_critical_2022 = stats.t.ppf(1 - alpha / 2, n_2022 - 1)
    ci_mean_2022 = (sample_mean_2022 - t_critical_2022 * se_2022,
                    sample_mean_2022 + t_critical_2022 * se_2022)
    print(f"{confidence_level * 100}% ДИ для среднего: ({ci_mean_2022[0]:.3f}, {ci_mean_2022[1]:.3f})")

    # Доверительный интервал для дисперсии и стандартного отклонения
    var_2022 = np.var(values_2022, ddof=1)
    chi2_lower_2022 = stats.chi2.ppf(alpha / 2, n_2022 - 1)
    chi2_upper_2022 = stats.chi2.ppf(1 - alpha / 2, n_2022 - 1)
    ci_var_2022 = ((n_2022 - 1) * var_2022 / chi2_upper_2022,
                   (n_2022 - 1) * var_2022 / chi2_lower_2022)
    print(f"{confidence_level * 100}% ДИ для дисперсии: ({ci_var_2022[0]:.3f}, {ci_var_2022[1]:.3f})")
    print(
        f"{confidence_level * 100}% ДИ для стандартного отклонения: ({np.sqrt(ci_var_2022[0]):.3f}, {np.sqrt(ci_var_2022[1]):.3f})")

    # Корреляция между годами
    # Корреляция между ОПЖ регионов в 1990 и 2022 годах
    print("\nКорреляция между ОПЖ регионов в 1990 и 2022 годах")

    correlation_1990_2022, p_value_corr = stats.pearsonr(values_1990, values_2022)
    print(f"Коэффициент корреляции Пирсона между ОПЖ регионов в 1990 и 2022 годами {correlation_1990_2022:.4f}")
    print(f"p-значение для корреляции: {p_value_corr:.4f}")
    if p_value_corr < 0.05:
        print("Корреляция статистически значима")
    else:
        print("Корреляция статистически не значима")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=values_1990, y=values_2022, s=50, alpha=0.7)
    plt.title('Корреляция ОПЖ регионов: 1990 и 2022 годы', fontsize=16)
    plt.xlabel('ОПЖ в 1990 году, лет', fontsize=12)
    plt.ylabel('ОПЖ в 2022 году, лет', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()