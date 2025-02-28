import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from PIL import Image

# 设置CPU核心数
cpu_num = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import torch

torch.set_num_threads(cpu_num)

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 数据读取和预处理函数
def read_and_preprocess_data(file_path):
    print("开始读取和预处理数据...")
    data = pd.read_excel(file_path)
    print("数据读取完成，开始检查缺失值和描述性统计...")

    # 替换掉不需要的字符
    data.replace(['#VALUE!', '/'], np.nan, regex=True, inplace=True)

    # 将对象类型的列转换为数值类型，无法转换的变为NaN
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 删除包含NaN的行
    data.dropna(inplace=True)  # 正确使用data变量

    # 打印缺失值数量
    print(data.isnull().sum())
    data.describe()

    # 绘制相关性热图
    corr_matrix = data.corr()
    save_heatmap(corr_matrix, 'person_correlation_heatmap.tif', 'Correlation Heatmap')

    with PdfPages('correlation_heatmap.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"size": 6})
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        fig.set_dpi(150)
        pdf.savefig(fig)
        plt.close()

    spearman_corr_matrix = data.corr(method='spearman')
    save_heatmap(spearman_corr_matrix, 'spearman_correlation_heatmap.tif', 'Spearman Correlation Heatmap')

    print("数据预处理完成。")
    return data


# 保存热图函数
def save_heatmap(corr_matrix, file_path, title):
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.savefig(file_path, dpi=1200, format='tif')
    plt.show()


# 模型评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"开始评估{model_name}模型...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"{model_name} 训练集评价指标:")
    print("均方误差 (MSE):", mse_train)
    print("均方根误差 (RMSE):", rmse_train)
    print("平均绝对误差 (MAE):", mae_train)
    print("拟合优度 (R - squared):", r2_train)
    print("\n")
    print(f"{model_name} 测试集评价指标:")
    print("均方误差 (MSE):", mse_test)
    print("均方根误差 (RMSE):", rmse_test)
    print("平均绝对误差 (MAE):", mae_test)
    print("拟合优度 (R - squared):", r2_test)
    print("\n" + "-" * 40 + "\n")
    print(f"{model_name}模型评估完成。")
    return mse_train, rmse_train, mae_train, r2_train, mse_test, rmse_test, mae_test, r2_test


# 计算残差函数
def calculate_residuals(model, X, y, dataset_name):
    print(f"开始计算{dataset_name}的残差...")
    y_pred = model.predict(X)
    residuals = y - y_pred
    residuals_df = pd.DataFrame({
        'True Value': y,
        'Predicted Value': y_pred,
        'Residual': residuals
    })
    residuals_df['Dataset'] = dataset_name
    print(f"{dataset_name}的残差计算完成。")
    return residuals_df


# 主函数
if __name__ == "__main__":
    file_path = r'C:\Users\mac\Desktop\浏河6.1.2.xls'
    data = read_and_preprocess_data(file_path)
    features = data.drop(['总磷'], axis=1)
    target = data['总磷']
    print("开始划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1234)
    print("训练集和测试集划分完成.")
    # 将X_train和y_train合并为一个DataFrame以便保存
    train_data = pd.concat([X_train, y_train], axis=1)
    # 类似地保存测试集
    test_data = pd.concat([X_test, y_test], axis=1)
    rf = RandomForestRegressor(n_jobs=-1, random_state=1234)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [50],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': [None],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
    }
    print("开始进行网格搜索...")
    grid_search = GridSearchCV(rf, param_grid, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error',
                                                        'neg_root_mean_squared_error', 'r2'],
                               refit='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("网格搜索完成.")
    print(grid_search.cv_results_['mean_test_r2'])
    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

# 绘制训练集的部分依赖图
feature = '浊度'
# 设置线的粗细和颜色
linewidth = 3
line_color = '#f47983'
# 设置坐标轴线粗细
axis_linewidth = 1.5
# 设置 x 轴和 y 轴标题字体大小
title_fontsize = 14
# 设置坐标轴刻度字体大小
tick_fontsize = 14

pdp = PartialDependenceDisplay.from_estimator(best_model, X_train, [feature], kind='average', percentiles=(0, 0.9977))
ax = pdp.axes_[0][0]
# 设置线的粗细和颜色
if len(ax.lines) > 0:
    line = ax.lines[0]
    line.set_linewidth(linewidth)
    line.set_color(line_color)
# 设置坐标轴线粗细
for spine in ax.spines.values():
    spine.set_linewidth(axis_linewidth)
# 设置 x 轴和 y 轴标题字体大小
ax.set_xlabel(ax.get_xlabel(), fontsize=title_fontsize)
ax.set_ylabel(ax.get_ylabel(), fontsize=title_fontsize)
# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title(f'Partial Dependence for {feature}', fontsize=16)

# 先保存为临时的无压缩 TIFF 文件
plt.savefig('temp.tif', dpi=1200, format='tif')

# 使用 PIL 打开并重新保存为压缩的 TIFF 文件
img = Image.open('temp.tif')
img.save(f'partial_dependence+{feature}_总_compressed2.tif', compression='tiff_lzw')

plt.show()