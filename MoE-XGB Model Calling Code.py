# -*- coding: utf-8 -*-

'''
MoE-XGB Model Calling Code

This code is a companion to the manuscript "X fri" submitted to [Journal Name].
It is used to call the MoE-XGB ground motion prediction model, which incorporates
station and seismic source latitude/longitude features.
If you encounter any issues, please leave a comment on the GitHub repository:
https://github.com/yourusername/MoE-XGB

Instructions:
- Modify the parameters between the "====" sections as indicated.
- The code between "****" sections is for calculations and should not be changed.
'''
# 导入 EarthquakeMoE 和 GatingNetwork 类
from MoExgb_pickle import EarthquakeMoE, GatingNetwork
from matplotlib import pyplot as plt
import pickle
import numpy as np
import os
import pandas as pd
import sys

# Set matplotlib font and size for consistent plotting
plt.rc('font', family='Times New Roman', size=20)

# 1. Set the model file path
# ===========================================================================
# Change the path to the MoE-XGB.pickle file downloaded from the repository.
# Example: r'C:\Users\YourName\Desktop\MoE-XGB.pickle'

modelFilePath = r'F:\戴云泽\SDEE论文代码\论文\C&G\github\MOE-xgb.pickle'
# ===========================================================================

# 2. Set the ground motion input parameters
# ===========================================================================
# Input parameters for the MoE-XGB model
Depth_km = 17  # Depth. (km): Depth of earthquake source
Mag = 4.6  # Mag.: Magnitude, in MJMA
Rhypo = 18.66  # Rhypo: Hypo-central distance, in km
Vs30 = 449  # Vs30: Site condition, in m/s
Station_Lat = 35.6895  # Station Lat.: Station latitude, in degrees
Station_Long = 139.6917  # Station Long.: Station longitude, in degrees
Long = 139.6910  # Long.: Seismic source longitude, in degrees
Lat = 35.6890  # Lat.: Seismic source latitude, in degrees
Station_Height_m = 79  # Station Height(m): Station altitude, in m
mech = 'R'  # mech: Focal mechanism, 'R' for Reverse, 'S' for Strike-slip, 'N' for Normal
# ===========================================================================

# Convert focal mechanism to numerical value
if mech == 'R':
    mech_value = 1
elif mech == 'S':
    mech_value = 2
elif mech == 'N':
    mech_value = 3

# Prepare input array for the model
# Feature order: Depth. (km), Mag., Rhypo, Vs30, Station Lat., Station Long., Long., Lat., Station Height(m), mech
X = [Depth_km, Mag, Rhypo, Vs30, Station_Lat, Station_Long, Long, Lat, Station_Height_m, mech_value]
# ===========================================================================

# 3. Set the path for the model prediction data output (TXT format)
# If empty, no TXT file will be saved. Output file name: SA-output.txt
# Example: r'C:\Users\YourName\Desktop'
txtFilePath = r'F:\戴云泽\SDEE论文代码\论文\C&G\github'

# 4. Set the path for the attenuation curve image (PNG format)
# If empty, no image will be saved. Output file name: SA-figure.png
# Example: r'C:\Users\YourName\Desktop'
curveFilePath = r'F:\戴云泽\SDEE论文代码\论文\C&G\github'
# ===========================================================================

# Define periods for output (PGA and spectral accelerations)
XName = ['SA0', 'SA0.01', 'SA0.02', 'SA0.03', 'SA0.04', 'SA0.05', 'SA0.06',
         'SA0.07', 'SA0.08', 'SA0.09', 'SA0.1', 'SA0.12', 'SA0.14',
         'SA0.15', 'SA0.16', 'SA0.18', 'SA0.2', 'SA0.25', 'SA0.3',
         'SA0.35', 'SA0.4', 'SA0.45', 'SA0.5', 'SA0.6', 'SA0.7',
         'SA0.8', 'SA0.9', 'SA1', 'SA1.5', 'SA2', 'SA2.5',
         'SA3', 'SA3.5', 'SA4', 'SA5']

# Convert XName to sa_list (numerical values)
sa_list = [0 if period == 'SA0' else float(period.replace('SA', '')) for period in XName]

# Model prediction and output calculations
# ****************************************************************************
# Load the MoE-XGB model
try:
    print(f"Loading model from: {modelFilePath}")
    with open(modelFilePath, 'rb') as file:
        model = pickle.load(file)
    print("模型加载成功")

    # 获取模型的特征列名
    if hasattr(model, 'feature_columns'):
        feature_columns = model.feature_columns
        print(f"模型特征列: {feature_columns}")
    else:
        # 如果模型没有保存特征列名，使用默认特征列名
        feature_columns = ['Depth. (km)', 'Mag.', 'Rhypo', 'Vs30', 'Station Lat.',
                           'Station Long.', 'Long.', 'Lat.', 'Station Height(m)', 'mech']
        print(f"使用默认特征列: {feature_columns}")

    # 创建包含输入数据的DataFrame
    input_df = pd.DataFrame([X], columns=feature_columns)
    print("输入数据:")
    print(input_df)

except Exception as e:
    print(f"加载模型时发生错误: {e}")
    print("请检查模型文件路径是否正确，文件是否完整")
    sys.exit(1)

# Predict using the input parameters for each SA period
# 预测部分代码
modelPre = []
try:
    for period in XName:  # 直接使用周期字符串
        print(f"Predicting for period: {period}")
        prediction = model.predict(input_df, period)  # 传入周期字符串
        modelPre.append(prediction[0])
    print("所有周期预测完成")

except Exception as e:
    print(f"预测过程中发生错误: {e}")
    print("请检查输入数据格式是否正确")
    sys.exit(1)

# Convert to numpy array for consistency
modelPre = np.array(modelPre)

# Convert logarithmic predictions to gal
modelPreExp = np.exp(modelPre)

# Print predictions to console
print("\n{:^10} {:^15}".format("T(s)", "gal"))
print("=" * 25)
for period, SA in zip(XName, modelPreExp):
    if period == 'SA0':
        print("{:^10} {:^15.3f}".format('PGA', SA))
    else:
        print("{:^10} {:^15.3f}".format(period.replace('SA', ''), SA))
# ****************************************************************************

# Save predictions to TXT file (if specified)
# ****************************************************************************
if txtFilePath and os.path.exists(txtFilePath):
    output_file = os.path.join(txtFilePath, 'SA-output.txt')
    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(
                'Depth. (km)    Mag.    Rhypo    Vs30    Station Lat.    Station Long.    Long.    Lat.    Station Height(m)    mech\n')
            output.write(
                '{:.3f}         {:.2f}     {:.3f}    {:.3f}    {:.4f}         {:.4f}          {:.4f}     {:.4f}     {:.3f}            {}\n\n'. \
                format(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], mech))
            output.write("{:^10} {:^15}\n".format("T(s)", "gal"))
            output.write('{}\n'.format("=" * 25))
            for period, SA in zip(XName, modelPreExp):
                if period == 'SA0':
                    output.write("{:^10} {:^15.3f}\n".format('PGA', SA))
                else:
                    output.write("{:^10} {:^15.3f}\n".format(period.replace('SA', ''), SA))
        print(f'输出文件已保存至 {output_file}')
    except Exception as e:
        print(f"保存TXT文件时出错: {e}")
elif txtFilePath:
    print(f'\033[1;31;40m提示: 路径 {txtFilePath} 在您的计算机上不存在。\033[0m')
    print("如果您不需要输出文件，请忽略此消息。")
# ****************************************************************************

# Plot and save attenuation curve (if specified)
# ****************************************************************************
SAListNum = [float(I.replace('SA', '')) for I in XName if I != 'SA0']

try:
    plt.figure(figsize=(10, 8), dpi=100)
    plt.plot(SAListNum, modelPreExp[1:], color='blue', marker='*', linestyle='-', linewidth=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.yticks([0.01, 0.1, 1, 10, 100, 1000])
    plt.tick_params(which='major', length=5, width=2)
    plt.tick_params(which='minor', length=5, width=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Period, T (s)', fontsize=14)
    plt.ylabel('Spectral Acceleration, SA (gal)', fontsize=20)
    plt.title('Attenuation Curve', fontsize=16)

    # 添加参数信息
    param_text = (f'Depth = {X[0]} km\nMagnitude = {X[1]}\n'
                  f'Rhypo = {X[2]} km\nVs30 = {X[3]} m/s\n'
                  f'Station: Lat = {X[4]}°, Lon = {X[5]}°\n'
                  f'Source: Lat = {X[7]}°, Lon = {X[6]}°\n'
                  f'Height = {X[8]} m\nMechanism = {mech}')
    plt.annotate(param_text, xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=10)

    # 添加版权信息
    plt.figtext(0.5, 0.01, '© MoE-XGB Model Prediction',
                ha='center', fontsize=10, color='gray')

    # 保存图像
    if curveFilePath and os.path.exists(curveFilePath):
        output_image = os.path.join(curveFilePath, 'SA-figure.png')
        plt.savefig(output_image, bbox_inches='tight')
        print(f'衰减曲线已保存至 {output_image}')
    elif curveFilePath:
        print(f'\033[1;31;40m提示: 路径 {curveFilePath} 在您的计算机上不存在。\033[0m')
        print("如果您不需要输出图像，请忽略此消息。")

    # 显示图像
    plt.show()

except Exception as e:
    print(f"绘制衰减曲线时出错: {e}")
# ****************************************************************************