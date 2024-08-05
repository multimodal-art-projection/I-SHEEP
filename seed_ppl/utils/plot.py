import matplotlib.pyplot as plt

# 给定的数据
data = [{'Name': 'Alice', 'Age': 25, 'City': 'New York'},
        {'Name': 'Bob', 'Age': 30, 'City': 'Los Angeles'},
        {'Name': 'Charlie', 'Age': 35, 'City': 'Chicago'}]

# 从字典列表中提取年龄数据
ages = [person['Age'] for person in data]

# 使用Matplotlib绘制直方图
plt.figure(figsize=(8, 4))  # 设置图形大小
plt.hist(ages, bins=5, color='blue', alpha=0.7)  # bins参数定义直方图的区间数
plt.title('Age Distribution')  # 添加标题
plt.xlabel('Age')  # X轴标签
plt.ylabel('Frequency')  # Y轴标签
plt.grid(True)  # 显示网格
plt.show()  # 显示图形
