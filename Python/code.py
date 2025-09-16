import numpy as np  # NumPy 提供矩阵运算、线性代数运算和数组结构，导入 numpy 库并命名为 np，用于矩阵和数值计算

class KalmanFilter(object):  # 定义一个卡尔曼滤波器类，继承自 object
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        # 构造函数，接受系统矩阵 F、控制矩阵 B、观测矩阵 H、过程噪声协方差 Q、
        # 观测噪声协方差 R、初始协方差 P 和初始状态 x0（均可选）

        if(F is None or H is None):  # 如果没有提供系统动力学矩阵 F 或观测矩阵 H，则不能初始化
            raise ValueError("Set proper system dynamics.")  # 抛出异常，提示需要这些矩阵

        self.n = F.shape[1]  # 状态向量的维度 n（使用 F 的列数），注意这里假设 F 是方阵或合适维度
        self.m = H.shape[1]  # 观测矩阵 H 的列数，这里赋值为 H 的列数（原代码使用的是 H.shape[1]）

        self.F = F  # 状态转移矩阵 F
        self.H = H  # 观测矩阵 H
        self.B = 0 if B is None else B  # 如果未提供控制矩阵 B，则默认为 0（无控制输入），否则使用提供的 B
        self.Q = np.eye(self.n) if Q is None else Q  # 过程噪声协方差 Q，默认使用 n×n 单位矩阵
        self.R = np.eye(self.n) if R is None else R  # 观测噪声协方差 R，默认使用 n×n 单位矩阵（注意：通常 R 的维度应为观测维度）
        self.P = np.eye(self.n) if P is None else P  # 状态估计协方差 P，默认 n×n 单位矩阵
        self.x = np.zeros((self.n, 1)) if x0 is None else x0  # 初始状态向量 x，默认全 0 列向量（n×1）

    def predict(self, u = 0):  # 预测步骤，接受可选的控制向量 u（默认 0）
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)  # 根据线性模型预测下一个状态 x = F x + B u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # 预测协方差 P = F P F^T + Q
        return self.x  # 返回预测的状态向量

    def update(self, z):  # 更新步骤，接受观测值 z
        y = z - np.dot(self.H, self.x)  # 计算创新（残差）y = z - H x
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))  # 计算创新协方差 S = R + H P H^T
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 计算卡尔曼增益 K = P H^T S^{-1}
        self.x = self.x + np.dot(K, y)  # 更新状态估计 x = x + K y
        I = np.eye(self.n)  # 单位矩阵 I（n×n）
        # 更新协方差 P，使用数值稳定的 Joseph 形式：P = (I - K H) P (I - K H)^T + K R K^T
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():  # 一个示例函数，演示如何使用 KalmanFilter 类
	dt = 1.0/60  # 时间步长 dt，假设采样率为 60 Hz
	F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])  # 状态转移矩阵 F（3 状态：位置、速度、加速度的简单离散模型）
	H = np.array([1, 0, 0]).reshape(1, 3)  # 观测矩阵 H，只观测第一个状态（位置），因此是 1×3 矩阵
	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])  # 过程噪声协方差 Q（示例值）
	R = np.array([0.5]).reshape(1, 1)  # 观测噪声协方差 R（1×1，因为只有一个观测）

	x = np.linspace(-10, 10, 100)  # 生成 100 个在 -10 到 10 间均匀分布的点，作为自变量
	measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)  # 根据一个二次函数生成观测，并加入高斯噪声（均值 0，标准差 2）

	kf = KalmanFilter(F = F, H = H, Q = Q, R = R)  # 创建卡尔曼滤波器实例，传入 F、H、Q、R
	predictions = []  # 用于保存每步的预测结果

	for z in measurements:  # 对每一个观测值进行预测和更新循环
		predictions.append(np.dot(H,  kf.predict())[0])  # 先调用 predict()，将预测的观测 H x 存入 predictions（取标量值）
		kf.update(z)  # 然后用实际观测 z 更新滤波器状态

	import matplotlib.pyplot as plt  # 导入绘图库 matplotlib 的 pyplot 模块用于绘图
	plt.plot(range(len(measurements)), measurements, label = 'Measurements')  # 绘制观测值曲线
	plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')  # 绘制卡尔曼滤波器的预测曲线
	plt.legend()  # 显示图例
	plt.show()  # 显示图形窗口

if __name__ == '__main__':  # 当该脚本作为主程序运行时，执行 example() 函数
    example()  # 运行示例
