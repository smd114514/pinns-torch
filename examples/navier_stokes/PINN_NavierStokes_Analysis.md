# PINN-Torch: Navier-Stokes 方程代码分析

> 生成日期: 2026-07-21
> 项目: [pinns-torch](https://github.com/rezaakb/pinns-torch)

---

## 1. 相关文件位置

| 文件 | 路径 | 作用 |
|------|------|------|
| **训练主文件** | `examples/navier_stokes/train.py` | 主要 PDE 实现和训练入口 |
| **配置文件** | `examples/navier_stokes/configs/config.yaml` | 网络结构、超参数等配置 |
| **README** | `examples/navier_stokes/README.md` | 问题描述和方程定义 |
| **数据加载** | `train.py` 中 `read_data_fn` | 读取圆柱绕流数据 |
| **微分方程** | `train.py` 中 `pde_fn` | 核心微分方程实现 |
| **输出转换** | `train.py` 中 `output_fn` | 网络输出到物理量的转换 |
| **梯度计算工具** | `pinnstorch/utils/gradient_fn.py` | 自动微分工具函数 |
| **PINN 模块** | `pinnstorch/models/pinn_module.py` | LightningModule 基类 |
| **训练入口** | `pinnstorch/train.py` | 训练流程编排 |
| **绘图工具** | `pinnstorch/utils/plotting.py` | Navier-Stokes 结果可视化 |

---

## 2. 微分方程代码实现

### 2.1 网络输出 → 物理量转换 (`output_fn`)

```python
def output_fn(outputs, x, y, t):
    """将网络输出 [ψ, p] 转换为物理量 [u, v, p]"""

    outputs["u"] = pinnstorch.utils.gradient(outputs["psi"], y)[0]   # u = ∂ψ/∂y
    outputs["v"] = -pinnstorch.utils.gradient(outputs["psi"], x)[0]  # v = -∂ψ/∂x

    return outputs
```

**说明**: 网络输出流函数 $\psi$ 和压力 $p$，通过自动微分计算速度场 $u, v$，满足不可压缩条件 $\nabla \cdot \vec{u} = 0$。

### 2.2 Navier-Stokes 方程 (PDE) 实现 (`pde_fn`)

```python
def pde_fn(outputs, x, y, t, extra_variables):
    """定义偏微分方程 (PDEs)"""

    # === u 速度分量的一阶和二阶偏导数 ===
    u_x, u_y, u_t = pinnstorch.utils.gradient(outputs["u"], [x, y, t])
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    u_yy = pinnstorch.utils.gradient(u_y, y)[0]

    # === v 速度分量的一阶和二阶偏导数 ===
    v_x, v_y, v_t = pinnstorch.utils.gradient(outputs["v"], [x, y, t])
    v_xx = pinnstorch.utils.gradient(v_x, x)[0]
    v_yy = pinnstorch.utils.gradient(v_y, y)[0]

    # === 压力 p 的一阶偏导数 ===
    p_x, p_y = pinnstorch.utils.gradient(outputs["p"], [x, y])

    # === u 动量方程 (NS 第一个方程) ===
    outputs["f_u"] = (
        u_t                              # 非稳态项 ∂u/∂t
        + extra_variables["l1"] * (
            outputs["u"] * u_x + outputs["v"] * u_y
        )                                 # 对流项 λ₁(u∂u/∂x + v∂u/∂y)
        + p_x                            # 压力梯度项 ∂p/∂x
        - extra_variables["l2"] * (u_xx + u_yy)  # 扩散项 λ₂(∂²u/∂x² + ∂²u/∂y²)
    )

    # === v 动量方程 (NS 第二个方程) ===
    outputs["f_v"] = (
        v_t                              # 非稳态项 ∂v/∂t
        + extra_variables["l1"] * (
            outputs["u"] * v_x + outputs["v"] * v_y
        )                                 # 对流项 λ₁(u∂v/∂x + v∂v/∂y)
        + p_y                            # 压力梯度项 ∂p/∂y
        - extra_variables["l2"] * (v_xx + v_yy)  # 扩散项 λ₂(∂²v/∂x² + ∂²v/∂y²)
    )

    return outputs
```

---

## 3. 数学方程

### 逆向 Navier-Stokes 方程

**u 动量方程:**
$$
f_u = \frac{\partial u}{\partial t} + \lambda_1 \left( u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} \right) + \frac{\partial p}{\partial x} - \lambda_2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
$$

**v 动量方程:**
$$
f_v = \frac{\partial v}{\partial t} + \lambda_1 \left( u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} \right) + \frac{\partial p}{\partial y} - \lambda_2 \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
$$

### 变量说明

| 符号 | 含义 | 类型 |
|------|------|------|
| $u$ | x 方向速度分量 | 网络输出 → 推导 |
| $v$ | y 方向速度分量 | 网络输出 → 推导 |
| $p$ | 压力 | 网络直接输出 |
| $\psi$ | 流函数 (Stream Function) | 网络直接输出 |
| $\lambda_1$ | 未知参数 1 (逆问题) | `extra_variables["l1"]` |
| $\lambda_2$ | 未知参数 2 (逆问题) | `extra_variables["l2"]` |

### 连续性假设

由于使用流函数 $\psi$ 表示速度场，自动满足不可压缩条件:
$$
u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x} \quad \Rightarrow \quad \nabla \cdot \vec{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

---

## 4. 网络结构

### 神经网络架构

```yaml
net:
  _target_: pinnstorch.models.FCN
  layers: [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]  # 9层 每层20神经元
  output_names: [psi, p]
```

| 属性 | 值 |
|------|------|
| **输入层** | 3 个神经元 $(x, y, t)$ |
| **隐藏层** | 8 层 × 20 神经元 |
| **输出层** | 2 个神经元 $(\psi, p)$ |
| **激活函数** | 默认 (FCN 实现) |

### 参数配置

```yaml
model:
  extra_variables:
    l1: 0.0   # λ₁ 初始值 (训练中学习)
    l2: 0.0   # λ₂ 初始值 (训练中学习)

trainer:
  accelerator: gpu
  max_epochs: 250000
  check_val_every_n_epoch: 250001  # 仅训练不验证
```

---

## 5. 数据流

### 5.1 数据加载

```python
def read_data_fn(root_path):
    data = pinnstorch.utils.load_data(root_path, "cylinder_nektar_wake.mat")
    x = data["X_star"][:, 0:1]    # N x 1  空间坐标 x
    y = data["X_star"][:, 1:2]    # N x 1  空间坐标 y
    t = data["t"]                 # T x 1  时间点
    U_star = data["U_star"]       # N x 2 x T  速度场 (u, v)
    exact_u = U_star[:, 0, :]     # N x T  u速度
    exact_v = U_star[:, 1, :]     # N x T  v速度
    exact_p = data["p_star"]      # N x T  压力
    return pinnstorch.data.PointCloudData(
        spatial=[x, y], time=[t],
        solution={"u": exact_u, "v": exact_v, "p": exact_p}
    )
```

### 5.2 数据采样

```yaml
train_datasets:
  - mesh_sampler:
      num_sample: 5000           # 采样点数
      solution: [u, v]           # 需要监督的解变量
      collection_points: [f_u, f_v]  # PDE采集点 (残差计算)
```

### 5.3 训练流程

```
网络输入 (x, y, t)
    │
    ▼
FCN 前向传播 → [ψ, p]
    │
    ▼
output_fn → [u, v, p]  (从ψ导出的物理量)
    │
    ▼
pde_fn → [f_u, f_v]    (PDE残差)
    │
    ▼
损失计算: SSE_loss = SSE(u_true, u_pred) + SSE(v_true, v_pred) + SSE(f_u, 0) + SSE(f_v, 0)
    │
    ▼
反向传播 → 更新网络权重 + λ₁, λ₂
```

---

## 6. 梯度计算工具

`pinnstorch/utils/gradient_fn.py` 提供自动微分工具:

```python
# 前向模式梯度
def fwd_gradient(dy, dx, create_graph=True) -> List[torch.Tensor]

# 反向模式梯度 (核心)
def gradient(dy, dx, ones_like_tensor=None, create_graph=True) -> List[torch.Tensor]
```

实现基于 `torch.autograd.grad`，支持高阶导数的计算图构建。

---

## 7. 训练入口

```python
@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    pinnstorch.utils.extras(cfg)
    metric_dict, _ = pinnstorch.train(
        cfg,
        read_data_fn=read_data_fn,   # 数据读取函数
        pde_fn=pde_fn,               # PDE 方程函数
        output_fn=output_fn          # 输出转换函数
    )
    return pinnstorch.utils.get_metric_value(
        metric_dict, cfg.get("optimized_metric")
    )
```

### 运行命令

```bash
# 从项目根目录运行
python examples/navier_stokes/train.py

# 覆盖参数
python examples/navier_stokes/train.py trainer.max_epochs=20 n_train=3000
```

---

## 8. 问题类型

| 属性 | 说明 |
|------|------|
| **PDE 类型** | 连续逆问题 (Continuous Inverse) |
| **求解策略** | 流函数法，自动满足不可压缩约束 |
| **未知参数** | $\lambda_1, \lambda_2$ (扩散系数) |
| **损失函数** | SSE (Sum of Squared Errors) |
| **离散方法** | 基于点的无网格法 (PointCloud) |
| **边界条件** | 通过数据点隐式满足 |

---

## 9. 可视化

配置使用专用绘图函数:

```yaml
plotting:
  _target_: pinnstorch.utils.plot_navier_stokes
```

函数位于 `pinnstorch/utils/plotting.py`，用于绘制速度场和压力场的预测结果与真实解的对比。
