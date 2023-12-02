# OptimizeMethod

## 简介
传统常见最优化方法C++封装，包括单峰区间搜索、黄金分割区间搜索、精确线搜索（单峰区间+黄金分割）、Armijo 非精确线搜索、最速下降法、拟牛顿BFGS算法、拟牛顿DFP算法等

## 依赖库（已包含在3rdparty内）

`spdlog` - 轻量C++日志库

`Eigen` - 矩阵C++库

## 算例
### Assignment 1
使用最速下降法求$f(x_1, x_2) = 3 x_1^2 + 2s_2^2 - 4 x_1 - 6x_2$，要求线搜索方法使用单峰区间搜索方法 + 黄金分割法。

```C++
double target_function(Eigen::VectorXd x)
{
    double const x0 = x[0];
    double const y0 = x[1];
    return 3 * x0 * x0 + 2 * y0 * y0 - 4 * x0 - 6 * y0;
}

int main()
{
    Eigen::VectorXd x0(2);
    x0 << 0, 0;
    OptimizeMethod opti = OptimizeMethod(target_function, spdlog::level::debug);
    opti.gradient_descent(100, 1e-6, 0.01, x0);
    #if defined(_WIN32) || defined(_WIN64)
    system("pause");
    #endif
}
```

### Assignment 2
分别使用拟牛顿方法中BFGS方法与DFP方法，采用Armijo准则进行非精确线搜索方法，使用迭代初始点$(2,2)^T$，计算$f(x_1, x_2) = x_1^2 - x_1 x_2 + x_2^2 + 2x_1 - 4x_2$的最小值。

```C++
#include <OptimizeMethod.h>
#include <spdlog/spdlog.h>
#include <cstdlib>

double target_function(Eigen::VectorXd x)
{
    double const x1 = x[0];
    double const x2 = x[1];
    return x1 * x1 - x1 * x2 + x2 * x2 + 2 * x1 - 4 * x2;
}

Eigen::VectorXd grad_target_function(Eigen::VectorXd x)
{
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    double const x1 = x[0];
    double const x2 = x[1];
    grad[0] = 2 * x1 - x2 + 2;
    grad[1] = 2 * x2 - x1 - 4;
    return grad;
}

int main()
{
    Eigen::VectorXd x0(2);
    x0 << -2, 2;
    OptimizeMethod opti = OptimizeMethod(target_function, grad_target_function, spdlog::level::debug);
    opti.Newtow_BFGS(100, 0.5, 0.2, 1e-6, 0.0001, x0);
    opti.Newtow_DFP(100, 0.5, 0.2, 1e-6, 0.0001, x0);
    #if defined(_WIN32) || defined(_WIN64)
    system("pause");
    #endif
}

```