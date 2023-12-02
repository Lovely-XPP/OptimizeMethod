// Copyright (c) 2023 易鹏 中山大学航空航天学院
// Copyright (c) 2023 Peng Yi, Sun Yat-Sen University, School of Aeronautics and Astronautics
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <OptimizeMethod.h>

Eigen::VectorXd OptimizeMethod::calculate_grad(Eigen::VectorXd x, double step)
{
    // 如果目标函数梯度有解析解输入，则采用解析解
    if (grad_target_function != nullptr)
    {
        return grad_target_function(x);
    }
    // 如果目标函数梯度没有解析解，则使用数值梯度进行近似
    int dim = x.size();
    Eigen::VectorXd f_gradient = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd x_gradient = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(dim);
    // 求数值梯度
    for (size_t i = 0; i < dim; i++)
    {
        x_gradient = x;
        x_gradient[i] = x[i] - step;
        f_gradient[0] = target_function(x_gradient);
        x_gradient[i] = x[i] + step;
        f_gradient[1] = target_function(x_gradient);
        grad[i] = (f_gradient[1] - f_gradient[0]) / (2 * step);
    }
    return grad;
}

OptimizeMethod::OptimizeMethod(double (*target)(Eigen::VectorXd input), spdlog::level::level_enum log_lv, bool show_sub_info)
{
    // 赋值到类变量
    show_sub_iter_info = show_sub_info;
    target_function = target;
    // 设置输出格式
    log_level = log_lv;
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%l] %v %$");
    spdlog::set_level(log_level);
}

OptimizeMethod::OptimizeMethod(double (*target)(Eigen::VectorXd input), Eigen::VectorXd (*grad_target)(Eigen::VectorXd input), spdlog::level::level_enum log_lv, bool show_sub_info)
{
    // 赋值到类变量
    show_sub_iter_info = show_sub_info;
    target_function = target;
    grad_target_function = grad_target;
    // 设置输出格式
    log_level = log_lv;
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%l] %v %$");
    spdlog::set_level(log_level);
}

Eigen::VectorXd OptimizeMethod::search_single_peak(int const max_iteration, double const step, Eigen::VectorXd x_0, Eigen::VectorXd search_path, double const alpha_0)
{
    // 如果输入的初始条件与搜索路径维度不同，则报错
    if (x_0.size() != search_path.size())
    {
        spdlog::error("[search_single_peak] The size of Initial Condition is not match to Search Path.");
        exit(0);
    }
    // 判断步长条件
    if (step <= 0)
    {
        spdlog::error("[search_single_peak] Step must be > 0.");
        exit(0);
    }
    // 判断alpha_0条件
    if (alpha_0 < 0)
    {
        spdlog::error("[search_single_peak] alpha_0 must be >= 0.");
        exit(0);
    }
    // 判断最大迭代条件
    if (max_iteration <= 0)
    {
        spdlog::error("[search_single_peak] max_iteration must be > 0.");
        exit(0);
    }
    // 输出设置信息
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::debug("[search_single_peak] [Setting]  Max Iter: {0:d}  Step: {1:.3e}  Alpha_0: {2:.4f}", max_iteration, step, alpha_0);
    }

    // 初始化参数
    int k = 0;
    double h = step;
    double alpha;
    double alpha_k = alpha_0;
    double alpha_k_;
    double phi;
    double phi_;
    Eigen::VectorXd x = x_0;

    // 第 1 步：计算phi_0
    x = x + alpha_k * search_path;
    phi = target_function(x);

    // 第 2 步
    for (; k <= max_iteration; k++)
    {
        alpha_k_ = alpha_k + h;
        x = x_0 + alpha_k_ * search_path;
        phi_ = target_function(x);
        if (line_search_mode || (!line_search_mode && show_sub_iter_info))
        {
            spdlog::debug("[search_single_peak] [iter {0}]  a(k): {1:.6f} f(k): {2:.6f} a(k+1): {3:.6f} phi(k+1): {4:.6f}", k, alpha_k, phi, alpha_k_, phi_);
        }
        // 第 3 步
        if (phi_ < phi)
        {
            h = 2 * h;
            alpha = alpha_k;
            alpha_k = alpha_k_;
            phi = phi_;
            continue;
        }
        else
        {
            if (k == 0)
            {
                alpha = alpha_k;
                alpha_k = alpha_k_;
                h = -h;
                continue;
            }
            else
            {
                break;
            }
        }
    }
    double a_min = min(alpha, alpha_k_);
    double a_max = max(alpha, alpha_k_);
    Eigen::VectorXd result(2);
    result << a_min, a_max;
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::info("[search_single_peak]  Iter: {0:d}  Result: [{1:.4f}, {2:.4f}]", k, a_min, a_max);
    }
    return result;
}

double OptimizeMethod::search_single_peak_gold(int const max_iteration, double const eps, Eigen::VectorXd x_0, Eigen::VectorXd search_path, Eigen::VectorXd init_region)
{
    // 如果输入的初始条件与搜索路径维度不同，则报错
    if (x_0.size() != search_path.size())
    {
        spdlog::error("[search_single_peak_gold] The size of Initial Condition is not match to Search Path.");
        exit(0);
    }
    // 判断误差限条件
    if (eps <= 0)
    {
        spdlog::error("[search_single_peak_gold] Error Limitation (eps) must be > 0.");
        exit(0);
    }
    // 判断最大迭代条件
    if (max_iteration <= 0)
    {
        spdlog::error("[search_single_peak_gold] max_iteration must be > 0.");
        exit(0);
    }
    // 确认初始区域从小到大排
    double a = init_region[0];
    double b = init_region[1];
    init_region[0] = min(a, b);
    init_region[1] = max(a, b);
    // 输出设置信息
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::debug("[search_single_peak_gold] [Setting]  Max Iter: {0:d}  Error Limitation: {1:.6f}  Initial Region: [{2:.4f} {3:.4f}]", max_iteration, eps, init_region[0], init_region[1]);
    }

    // 初始化
    a = init_region[0];
    b = init_region[1];
    double p = a + 0.382 * (b - a);
    double q = a + 0.618 * (b - a);
    double phi_p, phi_q;
    double eps_k = 0;
    int k = 0;
    Eigen::VectorXd x_p = x_0;
    Eigen::VectorXd x_q = x_0;
    double result = 0;
    x_p = x_0 + p * search_path;
    x_q = x_0 + q * search_path;
    phi_p = target_function(x_p);
    phi_q = target_function(x_q);

    for (; k <= max_iteration; k++)
    {
        if (phi_p <= phi_q)
        {
            eps_k = abs(q - a);
            if (line_search_mode || (!line_search_mode && show_sub_iter_info))
            {
                spdlog::debug("[search_single_peak_gold] [iter {0}]  Error: {1:.8f}", k, eps_k);
            }
            if (eps_k <= eps)
            {
                result = p;
                break;
            }
            a = a;
            b = q;
            phi_q = phi_p;
            q = p;
            p = a + 0.382 * (b - a);
            x_p = x_0 + p * search_path;
            phi_p = target_function(x_p);
        }
        else
        {
            eps_k = abs(b - q);
            if (line_search_mode || (!line_search_mode && show_sub_iter_info))
            {
                spdlog::debug("[search_single_peak_gold] [iter {0}]  Error: {1:.8f}", k, eps_k);
            }
            if (eps_k <= eps)
            {
                result = q;
                break;
            }
            a = p;
            b = b;
            phi_p = phi_q;
            p = q;
            q = a + 0.618 * (b - a);
            x_q = x_0 + q * search_path;
            phi_q = target_function(x_q);
        }
    }
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::info("[search_single_peak_gold]  Iter: {0:d}  Result: {1:.8f}", k, result);
    }
    return result;
}

double OptimizeMethod::line_search(int const max_iteration, double const step, double const eps, Eigen::VectorXd x_0, Eigen::VectorXd search_path, double const alpha_0)
{
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::debug("[line_search] [Setting]  Max Iter: {0:d}  Step: {1:.6f}  Error Limitation: {1:.6f}  Alpha_0: {3:.6f}", max_iteration, step, eps, alpha_0);
    }
    double result = search_single_peak_gold(max_iteration, eps, x_0, search_path, search_single_peak(max_iteration, step, x_0, search_path, alpha_0));
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::info("[line_search]  Result: {0:.8f}", result);
    }
    return result;
}

double OptimizeMethod::armijo_search(int const max_iteration, double const step, double const beta, double const sigma, Eigen::VectorXd x_0, Eigen::VectorXd search_path)
{
    // 如果输入的初始条件与搜索路径维度不同，则报错
    if (x_0.size() != search_path.size())
    {
        spdlog::error("[armijo_search] The size of Initial Condition is not match to Search Path.");
        exit(0);
    }
    // 判断步长条件
    if (step <= 0)
    {
        spdlog::error("[armijo_search] Step must be > 0.");
        exit(0);
    }
    // 判断最大迭代条件
    if (max_iteration <= 0)
    {
        spdlog::error("[armijo_search] max_iteration must be > 0.");
        exit(0);
    }
    // 判断beta条件
    if (beta <= 0 || beta >= 1)
    {
        spdlog::error("[armijo_search] beta must feet 0 < beta < 1.");
        exit(0);
    }
    // 判断sigma条件
    if (sigma <= 0 || sigma >= 1)
    {
        spdlog::error("[armijo_search] beta must feet 0 < sigma < 0.05.");
        exit(0);
    }

    // 初始化
    int k = 0;
    double err = 0;
    int dim = x_0.size();
    Eigen::VectorXd x = x_0;
    Eigen::VectorXd x_ = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dim);
    // 计算梯度
    gradient = calculate_grad(x, step);
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::debug("[armijo_search] [Setting]  Max Iter: {0:d}  Step: {1:.6f}  beta: {2:.6f}  sigma: {3:.6f}", max_iteration, step, beta, sigma);
    }

    for (; k <= max_iteration; k++)
    {
        // 计算新的函数值
        x_ = x + pow(beta, k) * search_path;
        err = target_function(x_) - target_function(x) - sigma * pow(beta, k) * gradient.transpose() * search_path;
        if (line_search_mode || (!line_search_mode && show_sub_iter_info))
        {
            spdlog::debug("[armijo_search] [Iter {0:d}]  Error: {1:.6f}", k, err);
        }

        // 满足条件则停止
        if (err <= 0)
        {
            x = x_;
            break;
        }
    }
    double result = pow(beta, k);
    if (line_search_mode || (!line_search_mode && show_sub_iter_info))
    {
        spdlog::info("[armijo_search]  Iter: {0:d}  Result: {1:.8f}", k, result);
    }
    return result;
}

Eigen::VectorXd OptimizeMethod::gradient_descent(int const max_iteration, double const eps, double const step, Eigen::VectorXd x_0)
{
    // 设置为非线索模式
    line_search_mode = false;
    // 判断误差限条件
    if (eps <= 0)
    {
        spdlog::error("[gradient_descent] Error Limitation (eps) must be > 0.");
        exit(0);
    }
    // 判断步长条件
    if (step <= 0)
    {
        spdlog::error("[gradient_descent] Step must be > 0.");
        exit(0);
    }
    // 判断最大迭代条件
    if (max_iteration <= 0)
    {
        spdlog::error("[gradient_descent] max_iteration must be > 0.");
        exit(0);
    }

    // 初始化
    int dim = x_0.size();
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd x = x_0;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(dim);
    double err = 0;
    double alpha = 0;
    int k = 0;
    spdlog::debug("[gradient_descent] [Setting]  Max Iter: {0:d}  Step: {1:.6f}  Error Limit: {2:.6f}", max_iteration, step, eps);
    spdlog::info("[gradient_descent]  ******************** Start ********************");

    for (; k <= max_iteration; k++)
    {
        // 求数值梯度及其范数
        gradient = calculate_grad(x, step);
        err = gradient.norm();
        spdlog::debug("[gradient_descent] [iter {0}]  Error: {1:.8f}", k, err);

        // 判断是否满足迭代停止条件
        if (err <= eps)
        {
            break;
        }

        // 线搜索找到合适的步长因子
        d = -gradient;
        alpha = line_search(100, 0.01, 1e-5, x, d, 0);

        // 更新参数
        x = x + alpha * d;
    }
    spdlog::info("[gradient_descent]  Iter: {0:d}  Result: [{1:.8f} {2:.8f}]  Function Value: {3:.8f}", k, x[0], x[1], target_function(x));
    spdlog::info("[gradient_descent]  ********************  End  ********************");
    // 重置为线索模式
    line_search_mode = true;
    return x;
}

Eigen::VectorXd OptimizeMethod::Newtow_BFGS(int const max_iteration, double const beta, double const sigma, double const eps, double const step, Eigen::VectorXd x_0)
{
    // 设置为非线索模式
    line_search_mode = false;
    // 判断误差限条件
    if (eps <= 0)
    {
        spdlog::error("[Newtow_BFGS] Error Limitation (eps) must be > 0.");
        exit(0);
    }
    // 判断步长条件
    if (step <= 0)
    {
        spdlog::error("[Newtow_BFGS] Step must be > 0.");
        exit(0);
    }
    // 判断最大迭代条件
    if (max_iteration <= 0)
    {
        spdlog::error("[Newtow_BFGS] max_iteration must be > 0.");
        exit(0);
    }
    // 判断beta条件
    if (beta <= 0 || beta >= 1)
    {
        spdlog::error("[Newtow_BFGS] beta must feet 0 < beta < 1.");
        exit(0);
    }
    // 判断sigma条件
    if (sigma <= 0 || sigma >= 1)
    {
        spdlog::error("[Newtow_BFGS] beta must feet 0 < sigma < 0.05.");
        exit(0);
    }

    // 初始化变量
    // 初始化矩阵Bk为单位阵
    int dim = x_0.size();
    Eigen::MatrixXd Bk = Eigen::VectorXd::Ones(dim).asDiagonal();
    Eigen::VectorXd gk = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd dk = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd xk = x_0;
    Eigen::VectorXd xk_ = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd sk = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd yk = Eigen::VectorXd::Zero(dim);
    int k = 0;
    double alpha = 0;
    double err = 0;
    spdlog::debug("[Newtow_BFGS] [Setting]  Max Iter: {0:d}  Step: {1:.6f}  beta: {2:.6f}  sigma: {3:.6f}  Error Limit: {4:.6f}", max_iteration, step, beta, sigma, eps);
    spdlog::info("[Newtow_BFGS]  ******************** Start ********************");

    // 开始循环算法
    for (; k <= max_iteration; k++)
    {
        // 求梯度
        gk = calculate_grad(xk, step);
        err = gk.norm();
        spdlog::debug("[Newtow_BFGS] [iter {0}]  Error: {1:.8f}", k, err);

        // 判断是否满足迭代停止条件
        if (err <= eps)
        {
            break;
        }

        // 求解线性方程组，由于Bk对称正定，采用LLT分解求解
        dk = -Bk.llt().solve(gk);

        // Armijo线搜索找到合适的步长因子
        alpha = armijo_search(max_iteration, step, beta, sigma, xk, dk);

        // 校正 Bk
        xk_ = xk + alpha * dk;
        sk = xk_ - xk;
        yk = calculate_grad(xk_, step) - gk;
        if (sk.transpose() * yk > 0)
        {
            Bk = Bk - Bk * sk * sk.transpose() * Bk / (sk.transpose() * Bk * sk) + yk * yk.transpose() / (yk.transpose() * sk);
        }

        // 更新参数
        xk = xk_;
    }
    spdlog::info("[Newtow_BFGS]  Iter: {0:d}  Result: [{1:.8f} {2:.8f}]  Function Value: {3:.8f}", k, xk[0], xk[1], target_function(xk));
    spdlog::info("[Newtow_BFGS]  ********************  End  ********************");
    // 重置为线索模式
    line_search_mode = true;
    return xk;
}

Eigen::VectorXd OptimizeMethod::Newtow_DFP(int const max_iteration, double const beta, double const sigma, double const eps, double const step, Eigen::VectorXd x_0)
{
    // 设置为非线索模式
    line_search_mode = false;
    // 判断误差限条件
    if (eps <= 0)
    {
        spdlog::error("[Newtow_DFP] Error Limitation (eps) must be > 0.");
        exit(0);
    }
    // 判断步长条件
    if (step <= 0)
    {
        spdlog::error("[Newtow_DFP] Step must be > 0.");
        exit(0);
    }
    // 判断最大迭代条件
    if (max_iteration <= 0)
    {
        spdlog::error("[Newtow_DFP] max_iteration must be > 0.");
        exit(0);
    }
    // 判断beta条件
    if (beta <= 0 || beta >= 1)
    {
        spdlog::error("[Newtow_DFP] beta must feet 0 < beta < 1.");
        exit(0);
    }
    // 判断sigma条件
    if (sigma <= 0 || sigma >= 1)
    {
        spdlog::error("[Newtow_DFP] beta must feet 0 < sigma < 0.05.");
        exit(0);
    }

    // 初始化变量
    int dim = x_0.size();
    // 初始化矩阵Hk为单位阵
    Eigen::MatrixXd Hk = Eigen::VectorXd::Ones(dim).asDiagonal();
    Eigen::VectorXd gk = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd dk = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd xk = x_0;
    Eigen::VectorXd xk_ = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd sk = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd yk = Eigen::VectorXd::Zero(dim);
    int k = 0;
    double alpha = 0;
    double err = 0;
    spdlog::debug("[Newtow_DFP] [Setting]  Max Iter: {0:d}  Step: {1:.6f}  beta: {2:.6f}  sigma: {3:.6f}  Error Limit: {4:.6f}", max_iteration, step, beta, sigma, eps);
    spdlog::info("[Newtow_DFP]  ******************** Start ********************");

    // 开始循环算法
    for (; k <= max_iteration; k++)
    {
        // 求梯度
        gk = calculate_grad(xk, step);
        err = gk.norm();
        spdlog::debug("[Newtow_DFP] [iter {0}]  Error: {1:.8f}", k, err);

        // 判断是否满足迭代停止条件
        if (err <= eps)
        {
            break;
        }

        // 计算下降方向
        dk = -Hk * gk;

        // Armijo线搜索找到合适的步长因子
        alpha = armijo_search(max_iteration, step, beta, sigma, xk, dk);

        // 校正 Hk
        xk_ = xk + alpha * dk;
        sk = xk_ - xk;
        yk = calculate_grad(xk_, step) - gk;
        if (sk.transpose() * yk > 0)
        {
            Hk = Hk - Hk * yk * yk.transpose() * Hk / (yk.transpose() * Hk * yk) + sk * sk.transpose() / (sk.transpose() * yk);
        }

        // 更新参数
        xk = xk_;
    }
    spdlog::info("[Newtow_DFP]  Iter: {0:d}  Result: [{1:.8f} {2:.8f}]  Function Value: {3:.8f}", k, xk[0], xk[1], target_function(xk));
    spdlog::info("[Newtow_DFP]  ********************  End  ********************");
    // 重置为线索模式
    line_search_mode = true;
    return xk;
}
