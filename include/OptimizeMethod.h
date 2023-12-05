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

#ifndef OPTIMIZEMETHOD_H
#define OPTIMIZEMETHOD_H

#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace std;

// 使用类封装所有函数，方便调用
class OptimizeMethod
{
    // 私有空间
    private:
        /// @brief 目标函数指针
        double (*target_function)(Eigen::VectorXd input) = nullptr;
        Eigen::VectorXd (*grad_target_function)(Eigen::VectorXd input) = nullptr;
        /// @brief 输出日志等级
        spdlog::level::level_enum log_level;
        /// @brief 是否输出子迭代信息
        bool show_sub_iter_info = false;
        /// @brief 是否执行线搜索
        bool line_search_mode = true;

    // 公开空间
    public:
        /// @brief 计算二阶数值梯度
        /// @param x 计算梯度的点
        /// @param step 差分步长
        /// @return 在指定点的梯度值
        Eigen::VectorXd calculate_grad(Eigen::VectorXd x, double step);

        /// @brief 初始化优化方法类，目前算法支持：单峰区间搜索、黄金分割区间搜索、精确线搜索（单峰区间+黄金分割）、Armijo 非精确线搜索、最速下降法、拟牛顿BFGS算法、拟牛顿DFP算法
        /// @param target 目标函数指针，无目标函数的梯度解析函数，计算过程将采用数值梯度
        /// @param log_lv 输出日志，支持 spdlog::debug / spdlog::level::info，默认为 spdlog::level::info
        /// @param show_sub_info 是否输出迭代过程中的子迭代过程（线搜索迭代过程）
        OptimizeMethod(double (*target)(Eigen::VectorXd input), spdlog::level::level_enum log_lv = spdlog::level::info, bool show_sub_info = false);

        /// @brief 初始化优化方法类，目前算法支持：单峰区间搜索、黄金分割区间搜索、精确线搜索（单峰区间+黄金分割）、Armijo 非精确线搜索、最速下降法、拟牛顿BFGS算法、拟牛顿DFP算法
        /// @param target 目标函数指针
        /// @param grad_target 目标梯度函数指针
        /// @param log_lv 输出日志，支持 spdlog::debug / spdlog::level::info，默认为 spdlog::level::info
        /// @param show_sub_info 是否输出迭代过程中的子迭代过程（线搜索迭代过程）
        OptimizeMethod(double (*target)(Eigen::VectorXd input), Eigen::VectorXd (*grad_target)(Eigen::VectorXd input), spdlog::level::level_enum log_lv = spdlog::level::info, bool show_sub_info = false);

        /// @brief 单峰区间搜索算法
        /// @param max_iteration 最大迭代步数
        /// @param step 搜索步长，需要大于0
        /// @param x_0 初始迭代点
        /// @param search_path 搜索方向
        /// @param alpha_0 初始长度
        /// @return 单峰区间[a, b]
        Eigen::VectorXd search_single_peak(int const max_iteration, double const step, Eigen::VectorXd x_0, Eigen::VectorXd search_path, double const alpha_0);

        /// @brief 黄金分割区间搜索算法
        /// @param max_iteration 最大迭代步数
        /// @param eps 终止迭代误差限
        /// @param x_0 初始迭代点
        /// @param search_path 搜索方向
        /// @param init_region 初始搜索区间
        /// @return 搜索方向上的极小值点对应的步长
        double search_single_peak_gold(int const max_iteration, double const eps, Eigen::VectorXd x_0, Eigen::VectorXd search_path, Eigen::VectorXd init_region);

        /// @brief 精确线搜索算法（单峰区间+黄金分割）
        /// @param max_iteration 最大迭代步数
        /// @param step 搜索步长，需要大于0
        /// @param eps 终止迭代误差限
        /// @param x_0 初始迭代点
        /// @param search_path 搜索方向
        /// @param alpha_0 初始长度，需要大于等于0
        /// @return 搜索方向上的极小值点对应的步长
        double line_search(int const max_iteration, double const step, double const eps, Eigen::VectorXd x_0, Eigen::VectorXd search_path, double const alpha_0);

        /// @brief 基于 Armijo 准则的非精确线搜索算法
        /// @param max_iteration 最大迭代步数
        /// @param step 搜索步长，需要大于0
        /// @param beta Armijo 准则参数，0 < beta < 1
        /// @param sigma Armijo 准则参数，0 < sigma < 0.5
        /// @param x_0 初始迭代点
        /// @param search_path 搜索方向
        /// @return 搜索方向上的极小值点对应的步长
        double armijo_search(int const max_iteration, double const step, double const beta, double const sigma, Eigen::VectorXd x_0, Eigen::VectorXd search_path);

        /// @brief 最速下降法
        /// @param max_iteration 最大迭代步数
        /// @param eps 终止迭代误差限
        /// @param step 搜索步长，需要大于0
        /// @param x_0 初始迭代点
        /// @return 极小值点
        Eigen::VectorXd gradient_descent(int const max_iteration, double const eps, double const step, Eigen::VectorXd x_0);

        /// @brief 拟牛顿 BFGS 算法
        /// @param max_iteration 最大迭代步数
        /// @param beta Armijo 准则参数，0 < beta < 1
        /// @param sigma Armijo 准则参数，0 < sigma < 0.5
        /// @param eps 终止迭代误差限
        /// @param step 搜索步长，需要大于0
        /// @param x_0 初始迭代点
        /// @return 极小值点
        Eigen::VectorXd Newton_BFGS(int const max_iteration, double const beta, double const sigma, double const eps, double const step, Eigen::VectorXd x_0);

        /// @brief 拟牛顿 DFP 算法
        /// @param max_iteration 最大迭代步数
        /// @param beta Armijo 准则参数，0 < beta < 1
        /// @param sigma Armijo 准则参数，0 < sigma < 0.5
        /// @param eps 终止迭代误差限
        /// @param step 搜索步长，需要大于0
        /// @param x_0 初始迭代点
        /// @return 极小值点
        Eigen::VectorXd Newton_DFP(int const max_iteration, double const beta, double const sigma, double const eps, double const step, Eigen::VectorXd x_0);
};

#endif