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
    opti.Newton_BFGS(100, 0.5, 0.2, 1e-6, 0.0001, x0);
    opti.Newton_DFP(100, 0.5, 0.2, 1e-6, 0.0001, x0);
    #if defined(_WIN32) || defined(_WIN64)
    system("pause");
    #endif
}
