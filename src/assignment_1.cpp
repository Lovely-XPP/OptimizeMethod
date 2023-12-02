#include <OptimizeMethod.h>
#include <spdlog/spdlog.h>
#include <cstdlib>

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
