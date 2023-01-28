### Solving boundary value ODE using the shooting method and bisection method

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
using std::cout;
using std::vector;

double f1(const vector<double>& x, double t)
{
    return x[1];
}

double f2(const vector<double>& x, double t)
{
    return (-30.0*x[0])/(2.0+t*t*x[0]*x[0]);
}

vector<double> vector_multiply(double c, vector<double> k)
{
    for (int i=0; i<2; i++)
    {
        k[i] = c*k[i];
    }
    return k;
}

vector<double> vector_add(vector<double> x, const vector<double>& k)
{
    for (int i=0; i<2; i++)
    {
        x[i] += k[i];
    }
    return x;
}

double RungeKutta(double c)  // Evaluates x(10) for varying input c = dx/dt|t=0
{
    double I = 10.0; //interval size
    int n = 1000; //number of steps 
    double h = I/double(n); //step size

    vector<double> x = {3.0/4.0, c}; //initial data
    double t = 0.0;
    vector<double> k1(2), k2(2), k3(2), k4(2), x_intermediate(2);

    for (int i=0; i<n; i++)
    {
        k1 = {h*f1(x,t), h*f2(x,t)};
       
        x_intermediate = vector_add(x, vector_multiply(0.5, k1));
        k2 = {h*f1(x_intermediate,t+h/2.0), h*f2(x_intermediate,t+h/2.0)};

        x_intermediate = vector_add(x, vector_multiply(0.5, k2));
        k3 = {h*f1(x_intermediate,t+h/2.0), h*f2(x_intermediate,t+h/2.0)};

        x_intermediate = vector_add(x,k3);
        k4 = {h*f1(x_intermediate,t+h), h*f2(x_intermediate,t+h)};

        x[0] += (1.0/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        x[1] += (1.0/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
        t += h;
    }

    return x[0];
}

double RungeKuttaGraph(double c)  //Generates output data for plotting
{
    double I = 10.0; int n = 1000; double h = I/double(n); 
    vector<double> x = {3.0/4.0, c}; 
    double t = 0.0;
    vector<double> k1(2), k2(2), k3(2), k4(2), x_intermediate(2);

    for (int i=0; i<n; i++)
    {
        k1 = {h*f1(x,t), h*f2(x,t)};
        x_intermediate = vector_add(x, vector_multiply(0.5, k1));
        k2 = {h*f1(x_intermediate,t+h/2.0), h*f2(x_intermediate,t+h/2.0)};
        x_intermediate = vector_add(x, vector_multiply(0.5, k2));
        k3 = {h*f1(x_intermediate,t+h/2.0), h*f2(x_intermediate,t+h/2.0)};
        x_intermediate = vector_add(x,k3);
        k4 = {h*f1(x_intermediate,t+h), h*f2(x_intermediate,t+h)};

        x[0] += (1.0/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        x[1] += (1.0/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
        t += h;
        cout << t << " " << x[0] << "\n";  //generates output data for plotting
    }
    return x[0];
}

double b(double c)
{
    double x1 = -1.0; //boundary point
    double xc = RungeKutta(c); //x(10) evaluated for initial slope c
    return xc - x1;
}

double bisect(double c_lo, double c_hi) // bisection method to find c for b(c) = 0
{
    double b_lo = b(c_lo);
    double b_hi = b(c_hi);
    double c_mid;

    do
    {
        c_mid = 0.5 * (c_hi+c_lo);
        double b_mid = b(c_mid);
        if (b_mid*b_lo > 0.0) 
        {
            c_lo = c_mid; b_lo = b_mid;
        }
        else
        {
            c_hi = c_mid; b_hi = b_mid;
        }
        // cout << c_lo << " - " << c_mid << " - " << c_hi << "\n";
    } while (c_hi-c_lo > 1.0e-10);

    return 0.5 * (c_hi+c_lo);
}

int main()
{
    double I_c = 50; //interval for c
    int n_c = 2000; //number of steps taken for c
    double h_c = I_c/double(n_c); //step size for c
    double c = -25.0; //initial c

    for (int i=0; i<n_c; i++)
    {
        double b_c = b(c);
        // cout << c << " " << b_c << "\n"; 
        c += h_c;
    }

    vector<double> c_vector = {bisect(-20.0,-16.0), bisect(-7.0,-5.5), bisect(-4.0,-3.0), bisect(2.0,3.0), 
    bisect(4.0,5.0), bisect(7.0,8.0), bisect(20.0,22.0)};

    cout << std::setprecision(9);
    cout << "c1=" << c_vector[0] << " (test: RungeKutta(c1) = " << RungeKutta(c_vector[0]) << ")\n";
    cout << "c2=" << c_vector[1] << " (test: RungeKutta(c2) = " << RungeKutta(c_vector[1]) << ")\n";
    cout << "c3=" << c_vector[2] << " (test: RungeKutta(c3) = " << RungeKutta(c_vector[2]) << ")\n";
    cout << "c4=" << c_vector[3] << " (test: RungeKutta(c4) = " << RungeKutta(c_vector[3]) << ")\n";
    cout << "c5=" << c_vector[4] << " (test: RungeKutta(c5) = " << RungeKutta(c_vector[4]) << ")\n";
    cout << "c6=" << c_vector[5] << " (test: RungeKutta(c6) = " << RungeKutta(c_vector[5]) << ")\n";
    cout << "c7=" << c_vector[6] << " (test: RungeKutta(c7) = " << RungeKutta(c_vector[6]) << ")\n";

    // Produces data for plotting graphs
    //double x1 = RungeKuttaGraph(c_vector[0]);
    //double x2 = RungeKuttaGraph(c_vector[1]);
    //double x3 = RungeKuttaGraph(c_vector[2]);
    //double x4 = RungeKuttaGraph(c_vector[3]);
    //double x5 = RungeKuttaGraph(c_vector[4]);
    //double x6 = RungeKuttaGraph(c_vector[5]);
    //double x7 = RungeKuttaGraph(c_vector[6]);

    return 0;
}
