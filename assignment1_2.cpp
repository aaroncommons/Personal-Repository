#include <iostream>
#include <vector>
#include <cmath>
using std::cout;
using std::vector;

double f1(const vector<double>& y, double x)
{
    return y[1];
}

double f2(const vector<double>& y, double x)
{
    return y[2];
}

double f3(const vector<double>& y, double x) // d^3y/dx^3 = f(y, dy/dx, x)
{
    return -5*y[1] - 0.3*x*y[0]*y[0]*y[0]; 
}

vector<double> vector_multiply(double c, vector<double> k)
{
    for (int i=0; i<3; i++)
    {
        k[i] = c*k[i];
    }
    return k;
}

vector<double> vector_add(vector<double> y, const vector<double>& k)
{
    for (int i=0; i<3; i++)
    {
        y[i] += k[i];
    }
    return y;
}

int main()
{
    double I = 30.0; //interval size
    int n = 3000; //number of steps 
    double h = I/double(n); //step size

    vector<double> y = {1.0, 0.0, 0.0}; //initial data
    double x = 0.0;
    vector<double> k1(3), k2(3), k3(3), k4(3);
    vector<double> y_intermediate(3);

    for (int i=0; i<n; i++)
    {
        k1 = {h*f1(y,x), h*f2(y,x), h*f3(y,x)};
       
        y_intermediate = vector_add(y, vector_multiply(0.5, k1));
        k2 = {h*f1(y_intermediate,x+h/2.0), h*f2(y_intermediate,x+h/2.0), h*f3(y_intermediate,x+h/2.0)};

        y_intermediate = vector_add(y, vector_multiply(0.5, k2));
        k3 = {h*f1(y_intermediate,x+h/2.0), h*f2(y_intermediate,x+h/2.0), h*f3(y_intermediate,x+h/2.0)};

        y_intermediate = vector_add(y,k3);
        k4 = {h*f1(y_intermediate,x+h), h*f2(y_intermediate,x+h), h*f3(y_intermediate,x+h)};

        for (int j=0; j<3; j++)
        {
            y[j] += (1.0/6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
        }
        x += h;

        // cout << x << " " << y[0] << "\n";  //generates output data for plotting
    }

    cout << "numerical approximation of y(x=30) = " << y[0] << "\n";
    
    return 0;
}