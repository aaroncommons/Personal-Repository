#include <iostream>
#include <cmath>
using std::cout;
#include "MAU34601_rng.h"
#include <vector>

template <class T> class Field
{
    private:
        int nx_;
        int ny_;
        std::vector<T> data_;
        int index_(int x, int y) const 
        {
            x = (x+nx_)%nx_;
            y = (y+nx_)%ny_;
            return x + nx_*y;
        }
    public:
        Field(int nx, int ny) : nx_(nx), ny_(ny), data_(nx_*ny_)
        {
            int n = nx_*ny_;
            for (int i=0; i<n; i++) data_[i] = 0.0;
        }

    T& operator() (int x, int y) { return data_[index_(x,y)]; }
    T operator() (int x, int y) const { return data_[index_(x,y)]; }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
};

Ran r(57927482);

int action(Field<int>& F, int F_ij, int i, int j)
{
    int S = 0;
    if (F_ij != F(i-1,j)) { S++; }
    if (F_ij != F(i+1,j)) { S++; }
    if (F_ij != F(i,j-1)) { S++; }
    if (F_ij != F(i,j+1)) { S++; }

    return S;
}

void MetropolisStep(Field<int>& F, double beta, int q)
{
    for (int i=0; i<F.nx(); i++){
        for (int j=0; j<F.ny(); j++)
        {
            int F_new = int(q*r) + 1;
            int S = action(F, F(i,j), i, j);
            int S_new = action(F, F_new, i, j);
            int dS = S_new - S;

            double prob = exp(-beta*dS);
            if (dS < 0) { F(i,j) = F_new; }
            else if (prob > r) { F(i,j) = F_new; }
            else { F(i,j) = F(i,j); }
        }
    }
}

int max(const std::vector<int>& arr, int q)
{
    int m = arr[0];
    for (int i=0; i<q; i++)
    {
        if (arr[i] > m) 
            m = arr[i];
    }
    return m;
}

double Magnetisation(const Field<int>& F, int q)
{
    // Finding f(sigma) - fraction of most occuring site
    std::vector<int> A(q);
    for (int k=0; k<q; k++)
    {
        int count = 0; // counts number of times certain value occurs on grid
        for (int i=0; i<F.nx(); i++){
            for (int j=0; j<F.ny(); j++){
                if (F(i,j) == k+1) { count++; }
            }
        }
        A[k] = count;
    }
    double f = double(max(A,q))/(F.nx()*F.ny());
    double M = (q*f - 1)/(q-1);
    return M;
}

int main()
{
    int q = 3;
    int nx = 20; int ny = 20;
    Field<int> sigma(nx,ny); 
    for (int i=0; i<sigma.nx(); i++){
        for (int j=0; j<sigma.ny(); j++)
        { sigma(i,j) = int(q*r) + 1; }
    }
    
    double beta = 0.5;
    int N_steps = 10000;
    double step_size = 0.05;
    std::vector<double> Mag_array(N_steps);
    do
    {
        for (int i=0; i<N_steps; i++) 
        { 
            MetropolisStep(sigma, beta, q);
            double mag = Magnetisation(sigma, q);
            Mag_array[i] = mag;
        }

        // Calculating average magnetistion
        double total_mag = 0;
        int threshold = 1000;
        for (int m = threshold; m<N_steps; m++) { total_mag += Mag_array[m]; }
        double avg_mag = total_mag/(N_steps-threshold);

        // Calculating standard deviation
        double total_deviation = 0;
        for (int n = threshold; n<N_steps; n++)
        {
            double deviation = (Mag_array[n]-avg_mag) * (Mag_array[n]-avg_mag);
            total_deviation += deviation;
        }
        double st_deviation = sqrt(total_deviation/(N_steps-threshold));

        cout << beta << " " << avg_mag << " " << st_deviation << "\n";
        beta += step_size;
    } while (beta <= 1.51);

    return 0;
}