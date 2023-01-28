### Finds solutions to the Laplace equation given Dirichlet and Neumann boundary conditions

#include <iostream>
#include <vector>
#include <cmath>
using std::cout;

class Field
{
    private:
        int qx_;
        int qy_;
        std::vector<double> data_;
        int index_(int x, int y) const {return x + qx_ * y;}
    public:
        Field(int nx, int ny) : qx_(nx+1), qy_(ny+1), data_(qx_*qy_)
        {
            int n = qx_*qy_;
            for (int i=0; i<n; i++) data_[i] = 0.0;
        }

    double& operator() (int x, int y)
    { return data_[index_(x,y)]; }
    double operator() (int x, int y) const 
    { return data_[index_(x,y)]; }

    int nx() const { return qx_-1; }
    int ny() const { return qy_-1; }
};

void DirichletData(Field& q, Field& f, double d, double x1, double y1, double x2, double y2) 
//sets dirichlet data along a straight line to constant d given starting and final coordinates
{
    double hx = 1.0/double(q.nx()); double hy = 1.0/double(q.ny());
    if (x1 == x2) 
    {
        int i = round(x1*q.nx());
        int j1 = round(y1*q.ny()); // starting index 
        int j2 = round(y2*q.ny()); // final index
        for (j1; j1<=j2; j1++)
        {
            q(i,j1) = d;
            f(i,j1) = 1;
        }
    }
    else if (y1 == y2)
    {
        int i1 = round(x1*q.nx()); // starting index 
        int i2 = round(x2*q.nx()); // final index
        int j = round(y1*q.ny());
        for (i1; i1<=i2; i1++)
        {
            q(i1,j) = d;
            f(i1,j) = 1;
        }
    }
}

void SOR(Field& q, const Field& f, double w)
{
    for (int x=0; x<q.nx(); x++){
        for (int y=0; y<q.ny(); y++)
        {
            if (f(x,y) == 1) { q(x,y) = q(x,y); } // dirichlet data check
            else if (x == 0) { q(x,y) = (1.0/3.0)*(4.0*q(1,y)) - q(2,y); } // bottom neumann boudary
            else if (y == 0) { q(x,y) = (1.0/3.0)*(4.0*q(x,1) - q(x,2)); } // left neumann boundary
            else { q(x,y) = (1.0-w)*q(x,y) + 0.25*w*(q(x-1,y) + q(x,y-1) + q(x+1,y) + q(x,y+1)); }
        }
    } 
}

void printField(const Field& q)
{
    for (int i=0; i<=q.nx(); i++){
        for (int j=0; j<=q.ny(); j++)
        {
            double x = double(i)/double(q.nx()); double y = double(j)/double(q.ny());
            cout << x << " " << y << " " << q(i,j) << "\n";
        }
    } 
}

double norm(const Field& q)
{
    double sum = 0.0;
    for (int i=0; i<=q.nx(); i++){
        for (int j=0; j<=q.ny(); j++)
        {
            sum += q(i,j) * q(i,j);
        }
    }
    return sqrt(sum);
}

int main()
{
    int nx = 10*35; int ny = 10*35;
    double hx = 1.0/double(nx); double hy = 1.0/double(ny); 
    Field phi(nx,ny);
    Field F(nx,ny); // Tells us at which grid point there is known dirichlet data 

    for (int i=0; i<=phi.nx(); i++) 
    {
        phi(i,0) = 0.0; F(i,0) = 1.0; //bottom edge
        phi(i,phi.ny()) = i*hx; //top edge
    }
    for (int j=0; j<=phi.ny(); j++)
    {
        phi(0,j) = 0.0; F(0,j) = 1.0; //left edge
        phi(phi.nx(),j) = j*hy; //right edge
    } 
    // line B
    DirichletData(phi, F, -1.0, 0.6, 0.5, 0.8, 0.5);
    DirichletData(phi, F, -1.0, 0.8, 0.1, 0.8, 0.5);
    // box A
    DirichletData(phi, F, 1.0, 0.2, 0.7, 0.2, 0.9);
    DirichletData(phi, F, 1.0, 0.2, 0.9, 0.6, 0.9);
    DirichletData(phi, F, 1.0, 0.2, 0.7, 0.6, 0.7);
    DirichletData(phi, F, 1.0, 0.6, 0.7, 0.6, 0.9);

    double omega = 1.96;
    double tol = 1e-8; double diff = 1.0; int k = 0;
    while (diff > tol)
    {
        double iter_k = norm(phi);
        SOR(phi,F,omega);
        double iter_k1 = norm(phi);
        diff = std::abs(iter_k - iter_k1);
        k ++;
        // cout << k << " " << norm(phi) << "\n"; // generates output data for plotting
    }
    cout << "Convergence with tolerance " << tol << " after k = " << k << " steps for omega = " << omega << " and grid size " << nx << " x " << ny << "\n";

    //printField(phi);

    // derivative computation
    double xd = 0.3; double yd = 0.5;
    int id = round(xd*nx); int jd = round(yd*ny);
    double deriv = (1.0/(2.0*hy)) * (phi(id,jd+1) - phi(id,jd-1)); // central difference
    cout << "The derivatve of phi wrt y at the point (0.3, 0.5) is " << deriv << "\n";

    // neumann boundary condition
    Field phi_N(nx,ny); 
    Field F_N(nx,ny);
    for (int i=0; i<=phi_N.nx(); i++) { phi_N(i,phi_N.ny()) = i*hx; } // top edge
    for (int j=0; j<=phi_N.ny(); j++) { phi_N(phi_N.nx(),j) = j*hy; } // right edge
    DirichletData(phi_N, F_N, -1.0, 0.6, 0.5, 0.8, 0.5);
    DirichletData(phi_N, F_N, -1.0, 0.8, 0.1, 0.8, 0.5);
    DirichletData(phi_N, F_N, 1.0, 0.2, 0.7, 0.2, 0.9);
    DirichletData(phi_N, F_N, 1.0, 0.2, 0.9, 0.6, 0.9);
    DirichletData(phi_N, F_N, 1.0, 0.2, 0.7, 0.6, 0.7);
    DirichletData(phi_N, F_N, 1.0, 0.6, 0.7, 0.6, 0.9);

    int steps_N = k; 
    for (int l=0; l<steps_N; l++) { SOR(phi_N,F_N,omega); }

    //printField(phi_N);

    double deriv_N = (1.0/(2*hy)) * (phi_N(id,jd+1) - phi_N(id,jd-1));
    cout << "The derivatve of phi with Neumann b.c wrt y at the point (0.3, 0.5) is " << deriv_N << "\n";

    return 0;
}
