### A system of four planets, moving in a two-dimensional plane with masses, initial observed positions and velocities. 
### They move according to Newton’s inverse-square law of gravity. 
### Following code finds positions of planets using a leap-frog symplectic integrator.

#include <iostream>
#include <vector>
#include <cmath>
using std::vector;
using std::cout;

class Planet
{
    private:
        double m_;
    public:
        double pos[2];
        double vel[2];

    Planet(double m, double x0, double x1, double v0, double v1) // Constructor
    {
        m_ = m;
        pos[0] = x0; pos[1] = x1;
        vel[0] = v0; vel[1] = v1;
    }

    double mass() const{
        return m_;
    }
};

vector<double> F(Planet& p, vector<Planet>& solar_system)  // Finding acceleration of platnet p due to other planets in solar system
{
    double G = 1.0; // gravitational constant
    vector<double> acc = {0,0}; 
    int n = solar_system.size();
    for (int j=0; j<n; j++)  // iterating through planets of solar system
    {
        if (p.mass() != solar_system[j].mass()) // single out 3 other planets contributing to force
        {
            vector<double> d = {p.pos[0] - solar_system[j].pos[0], p.pos[1] - solar_system[j].pos[1]};
            double r = sqrt(d[0]*d[0] + d[1]*d[1]); 
            double fj = -(G * solar_system[j].mass())/(r*r*r);

            for (int a=0; a<2; a++) // x or y coordinate
            {
                acc[a] += fj * d[a];
            }
        }
    }
    return acc;
}
 
void update_position(double h, Planet& p)
{
    for (int a=0; a<2; a++)
    {
        p.pos[a] += (h/2.0)*p.vel[a];
    }
}

void update_velocity(double h, Planet&p, vector<Planet>& solar_system)
{
    for (int a=0; a<2; a++)
    {
        p.vel[a] += h * F(p, solar_system)[a];
    }
}

void LeapFrogStep(double h, vector<Planet>& solar_system)
{
    int m = solar_system.size();
    for (int i=0; i<m; i++) {update_position(h, solar_system[i]);}
    for (int i=0; i<m; i++) {update_velocity(h, solar_system[i], solar_system);}
    for (int i=0; i<m; i++) {update_position(h, solar_system[i]);}
}

int main()
{
    vector<Planet> my_solar_system;

    // Add element to end of array
    my_solar_system.push_back(Planet(2.2, -0.5, 0.1, -0.84, 0.65));
    my_solar_system.push_back(Planet(0.8, -0.6, -0.2, 1.86, 0.7));
    my_solar_system.push_back(Planet(0.9, 0.5, 0.1, -0.44, -1.5));
    my_solar_system.push_back(Planet(0.4, 0.5, 0.4, 1.15, -1.6));

    double I = 5.0; // interval size
    int n = 5000; // number of steps
    double h = I/double(n); // step size
    double t = 0;
    for (int k=0; k<n; k++)
    {
        LeapFrogStep(h, my_solar_system);
        //cout << my_solar_system[0].pos[0] << " " << my_solar_system[0].pos[1] << " " << my_solar_system[1].pos[0] << " " << my_solar_system[1].pos[1] << " " 
        //<< my_solar_system[2].pos[0] << " " << my_solar_system[2].pos[1] << " " << my_solar_system[3].pos[0] << " " << my_solar_system[3].pos[1] << "\n";
        t += h;
    }
    cout << "Position of planet 0 at time t = " << t << " is (" << my_solar_system[0].pos[0] << " , " << my_solar_system[0].pos[1] << ") \n";
    cout << "Position of planet 1 at time t = " << t << " is (" << my_solar_system[1].pos[0] << " , " << my_solar_system[1].pos[1] << ") \n";
    cout << "Position of planet 2 at time t = " << t << " is (" << my_solar_system[2].pos[0] << " , " << my_solar_system[2].pos[1] << ") \n";
    cout << "Position of planet 3 at time t = " << t << " is (" << my_solar_system[3].pos[0] << " , " << my_solar_system[3].pos[1] << ") \n";

    return 0;
}
