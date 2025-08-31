#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
using namespace std;

struct Vec3
{
    double x, y, z;
};

struct Vec4
{
    double x, y, z, w;
};

struct Mat4
{
    double m[4][4]; // fila, columna
};

static const double EPS = 1e-6;

double dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3 &a)
{
    return sqrt(dot(a, a));
}

Vec3 sub(const Vec3 &a, const Vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec4 toVec4(const Vec3 &v, double w = 1.0)
{
    return {v.x, v.y, v.z, w};
}

Vec4 mul(const Mat4 &M, const Vec4 &v)
{
    Vec4 r{};
    r.x = M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z + M.m[0][3] * v.w;
    r.y = M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z + M.m[1][3] * v.w;
    r.z = M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z + M.m[2][3] * v.w;
    r.w = M.m[3][0] * v.x + M.m[3][1] * v.y + M.m[3][2] * v.z + M.m[3][3] * v.w;
    return r;
}

void printMat4(const Mat4 &M, const string &title)
{
    cout << "\n"
         << title << ":\n";
    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            cout << setw(12) << M.m[i][j] << " ";
        }
        cout << "\n";
    }
}

bool approxEqual(double a, double b, double eps = 1e-5)
{
    return fabs(a - b) <= eps;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "=== Parametros de la camara ===\n";
    cout << "Introduce u (right)\n";
    Vec3 u;
    cout << "u.x u.y u.z: ";
    cin >> u.x >> u.y >> u.z;

    cout << "Introduce v (up)\n";
    Vec3 v;
    cout << "v.x v.y v.z: ";
    cin >> v.x >> v.y >> v.z;

    cout << "Introduce w (forward)\n";
    Vec3 w;
    cout << "w.x w.y w.z: ";
    cin >> w.x >> w.y >> w.z;

    cout << "Introduce la posicion de la camara C\n";
    Vec3 C;
    cout << "C.x C.y C.z: ";
    cin >> C.x >> C.y >> C.z;

    // Validaciones de ortonormalidad (recomendado, no bloqueante)
    double nu = norm(u), nv = norm(v), nw = norm(w);
    bool okUnit = approxEqual(nu, 1.0, 1e-3) && approxEqual(nv, 1.0, 1e-3) && approxEqual(nw, 1.0, 1e-3);
    bool okOrtho = fabs(dot(u, v)) < 1e-3 && fabs(dot(u, w)) < 1e-3 && fabs(dot(v, w)) < 1e-3;

    if (!okUnit || !okOrtho)
    {
        cerr << "\n[ADVERTENCIA] Los vectores proporcionados no son exactamente ortonormales.\n";
        cerr << "Normas: |u|=" << nu << " |v|=" << nv << " |w|=" << nw << "\n";
        cerr << "Productos punto: u·v=" << dot(u, v) << " u·w=" << dot(u, w) << " v·w=" << dot(v, w) << "\n";
        cerr << "Se continuara de todos modos.\n";
    }

    // Matriz de Transformacion de Camara (View): Pc = R*(P - C) con R de filas u^T, v^T, w^T
    // En homogeneas: [ u.x u.y u.z  -u·C ]
    //                [ v.x v.y v.z  -v·C ]
    //                [ w.x w.y w.z  -w·C ]
    //                [ 0   0   0     1   ]
    Mat4 M_view{};
    M_view.m[0][0] = u.x;
    M_view.m[0][1] = u.y;
    M_view.m[0][2] = u.z;
    M_view.m[0][3] = -dot(u, C);
    M_view.m[1][0] = v.x;
    M_view.m[1][1] = v.y;
    M_view.m[1][2] = v.z;
    M_view.m[1][3] = -dot(v, C);
    M_view.m[2][0] = w.x;
    M_view.m[2][1] = w.y;
    M_view.m[2][2] = w.z;
    M_view.m[2][3] = -dot(w, C);
    M_view.m[3][0] = 0.0;
    M_view.m[3][1] = 0.0;
    M_view.m[3][2] = 0.0;
    M_view.m[3][3] = 1.0;

    printMat4(M_view, "Matriz de Transformacion de Camara (MTC / View)");

    // Proyeccion pinhole simple con plano de proyeccion z=1 y distancia focal f.
    // x_proj = (f * x_c) / z_c ; y_proj = (f * y_c) / z_c
    // Matriz homogenea minimalista para conseguir division por z:
    // [ f 0 0 0 ]
    // [ 0 f 0 0 ]
    // [ 0 0 1 0 ]   -> despues se hace division por w' (que sera z_c) si usamos la fila [0 0 1 0] como w'.
    // [ 0 0 1 0 ]
    cout << "\nIntroduce la distancia focal f (ej. 1, 2, 35, etc.): ";
    double f;
    cin >> f;

    Mat4 M_proj{};
    M_proj.m[0][0] = f;
    M_proj.m[0][1] = 0.0;
    M_proj.m[0][2] = 0.0;
    M_proj.m[0][3] = 0.0;
    M_proj.m[1][0] = 0.0;
    M_proj.m[1][1] = f;
    M_proj.m[1][2] = 0.0;
    M_proj.m[1][3] = 0.0;
    M_proj.m[2][0] = 0.0;
    M_proj.m[2][1] = 0.0;
    M_proj.m[2][2] = 1.0;
    M_proj.m[2][3] = 0.0;
    M_proj.m[3][0] = 0.0;
    M_proj.m[3][1] = 0.0;
    M_proj.m[3][2] = 1.0;
    M_proj.m[3][3] = 0.0;

    printMat4(M_proj, "Matriz de Proyeccion (MPC) pinhole simplificada");

    // Punto del mundo
    cout << "\n=== Punto a transformar ===\n";
    Vec3 Pw;
    cout << "P_mundo (x y z): ";
    cin >> Pw.x >> Pw.y >> Pw.z;

    // 1) Coordenadas de camara: Pc = M_view * [Pw,1]
    Vec4 Pw4 = toVec4(Pw, 1.0);
    Vec4 Pc4 = mul(M_view, Pw4);

    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    cout << "\nCoordenadas de camara Pc (antes de dividir por w, w deberia ser 1):\n";
    cout << "Pc = (" << Pc4.x << ", " << Pc4.y << ", " << Pc4.z << ", " << Pc4.w << ")\n";

    // 2) Proyeccion: Pp_h = M_proj * Pc4; luego dividir por w' (que aqui sera z_c)
    Vec4 Pp4 = mul(M_proj, Pc4);

    // Si usamos la ultima fila [0 0 1 0], entonces w' = z_c
    double wprime = Pp4.w; // deberia equivaler a Pc4.z
    if (fabs(wprime) < EPS)
    {
        cout << "\n[AVISO] El punto no es proyectable (z_c = 0). No se puede dividir.\n";
        return 0;
    }
    double x_proj = Pp4.x / wprime; // = f * x_c / z_c
    double y_proj = Pp4.y / wprime; // = f * y_c / z_c

    cout << "\nCoordenadas en el plano de proyeccion (z=1, escala f):\n";
    cout << "P_proy = (" << x_proj << ", " << y_proj << ")\n";

    // Info extra util
    cout << "\n[z_c del punto (profundidad en camara) = " << Pc4.z << " ]\n";
    if (Pc4.z < 0)
    {
        cout << "[Nota] z_c < 0 indica que el punto esta detras del centro de proyeccion segun tu convenio de ejes.\n";
    }

    cout << "\n=== Fin ===\n";
    return 0;
}