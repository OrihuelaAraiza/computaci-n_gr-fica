#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

using std::array;
using std::vector;

struct Vec3
{
    float x, y, z;
};
struct Vec4
{
    float x, y, z, w;
};
struct Mat4
{
    float m[4][4];
};

static inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(const Vec3 &a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline Vec3 operator/(const Vec3 &a, float s) { return {a.x / s, a.y / s, a.z / s}; }
static inline float dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3 &a, const Vec3 &b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline float length(const Vec3 &a) { return std::sqrt(dot(a, a)); }
static inline Vec3 normalize(const Vec3 &a)
{
    float L = length(a);
    return L > 0 ? a / L : a;
}

static inline Vec4 mul(const Mat4 &M, const Vec4 &v)
{
    Vec4 r;
    r.x = M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z + M.m[0][3] * v.w;
    r.y = M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z + M.m[1][3] * v.w;
    r.z = M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z + M.m[2][3] * v.w;
    r.w = M.m[3][0] * v.x + M.m[3][1] * v.y + M.m[3][2] * v.z + M.m[3][3] * v.w;
    return r;
}
static inline Mat4 mul(const Mat4 &A, const Mat4 &B)
{
    Mat4 R{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            R.m[i][j] = 0;
            for (int k = 0; k < 4; ++k)
                R.m[i][j] += A.m[i][k] * B.m[k][j];
        }
    return R;
}

static Mat4 identity()
{
    Mat4 M{};
    for (int i = 0; i < 4; ++i)
        M.m[i][i] = 1;
    return M;
}
static Mat4 translate(const Vec3 &t)
{
    Mat4 M = identity();
    M.m[0][3] = t.x;
    M.m[1][3] = t.y;
    M.m[2][3] = t.z;
    return M;
}
static Mat4 rotateX(float a)
{
    Mat4 M = identity();
    float c = std::cos(a), s = std::sin(a);
    M.m[1][1] = c;
    M.m[1][2] = -s;
    M.m[2][1] = s;
    M.m[2][2] = c;
    return M;
}
static Mat4 rotateY(float a)
{
    Mat4 M = identity();
    float c = std::cos(a), s = std::sin(a);
    M.m[0][0] = c;
    M.m[0][2] = s;
    M.m[2][0] = -s;
    M.m[2][2] = c;
    return M;
}
static Mat4 rotateZ(float a)
{
    Mat4 M = identity();
    float c = std::cos(a), s = std::sin(a);
    M.m[0][0] = c;
    M.m[0][1] = -s;
    M.m[1][0] = s;
    M.m[1][1] = c;
    return M;
}
static Mat4 lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up)
{
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 R = identity();
    R.m[0][0] = s.x;
    R.m[0][1] = s.y;
    R.m[0][2] = s.z;
    R.m[1][0] = u.x;
    R.m[1][1] = u.y;
    R.m[1][2] = u.z;
    R.m[2][0] = -f.x;
    R.m[2][1] = -f.y;
    R.m[2][2] = -f.z;
    Mat4 T = translate({-eye.x, -eye.y, -eye.z});
    return mul(R, T);
}
static Mat4 perspective(float fovy_deg, float aspect, float znear, float zfar)
{
    float f = 1.0f / std::tan((fovy_deg * 0.5f) * float(M_PI / 180.0));
    Mat4 P{};
    P.m[0][0] = f / aspect;
    P.m[1][1] = f;
    P.m[2][2] = (zfar + znear) / (znear - zfar);
    P.m[2][3] = (2 * znear * zfar) / (znear - zfar);
    P.m[3][2] = -1.0f;
    return P;
}

struct Vertex
{
    Vec3 posObj;
    cv::Vec3f color;
};
struct VOut
{
    Vec3 posCam;
    Vec4 posClip;
    cv::Vec3f colorOverZ;
    float invZ;
};
struct Tri
{
    int i0, i1, i2;
    cv::Vec3f faceColor;
    Vec3 normalCam;
};
struct Light
{
    Vec3 posCam;
    cv::Vec3f Ia{0.05f, 0.05f, 0.05f};
    cv::Vec3f Id{1, 1, 1};
    cv::Vec3f Is{1, 1, 1};
};
enum class ShadeMode
{
    Phong = 0,
    Depth = 1,
    VertexColor = 2
};

struct Raster
{
    int W, H;
    cv::Mat frame;
    vector<float> zbuf;
    ShadeMode mode = ShadeMode::Phong;
    Raster(int w, int h) : W(w), H(h)
    {
        frame = cv::Mat(H, W, CV_32FC3, cv::Scalar(0, 0, 0));
        zbuf.resize(W * H, std::numeric_limits<float>::infinity());
    }
    void clear(cv::Scalar bg = cv::Scalar(0, 0, 0))
    {
        frame.setTo(bg);
        std::fill(zbuf.begin(), zbuf.end(), std::numeric_limits<float>::infinity());
    }
    inline bool inBounds(int x, int y) const { return x >= 0 && x < W && y >= 0 && y < H; }
    static inline float edge(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c) { return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x); }
    void drawTriangle(const VOut &v0, const VOut &v1, const VOut &v2, const cv::Point2f &p0, const cv::Point2f &p1, const cv::Point2f &p2, const Tri &tri, const vector<Light> &lights)
    {
        float minx = std::floor(std::min({p0.x, p1.x, p2.x}));
        float maxx = std::ceil(std::max({p0.x, p1.x, p2.x}));
        float miny = std::floor(std::min({p0.y, p1.y, p2.y}));
        float maxy = std::ceil(std::max({p0.y, p1.y, p2.y}));
        minx = std::max(0.f, minx);
        miny = std::max(0.f, miny);
        maxx = std::min(float(W - 1), maxx);
        maxy = std::min(float(H - 1), maxy);
        float area = edge(p0, p1, p2);
        if (std::abs(area) < 1e-6f)
            return;
        float invZ0 = v0.invZ, invZ1 = v1.invZ, invZ2 = v2.invZ;
        for (int y = int(miny); y <= int(maxy); ++y)
        {
            for (int x = int(minx); x <= int(maxx); ++x)
            {
                cv::Point2f P(x + 0.5f, y + 0.5f);
                float w0 = edge(p1, p2, P);
                float w1 = edge(p2, p0, P);
                float w2 = edge(p0, p1, P);
                if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0))
                {
                    w0 /= area;
                    w1 /= area;
                    w2 /= area;
                    float invZ = w0 * invZ0 + w1 * invZ1 + w2 * invZ2;
                    float z = 1.0f / invZ;
                    int idx = y * W + x;
                    if (z >= zbuf[idx])
                        continue;
                    zbuf[idx] = z;
                    cv::Vec3f color(0, 0, 0);
                    if (mode == ShadeMode::Depth)
                    {
                        float zN = std::min(std::max((z - 0.1f) / (10.0f - 0.1f), 0.0f), 1.0f);
                        color = cv::Vec3f(1, 1, 1) * (1.0f - zN);
                    }
                    else if (mode == ShadeMode::VertexColor)
                    {
                        cv::Vec3f cOverZ = v0.colorOverZ * w0 + v1.colorOverZ * w1 + v2.colorOverZ * w2;
                        color = cOverZ * z;
                    }
                    else
                    {
                        Vec3 N = normalize(tri.normalCam);
                        Vec3 pOverZ = (v0.posCam * (w0 * invZ0) + v1.posCam * (w1 * invZ1) + v2.posCam * (w2 * invZ2));
                        Vec3 Pcam = pOverZ * z;
                        Vec3 V = normalize(Vec3{-Pcam.x, -Pcam.y, -Pcam.z});
                        cv::Vec3f base = tri.faceColor;
                        float ka = 0.10f, kd = 0.95f, ks = 0.55f, shininess = 32.0f;
                        cv::Vec3f sum(0, 0, 0);
                        for (const auto &Lgt : lights)
                        {
                            Vec3 L = normalize(Vec3{Lgt.posCam.x - Pcam.x, Lgt.posCam.y - Pcam.y, Lgt.posCam.z - Pcam.z});
                            float NdotL = std::max(0.0f, dot(N, L));
                            cv::Vec3f diff = (base.mul(Lgt.Id)) * (kd * NdotL);
                            Vec3 R = normalize(N * (2.0f * NdotL) - L);
                            float specPow = std::pow(std::max(0.0f, dot(R, V)), shininess);
                            cv::Vec3f spec = Lgt.Is * (ks * specPow);
                            cv::Vec3f amb = base.mul(Lgt.Ia) * ka;
                            sum += (amb + diff + spec);
                        }
                        color[0] = std::min(sum[0], 1.0f);
                        color[1] = std::min(sum[1], 1.0f);
                        color[2] = std::min(sum[2], 1.0f);
                    }
                    frame.at<cv::Vec3f>(y, x) = cv::Vec3f(color[2], color[1], color[0]);
                }
            }
        }
    }
};

static void drawPanel(cv::Mat &img, cv::Rect rect, float alpha = 0.55f)
{
    cv::Mat roi = img(rect);
    cv::Mat overlay(roi.size(), roi.type(), cv::Scalar(0.1f, 0.1f, 0.1f));
    cv::addWeighted(overlay, alpha, roi, 1.0f - alpha, 0.0, roi);
}
static void putLines(cv::Mat &img, int x, int y, const std::vector<std::string> &lines, double scale = 0.55, int thickness = 1)
{
    int baseline = 0;
    int lineH = int(std::round(cv::getTextSize("A", cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline).height + 8));
    for (size_t i = 0; i < lines.size(); ++i)
    {
        cv::putText(img, lines[i], {x, y + int(i) * lineH}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(1.0, 1.0, 1.0), thickness, cv::LINE_AA);
    }
}
static std::string modeName(ShadeMode m)
{
    switch (m)
    {
    case ShadeMode::Phong:
        return "Phong (1)";
    case ShadeMode::Depth:
        return "Depth as color (2)";
    case ShadeMode::VertexColor:
        return "Vertex color (3)";
    }
    return "";
}
static std::string saveScreenshot(const cv::Mat &frame)
{
    namespace fs = std::filesystem;
    fs::path dir = "screenshots";
    fs::create_directories(dir);
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << "frame_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".png";
    fs::path file = dir / oss.str();
    cv::Mat out8;
    frame.convertTo(out8, CV_8UC3, 255.0);
    cv::imwrite(file.string(), out8);
    return file.string();
}

int main()
{
    const int W = 800, H = 600;
    Raster rast(W, H);
    vector<Vertex> verts = {
        {{-1, -1, -1}, {1, 0, 0}}, {{1, -1, -1}, {0, 1, 0}}, {{1, 1, -1}, {0, 0, 1}}, {{-1, 1, -1}, {1, 1, 0}}, {{-1, -1, 1}, {1, 0, 1}}, {{1, -1, 1}, {0, 1, 1}}, {{1, 1, 1}, {1, 1, 1}}, {{-1, 1, 1}, {0.2f, 0.6f, 1}}};
    vector<Tri> tris = {
        {0, 1, 2, {0.9f, 0.3f, 0.3f}}, {0, 2, 3, {0.9f, 0.3f, 0.3f}}, {4, 6, 5, {0.3f, 0.9f, 0.3f}}, {4, 7, 6, {0.3f, 0.9f, 0.3f}}, {0, 4, 5, {0.3f, 0.3f, 0.9f}}, {0, 5, 1, {0.3f, 0.3f, 0.9f}}, {3, 2, 6, {0.9f, 0.9f, 0.3f}}, {3, 6, 7, {0.9f, 0.9f, 0.3f}}, {1, 5, 6, {0.9f, 0.3f, 0.9f}}, {1, 6, 2, {0.9f, 0.3f, 0.9f}}, {0, 3, 7, {0.3f, 0.9f, 0.9f}}, {0, 7, 4, {0.3f, 0.9f, 0.9f}}};
    Vec3 eye{3.5f, 2.2f, 4.2f};
    Vec3 center{0, 0, 0};
    Vec3 up{0, 1, 0};
    Mat4 V = lookAt(eye, center, up);
    Mat4 P = perspective(60.0f, float(W) / H, 0.1f, 20.0f);
    vector<Light> lights(2);
    lights[0].posCam = {1.5f, 1.5f, 2.0f};
    lights[1].posCam = {-2.0f, -1.0f, 1.5f};
    float angX = 0, angY = 0, angZ = 0;
    bool autorotate = true;
    bool showHelp = true;
    std::string toast;
    int toastFrames = 0;

    cv::namedWindow("Raster", cv::WINDOW_AUTOSIZE);

    while (true)
    {
        if (autorotate)
        {
            angY += 0.01f;
            angX += 0.005f;
        }
        Mat4 M = mul(mul(rotateY(angY), rotateX(angX)), rotateZ(angZ));
        Mat4 MV = mul(V, M);
        vector<VOut> vout(verts.size());
        for (size_t i = 0; i < verts.size(); ++i)
        {
            const auto &v = verts[i];
            Vec4 posObj4{v.posObj.x, v.posObj.y, v.posObj.z, 1.0f};
            Vec4 posCam4 = mul(MV, posObj4);
            Vec4 posClip = mul(P, posCam4);
            Vec3 posCam{posCam4.x, posCam4.y, posCam4.z};
            VOut o;
            o.posCam = posCam;
            o.posClip = posClip;
            float invZ = 1.0f / (std::max(1e-6f, std::abs(posCam4.z)));
            o.invZ = invZ;
            o.colorOverZ = cv::Vec3f(verts[i].color[0] * invZ, verts[i].color[1] * invZ, verts[i].color[2] * invZ);
            vout[i] = o;
        }
        for (auto &t : tris)
        {
            Vec3 a = vout[t.i0].posCam;
            Vec3 b = vout[t.i1].posCam;
            Vec3 c = vout[t.i2].posCam;
            t.normalCam = normalize(cross(b - a, c - a));
        }
        rast.clear();
        for (const auto &t : tris)
        {
            const auto &A = vout[t.i0], &B = vout[t.i1], &C = vout[t.i2];
            if (std::abs(A.posClip.w) < 1e-6f || std::abs(B.posClip.w) < 1e-6f || std::abs(C.posClip.w) < 1e-6f)
                continue;
            auto ndc = [](const Vec4 &p)
            { return cv::Point3f(p.x / p.w, p.y / p.w, p.z / p.w); };
            cv::Point3f a = ndc(A.posClip), b = ndc(B.posClip), c = ndc(C.posClip);
            auto toScreen = [&](const cv::Point3f &p)
            {
                float sx = (p.x * 0.5f + 0.5f) * rast.W;
                float sy = (1.0f - (p.y * 0.5f + 0.5f)) * rast.H;
                return cv::Point2f(sx, sy);
            };
            cv::Point2f pa = toScreen(a), pb = toScreen(b), pc = toScreen(c);
            rast.drawTriangle(A, B, C, pa, pb, pc, t, lights);
        }

        if (showHelp)
        {
            const int pad = 14;
            std::vector<std::string> lines = {
                "Rasterizador de Cubo  Controles:",
                "Rotar: W/S/A/D  |  Roll: Q/E  |  Auto-rotacion: SPACE",
                "Cambiar modo: 1=Phong, 2=Depth, 3=VertexColor",
                "Mover luz (X,Y): Flechas o I/J/K/L  |  Z: PgUp/PgDn o Z/X",
                "Guardar captura: P  (carpeta ./screenshots)",
                "Ocultar/mostrar ayuda: H  |  Salir: ESC",
                std::string("Modo: ") + modeName(rast.mode),
                std::string("Auto-rotacion: ") + (autorotate ? "ON" : "OFF"),
                "Luz1 pos (cam): x=" + std::to_string(lights[0].posCam.x) +
                    " y=" + std::to_string(lights[0].posCam.y) +
                    " z=" + std::to_string(lights[0].posCam.z)};
            double scale = 0.55;
            int thickness = 1, baseline = 0;
            int lineH = int(std::round(cv::getTextSize("A", cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline).height + 8));
            int panelH = int(lines.size() * lineH + pad * 1.2);
            int panelW = 620;
            cv::Rect panel(pad, pad, panelW, std::min(panelH, H - 2 * pad));
            drawPanel(rast.frame, panel, 0.55f);
            putLines(rast.frame, panel.x + 12, panel.y + 24, lines, scale, thickness);
        }
        else
        {
            cv::putText(rast.frame, "H: Ayuda  |  1/2/3 modos  |  Flechas/IJKL y Z/X mueven luz  |  P: captura  |  ESC salir",
                        {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(1.0, 1.0, 1.0), 1, cv::LINE_AA);
        }

        if (toastFrames > 0)
        {
            cv::putText(rast.frame, toast, {16, H - 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(1.0, 1.0, 1.0), 1, cv::LINE_AA);
            --toastFrames;
        }

        cv::imshow("Raster", rast.frame);
        int key = cv::waitKeyEx(1);
        if (key == 27)
            break;
        float step = 0.5f;
        switch (key)
        {
        case 'h':
        case 'H':
            showHelp = !showHelp;
            break;
        case ' ':
            autorotate = !autorotate;
            break;
        case 'a':
            angY -= 0.05f;
            break;
        case 'd':
            angY += 0.05f;
            break;
        case 'w':
            angX -= 0.05f;
            break;
        case 's':
            angX += 0.05f;
            break;
        case 'q':
            angZ -= 0.05f;
            break;
        case 'e':
            angZ += 0.05f;
            break;
        case '1':
            rast.mode = ShadeMode::Phong;
            break;
        case '2':
            rast.mode = ShadeMode::Depth;
            break;
        case '3':
            rast.mode = ShadeMode::VertexColor;
            break;
        case 65362:
        case 82:
        case 2490368:
            lights[0].posCam.y += step;
            break;
        case 65364:
        case 84:
        case 2621440:
            lights[0].posCam.y -= step;
            break;
        case 65361:
        case 81:
        case 2424832:
            lights[0].posCam.x -= step;
            break;
        case 65363:
        case 83:
        case 2555904:
            lights[0].posCam.x += step;
            break;
        case 'i':
        case 'I':
            lights[0].posCam.y += step;
            break;
        case 'k':
        case 'K':
            lights[0].posCam.y -= step;
            break;
        case 'j':
        case 'J':
            lights[0].posCam.x -= step;
            break;
        case 'l':
        case 'L':
            lights[0].posCam.x += step;
            break;
        case 'z':
        case 'Z':
            lights[0].posCam.z += step;
            break;
        case 'x':
        case 'X':
            lights[0].posCam.z -= step;
            break;
        case 0x21:
        case 2162688:
            lights[0].posCam.z += step;
            break;
        case 0x22:
        case 2228224:
            lights[0].posCam.z -= step;
            break;
        case 'p':
        case 'P':
        {
            std::string path = saveScreenshot(rast.frame);
            toast = "Captura guardada: " + path;
            toastFrames = 120;
        }
        break;
        default:
            break;
        }
    }
    return 0;
}