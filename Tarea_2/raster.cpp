#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>

struct V3{float x,y,z;}; struct V4{float x,y,z,w;}; struct M4{float m[4][4];};
static V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 operator*(V3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static float Len(V3 a){return std::sqrt(dot(a,a));}
static V3 nrm(V3 a){float L=Len(a);return L?V3{a.x/L,a.y/L,a.z/L}:a;}
static M4 I(){M4 r{};for(int i=0;i<4;i++)r.m[i][i]=1;return r;}
static V4 mul(M4 A,V4 v){V4 r;float q[4]{v.x,v.y,v.z,v.w};
    r.x=A.m[0][0]*q[0]+A.m[0][1]*q[1]+A.m[0][2]*q[2]+A.m[0][3]*q[3];
    r.y=A.m[1][0]*q[0]+A.m[1][1]*q[1]+A.m[1][2]*q[2]+A.m[1][3]*q[3];
    r.z=A.m[2][0]*q[0]+A.m[2][1]*q[1]+A.m[2][2]*q[2]+A.m[2][3]*q[3];
    r.w=A.m[3][0]*q[0]+A.m[3][1]*q[1]+A.m[3][2]*q[2]+A.m[3][3]*q[3]; return r;}
static M4 mul(M4 A,M4 B){M4 R{};for(int i=0;i<4;i++)for(int j=0;j<4;j++)for(int k=0;k<4;k++)R.m[i][j]+=A.m[i][k]*B.m[k][j];return R;}
static M4 T(V3 t){M4 M=I();M.m[0][3]=t.x;M.m[1][3]=t.y;M.m[2][3]=t.z;return M;}
static M4 Rx(float a){M4 M=I();float c=cosf(a),s=sinf(a);M.m[1][1]=c;M.m[1][2]=-s;M.m[2][1]=s;M.m[2][2]=c;return M;}
static M4 Ry(float a){M4 M=I();float c=cosf(a),s=sinf(a);M.m[0][0]=c;M.m[0][2]=s;M.m[2][0]=-s;M.m[2][2]=c;return M;}
static M4 Rz(float a){M4 M=I();float c=cosf(a),s=sinf(a);M.m[0][0]=c;M.m[0][1]=-s;M.m[1][0]=s;M.m[1][1]=c;return M;}
static M4 lookAt(V3 e,V3 c,V3 up){V3 f=nrm(c-e),s=nrm(cross(f,up)),u=cross(s,f);
    M4 R=I();R.m[0][0]=s.x;R.m[0][1]=s.y;R.m[0][2]=s.z;R.m[1][0]=u.x;R.m[1][1]=u.y;R.m[1][2]=u.z;R.m[2][0]=-f.x;R.m[2][1]=-f.y;R.m[2][2]=-f.z;
    return mul(R,T({-e.x,-e.y,-e.z}));}
static M4 perspective(float fovy,float asp,float n,float f){float g=1.f/tanf(0.5f*fovy*3.14159265f/180.f);M4 P{};
    P.m[0][0]=g/asp;P.m[1][1]=g;P.m[2][2]=(f+n)/(n-f);P.m[2][3]=(2*f*n)/(n-f);P.m[3][2]=-1;return P;}

struct Vtx{V3 p; cv::Vec3f col;};
struct VO{V3 cam; V4 clip; float invZ; cv::Vec3f colOverZ;};
struct Tri{int a,b,c; cv::Vec3f faceCol; V3 Ncam;};
enum Mode{PHONG=1, VERTEX=2};

int main(){
    int W=800,H=600; cv::Mat fb(H,W,CV_32FC3,cv::Scalar(0,0,0)); std::vector<float> zb(W*H,1e9);
    std::vector<Vtx> v={
        {{-1,-1,-1},{1,0,0}},{{ 1,-1,-1},{0,1,0}},{{ 1, 1,-1},{0,0,1}},{{-1, 1,-1},{1,1,0}},
        {{-1,-1, 1},{1,0,1}},{{ 1,-1, 1},{0,1,1}},{{ 1, 1, 1},{1,1,1}},{{-1, 1, 1},{0.2f,0.6f,1}}
    };
    std::vector<Tri> Tis={
        {0,1,2,{0.9f,0.3f,0.3f}}, {0,2,3,{0.9f,0.3f,0.3f}},
        {4,6,5,{0.3f,0.9f,0.3f}}, {4,7,6,{0.3f,0.9f,0.3f}},
        {0,4,5,{0.3f,0.3f,0.9f}}, {0,5,1,{0.3f,0.3f,0.9f}},
        {3,2,6,{0.9f,0.9f,0.3f}}, {3,6,7,{0.9f,0.9f,0.3f}},
        {1,5,6,{0.9f,0.3f,0.9f}}, {1,6,2,{0.9f,0.3f,0.9f}},
        {0,3,7,{0.3f,0.9f,0.9f}}, {0,7,4,{0.3f,0.9f,0.9f}}
    };
    V3 eye{3.5f,2.2f,4.2f}, up{0,1,0}; M4 V=lookAt(eye,{0,0,0},up), P=perspective(60,(float)W/H,0.1f,20.f);
    V3 L1{1.5f,1.5f,2.0f}, L2{-2.0f,-1.0f,1.5f};
    float ax=0,ay=0,az=0; int mode=PHONG; bool showHelp=true;

    auto ndc=[&](V4 q){return cv::Point3f(q.x/q.w,q.y/q.w,q.z/q.w);};
    auto toScr=[&](cv::Point3f p){return cv::Point2f((p.x*0.5f+0.5f)*W,(1.f-(p.y*0.5f+0.5f))*H);};
    auto edge=[](cv::Point2f a,cv::Point2f b,cv::Point2f c){return (c.x-a.x)*(b.y-a.y)-(c.y-a.y)*(b.x-a.x);};

    cv::namedWindow("Raster", cv::WINDOW_AUTOSIZE);
    while(true){
        M4 M = mul(mul(Ry(ay),Rx(ax)),Rz(az));
        M4 MV = mul(V,M);

        std::vector<VO> o(v.size());
        for(size_t i=0;i<v.size();++i){
            V4 p4{v[i].p.x,v[i].p.y,v[i].p.z,1}, c4=mul(MV,p4), cl=mul(P,c4);
            o[i].cam={c4.x,c4.y,c4.z}; o[i].clip=cl;
            float invZ = 1.f/std::max(1e-6f, std::fabs(c4.z));
            o[i].invZ = invZ;
            o[i].colOverZ = v[i].col * invZ;
        }
        for(auto& t:Tis){
            V3 A=o[t.a].cam,B=o[t.b].cam,C=o[t.c].cam;
            t.Ncam = nrm(cross(B-A,C-A));
        }

        fb.setTo(cv::Scalar(0,0,0)); std::fill(zb.begin(),zb.end(),1e9f);

        for(auto& tr:Tis){
            auto A=o[tr.a],B=o[tr.b],C=o[tr.c];
            if(std::fabs(A.clip.w)<1e-6f||std::fabs(B.clip.w)<1e-6f||std::fabs(C.clip.w)<1e-6f) continue;
            auto a=ndc(A.clip), b=ndc(B.clip), c=ndc(C.clip);
            auto pa=toScr(a), pb=toScr(b), pc=toScr(c);
            float minx=floorf(std::min({pa.x,pb.x,pc.x})), maxx=ceilf(std::max({pa.x,pb.x,pc.x}));
            float miny=floorf(std::min({pa.y,pb.y,pc.y})), maxy=ceilf(std::max({pa.y,pb.y,pc.y}));
            minx=std::max(0.f,minx); miny=std::max(0.f,miny); maxx=std::min((float)W-1,maxx); maxy=std::min((float)H-1,maxy);
            float Aarea=edge(pa,pb,pc); if(std::fabs(Aarea)<1e-6f) continue;

            for(int y=(int)miny;y<=(int)maxy;y++) for(int x=(int)minx;x<=(int)maxx;x++){
                cv::Point2f Pp(x+0.5f,y+0.5f);
                float w0=edge(pb,pc,Pp), w1=edge(pc,pa,Pp), w2=edge(pa,pb,Pp);
                if(!((w0>=0&&w1>=0&&w2>=0)||(w0<=0&&w1<=0&&w2<=0))) continue;
                w0/=Aarea; w1/=Aarea; w2/=Aarea;

                float invZ = w0*A.invZ + w1*B.invZ + w2*C.invZ;
                float z = 1.f/invZ;
                int idx=y*W+x; if(z>=zb[idx]) continue; zb[idx]=z;

                cv::Vec3f out(0,0,0);
                if(mode==VERTEX){
                    cv::Vec3f cOverZ = A.colOverZ*w0 + B.colOverZ*w1 + C.colOverZ*w2;
                    out = cOverZ * z;
                }else{
                    V3 pOverZ = A.cam*(w0*A.invZ) + B.cam*(w1*B.invZ) + C.cam*(w2*C.invZ);
                    V3 Pcam = pOverZ * z;
                    V3 N = tr.Ncam; V3 Vv = nrm(V3{-Pcam.x,-Pcam.y,-Pcam.z});
                    auto shade=[&](V3 Lpos){
                        V3 Ld = nrm(V3{Lpos.x-Pcam.x, Lpos.y-Pcam.y, Lpos.z-Pcam.z});
                        float ndl = std::max(0.f, dot(N,Ld));
                        V3 R = nrm(N*(2*ndl) - Ld);
                        float spec = std::pow(std::max(0.f, dot(R,Vv)), 24.f);
                        cv::Vec3f ka(0.10f,0.10f,0.10f), kd(0.95f,0.95f,0.95f), ks(0.55f,0.55f,0.55f);
                        return tr.faceCol.mul(ka) + tr.faceCol.mul(kd*ndl) + ks*spec;
                    };
                    cv::Vec3f s = shade(L1) + shade(L2);
                    out[0]=std::min(1.f,s[0]); out[1]=std::min(1.f,s[1]); out[2]=std::min(1.f,s[2]);
                }
                fb.at<cv::Vec3f>(y,x) = {out[2],out[1],out[0]};
            }
        }

        if(showHelp){
            int pad=12, th=1; double sc=0.55; int base=0;
            std::vector<std::string> lines = {
                "Raster Controles:",
                "Rotar: W/S/A/D  |  Roll: Q/E",
                "Modos: 1=Phong  |  2=VertexColor ",
                "Luz1 XY: I/J/K/L  |  Z: Z/X",
                "Ayuda: H  |  Salir: ESC",
                std::string("Modo: ") + (mode==PHONG?"Phong":"VertexColor"),
                "Luz1 (cam): x=" + std::to_string(L1.x) + " y=" + std::to_string(L1.y) + " z=" + std::to_string(L1.z)
            };
            int lineH = int(cv::getTextSize("A", cv::FONT_HERSHEY_SIMPLEX, sc, th, &base).height + 8);
            int h = (int)lines.size()*lineH + 24, w = 520;
            cv::Rect r(12,12,w,h); cv::Mat roi = fb(r);
            cv::Mat ov(roi.size(), roi.type(), cv::Scalar(0.1,0.1,0.1));
            cv::addWeighted(ov, 0.55, roi, 0.45, 0.0, roi);
            for(size_t i=0;i<lines.size();++i)
                cv::putText(fb, lines[i], {22,36+(int)i*lineH}, cv::FONT_HERSHEY_SIMPLEX, sc, cv::Scalar(1,1,1), th, cv::LINE_AA);
        }else{
            cv::putText(fb, "H: Ayuda | 1:Phong 2:VertexColor | IJKL/Z/X luz | ESC salir",
                        {12,26}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(1,1,1), 1, cv::LINE_AA);
        }

        cv::imshow("Raster", fb);
        int k = cv::waitKeyEx(1); if(k==27) break;
        float step=0.5f;
        if(k=='h'||k=='H') showHelp=!showHelp;
        if(k=='1') mode=PHONG; if(k=='2') mode=VERTEX;
        if(k=='a') ay-=0.05f; if(k=='d') ay+=0.05f; if(k=='w') ax-=0.05f; if(k=='s') ax+=0.05f; if(k=='q') az-=0.05f; if(k=='e') az+=0.05f;
        if(k=='i'||k=='I') L1.y+=step; if(k=='k'||k=='K') L1.y-=step;
        if(k=='j'||k=='J') L1.x-=step; if(k=='l'||k=='L') L1.x+=step;
        if(k=='z'||k=='Z') L1.z+=step; if(k=='x'||k=='X') L1.z-=step;
    }
    return 0;
}