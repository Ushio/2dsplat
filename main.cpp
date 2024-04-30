#include "pr.hpp"
#include <iostream>
#include <memory>

glm::uvec3 pcg3d(glm::uvec3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}


glm::vec3 randomsign3d(glm::uvec3 v)
{
    glm::vec3 r = glm::vec3(pcg3d(v)) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    return glm::sign(r - glm::vec3(0.5f, 0.5f, 0.5f));
}

struct Splat
{
    glm::vec2 pos;
    float radius;
    glm::vec3 color;
};

int focus = -1;//510

Splat perturb(Splat splat, uint32_t i, uint32_t perturbIdx, float s )
{
    if (i != focus && focus != -1)
        return splat;

    glm::vec3 r0 = randomsign3d({ i, 0, perturbIdx });
    glm::vec3 r1 = randomsign3d({ i, 1, perturbIdx });
    // glm::vec3 r0 = glm::vec3(pcg3d({ i, 0, perturbIdx })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    // glm::vec3 r1 = glm::vec3(pcg3d({ i, 1, perturbIdx })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

    splat.pos.x += s * glm::mix(-0.05f, 0.05f, r0.x);
    splat.pos.y += s * glm::mix(-0.05f, 0.05f, r0.y);

    splat.radius += s * glm::mix(-0.1f, 0.1f, r0.z);
    splat.color.x += s * glm::mix(-0.01f, 0.01f, r1.x);
    splat.color.y += s * glm::mix(-0.01f, 0.01f, r1.y);
    splat.color.z += s * glm::mix(-0.01f, 0.01f, r1.z);

    // constraints
    splat.radius = glm::clamp(splat.radius, 4.0f, 8.0f);
    splat.color = glm::clamp(splat.color, { 0.0f, 0.0f,0.0f }, { 1.0f ,1.0f ,1.0f });

    return splat;
}
float lengthSquared(glm::vec2 v)
{
    return v.x * v.x + v.y * v.y;
}
float lengthSquared(glm::vec3 v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
void drawSplats( pr::Image2DRGBA32* image, int* splatIndices, const std::vector<Splat>& splats, uint32_t perturbIdx, float s )
{
    int w = image->width();
    int h = image->height();
    for( int i = 0; i < splats.size() ; i++ )
    {
        Splat splat = splats[i];

        // Apply perturb
        splat = perturb(splat, i, perturbIdx, s);

        glm::ivec2 lower = glm::ivec2( glm::floor( splat.pos - glm::vec2(splat.radius, splat.radius) ) );
        glm::ivec2 upper = glm::ivec2( glm::ceil( splat.pos + glm::vec2(splat.radius, splat.radius) ) );

        lower = glm::clamp(lower, glm::ivec2(0, 0), glm::ivec2(w - 1, h - 1));
        upper = glm::clamp(upper, glm::ivec2(0, 0), glm::ivec2(w - 1, h - 1));

        for (int y = lower.y; y <= upper.y; y++)
        {
            for (int x = lower.x; x <= upper.x; x++)
            {
                float d2 = lengthSquared(splat.pos - glm::vec2((float)x, (float)y));
                if (d2 < splat.radius * splat.radius)
                {
                    (*image)(x, y) = glm::vec4(splat.color, 1.0f);

                    if (splatIndices)
                    {
                        splatIndices[y * w + x] = i;
                    }
                }
            }
        }
    }
}


float epsDiv( float x, float y, float eps )
{
    // return x / y;
    if (y < 0.0f)
    {
        eps = -eps;
    }
    return x / (y + eps);
}

const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.99f;

struct Adam
{
    float m_m;
    float m_v;

    float optimize( float value, float g, float alpha, float beta1t, float beta2t )
    {
        float s = alpha;
        float m = ADAM_BETA1 * m_m + (1.0f - ADAM_BETA1) * g;
        float v = ADAM_BETA2 * m_v + (1.0f - ADAM_BETA2) * g * g;
        m_m = m;
        m_v = v;
        float m_hat = m / (1.0f - beta1t);
        float v_hat = v / (1.0f - beta2t);

        const float ADAM_E = 1.0e-15f;
        return value - s * m_hat / (sqrt(v_hat) + ADAM_E);
    }
};
struct SplatAdam
{
    Adam pos[2];
    Adam radius;
    Adam color[3];
};


int main() {
    using namespace pr;

    SetDataDir(ExecutableDir());

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = true;

    double e = GetElapsedTime();

    ITexture* textureRef = CreateTexture();
    Image2DRGBA32 imageRef;
    {
        Image2DRGBA8 image;
        image.load("squirrel_cls_mini.jpg");
        imageRef = Image2DRGBA8_to_Image2DRGBA32(image);
    }
    // std::fill(imageRef.data(), imageRef.data() + imageRef.width() * imageRef.height(), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
    //for (int y = 0; y < imageRef.height(); y++)
    //{
    //    for (int x = 0; x < imageRef.width(); x++)
    //    {
    //        imageRef(x, y) = glm::vec4((float)x / imageRef.width(), 1- (float)x / imageRef.width(), 0.0f, 1.0f);
    //    }
    //}

    textureRef->upload(imageRef);
    
    std::vector<Splat> splats;

    for (int i = 0; i < 512; i++)
    {
        glm::vec3 r0 = glm::vec3(pcg3d({ i, 0, 0xFFFFFFFF })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        glm::vec3 r1 = glm::vec3(pcg3d({ i, 1, 0xFFFFFFFF })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

        Splat s;
        s.pos.x = glm::mix(r0.x, (float)imageRef.width() - 1, r0.x);
        s.pos.y = glm::mix(r0.y, (float)imageRef.height() - 1, r0.y);
        s.radius = 4;// 8 + 32 * r0.z;
        // s.color = r1;
        s.color = { 0.5f ,0.5f ,0.5f };
        splats.push_back(s);
    }

    float beta1t = 1.0f;
    float beta2t = 1.0f;
    std::vector<SplatAdam> splatAdams(splats.size());


    ITexture* tex0 = CreateTexture();
    ITexture* tex1 = CreateTexture();
    Image2DRGBA32 image0;
    image0.allocate(imageRef.width(), imageRef.height());
    Image2DRGBA32 image1;
    image1.allocate(imageRef.width(), imageRef.height());

    std::vector<int> indices0(imageRef.width()* imageRef.height());
    std::vector<int> indices1(imageRef.width() * imageRef.height());

    // drawSplats(&image0, splats, 0 );

    //for (int y = 0; y < image0.height(); y++)
    //{
    //    for (int x = 0; x < image0.width(); x++)
    //    {
    //        glm::vec3 r0 = glm::vec3(pcg3d({ x, y, 0 })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    //        image0(x, y) = glm::vec4(r0, 1.0f);
    //    }
    //}

    // tex0->upload(image0);
    int N = 64;
    int perturbIdx = 0;

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        static float scale = 1.0f;

        // gradients
        std::vector<Splat> dSplats(splats.size());
        
        
        for (int k = 0; k < N; k++)
        {
            // clear images
            std::fill(image0.data(), image0.data() + image0.width() * image0.height(), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            std::fill(image1.data(), image1.data() + image1.width() * image1.height(), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            std::fill(indices0.begin(), indices0.end(), -1);
            std::fill(indices1.begin(), indices1.end(), -1);

            drawSplats(&image0, indices0.data(), splats, perturbIdx, scale);
            //if(k + 1 == N)
            //{
            //    tex0->upload(image0);
            //}
            //tex0->upload(image0);

            drawSplats(&image1, indices1.data(), splats, perturbIdx, -scale);
            //tex1->upload(image1);

            // accumurate derivatives
        
            for (int y = 0; y < image0.height(); y++)
            {
                for (int x = 0; x < image0.width(); x++)
                {
                    // should take both?
                    int i = indices0[y * image0.width() + x];
                    if( i == -1 )
                    {
                        //i = indices1[y * image0.width() + x];
                        //
                        //if( i == -1)
                            continue;
                    }

                    if (i != focus && focus != -1)
                        continue;

                    glm::vec3 d0 = imageRef(x, y) - image0(x, y);
                    glm::vec3 d1 = imageRef(x, y) - image1(x, y);
                    float fwh0 = lengthSquared(d0);
                    float fwh1 = lengthSquared(d1);
                    float df = fwh0 - fwh1;
                
                    Splat s0 = perturb(splats[i], i, perturbIdx, scale );
                    Splat s1 = perturb(splats[i], i, perturbIdx, -scale );

                    // x2 is missing
                    dSplats[i].pos.x += epsDiv( df, s0.pos.x - s1.pos.x, 1.0e-15f );
                    dSplats[i].pos.y += epsDiv( df, s0.pos.y - s1.pos.y, 1.0e-15f );

                    dSplats[i].radius += epsDiv( df, s0.radius - s1.radius, 1.0e-15f );

                    dSplats[i].color.x += epsDiv( df, s0.color.x - s1.color.x, 1.0e-15f );
                    dSplats[i].color.y += epsDiv( df, s0.color.y - s1.color.y, 1.0e-15f );
                    dSplats[i].color.z += epsDiv( df, s0.color.z - s1.color.z, 1.0e-15f );
                }
            }
            perturbIdx++;
        }

        // gradient decent
        beta1t *= ADAM_BETA1;
        beta2t *= ADAM_BETA2;

        float alpha = 1.0f;
        // float alpha = 0.0000001f;

        for (int i = 0; i < splats.size(); i++)
        {
            if (i != focus && focus != -1)
                continue;

            splats[i].pos.x = splatAdams[i].pos[0].optimize(splats[i].pos.x, dSplats[i].pos.x / (float)N, 0.05f * alpha, beta1t, beta2t);
            splats[i].pos.y = splatAdams[i].pos[1].optimize(splats[i].pos.y, dSplats[i].pos.y / (float)N, 0.05f * alpha, beta1t, beta2t);
            
            if ( i == focus)
                printf("%.5f %.5f\n", dSplats[i].color.x / (float)N, dSplats[i].color.y / (float)N);

            splats[i].radius = splatAdams[i].radius.optimize(splats[i].radius, dSplats[i].radius / (float)N, 0.1f * alpha, beta1t, beta2t);

            splats[i].color.x = splatAdams[i].color[0].optimize(splats[i].color.x, dSplats[i].color.x / (float)N, 0.05f * alpha, beta1t, beta2t );
            splats[i].color.y = splatAdams[i].color[1].optimize(splats[i].color.y, dSplats[i].color.y / (float)N, 0.05f * alpha, beta1t, beta2t );
            splats[i].color.z = splatAdams[i].color[2].optimize(splats[i].color.z, dSplats[i].color.z / (float)N, 0.05f * alpha, beta1t, beta2t );

            //splats[i].color -= 0.0001f * dSplats[i].color / (float)N;

            // constraints
            splats[i].pos.x = glm::clamp(splats[i].pos.x, 0.0f, (float)imageRef.width() - 1);
            splats[i].pos.y = glm::clamp(splats[i].pos.y, 0.0f, (float)imageRef.height() - 1);
            splats[i].radius = glm::clamp(splats[i].radius, 4.0f, 8.0f);
            splats[i].color = glm::clamp(splats[i].color, { 0.0f, 0.0f ,0.0f }, { 1.0f ,1.0f ,1.0f });
        }

        std::fill(image0.data(), image0.data() + image0.width() * image0.height(), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        drawSplats(&image0, 0, splats, 0, 0);
        //for (int y = 0; y < image0.height(); y++)
        //{
        //    for (int x = 0; x < image0.width(); x++)
        //    {
        //        int idx = indices0[y * image0.width() + x];
        //        if (idx != -1)
        //        {
        //            image0(x, y) = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
        //        }
        //    }
        //}

        tex0->upload(image0);

        double mse = 0.0;
        for (int y = 0; y < image0.height(); y++)
        {
            for (int x = 0; x < image0.width(); x++)
            {
                glm::vec3 d = image0(x, y) - imageRef(x, y);
                mse += lengthSquared(d);
            }
        }

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::Text("mse = %.3f", mse);
        ImGui::Text("perturbIdx = %d", perturbIdx);
        ImGui::Image(textureRef, ImVec2(textureRef->width(), textureRef->height()));
        ImGui::Image(tex0, ImVec2(tex0->width(), tex0->height()));
        ImGui::Image(tex1, ImVec2(tex1->width(), tex1->height()));

        ImGui::End();

        ImGui::SetNextWindowPos({ 800, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 300 }, ImGuiCond_Once);
        ImGui::Begin("Params");
        ImGui::SliderFloat("scale", &scale, 0, 1);
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();

    return 0;
}
