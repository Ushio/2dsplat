#include "pr.hpp"
#include <iostream>
#include <memory>

uint32_t pcg(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

glm::uvec3 pcg3d(glm::uvec3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

glm::vec3 sign_of(glm::vec3 v)
{
    return {
        v.x < 0.0f ? -1.0f : 1.0f,
        v.y < 0.0f ? -1.0f : 1.0f,
        v.z < 0.0f ? -1.0f : 1.0f
    };
}
float sign_of(float v)
{
    return v < 0.0f ? -1.0f : 1.0f;
}

struct Splat
{
    glm::vec2 pos;
    float radius;
    glm::vec3 color;
};

int focus = -1;//510

#define POS_PURB 0.1f
#define RADIUS_PURB 0.1f
#define COLOR_PURB 0.01f

#define RADIUS_MAX 16.0f

enum
{
    SIGNBIT_POS_X = 0,
    SIGNBIT_POS_Y,
    SIGNBIT_RADIUS,
    SIGNBIT_COL_R,
    SIGNBIT_COL_G,
    SIGNBIT_COL_B,
};

bool bitAt(uint32_t u, uint32_t i)
{
    return u & (1u << i);
}

// 0: +1, 1: -1
float signAt( uint32_t u, uint32_t i )
{
    return bitAt( u, i ) ? -1.0f : 1.0f;
}

uint32_t splatRng(uint32_t i, uint32_t perturbIdx)
{
    return pcg(i + pcg(perturbIdx));
}

Splat perturb(Splat splat, uint32_t r, float s )
{
    splat.pos.x += s * POS_PURB * signAt(r, SIGNBIT_POS_X);
    splat.pos.y += s * POS_PURB * signAt(r, SIGNBIT_POS_Y);

    splat.radius += s * RADIUS_PURB * signAt(r, SIGNBIT_RADIUS);

    splat.color.x += s * COLOR_PURB * signAt(r, SIGNBIT_COL_R);
    splat.color.y += s * COLOR_PURB * signAt(r, SIGNBIT_COL_G);
    splat.color.z += s * COLOR_PURB * signAt(r, SIGNBIT_COL_B);

    // constraints
    splat.radius = glm::clamp(splat.radius, 4.0f, RADIUS_MAX);
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
void drawSplats( pr::Image2DRGBA32* image, std::vector<int>* splatIndices, const std::vector<Splat>& splats, uint32_t perturbIdx, float s )
{
    int w = image->width();
    int h = image->height();
    for( int i = 0; i < splats.size() ; i++ )
    {
        Splat splat = splats[i];

        // Apply perturb
        splat = perturb(splat, splatRng( i, perturbIdx ), s);

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
                    float T = std::expf( -2 * d2 / (splat.radius * splat.radius));
                    glm::vec3 c = (*image)(x, y);
                    c = glm::mix( c, splat.color, T );
                    (*image)(x, y) = glm::vec4(c, 1.0f);

                    if (splatIndices)
                    {
                        splatIndices[y * w + x].push_back(i);
                    }
                }
            }
        }
    }
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
    camera.origin = { 0, 0, 4 };
    camera.lookat = { 0, 0, 0 };

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
    
    int NSplat = 512;
    std::vector<Splat> splats(NSplat);

    float beta1t = 1.0f;
    float beta2t = 1.0f;
    std::vector<SplatAdam> splatAdams(splats.size());


    ITexture* tex0 = CreateTexture();
    ITexture* tex1 = CreateTexture();
    Image2DRGBA32 image0;
    image0.allocate(imageRef.width(), imageRef.height());
    Image2DRGBA32 image1;
    image1.allocate(imageRef.width(), imageRef.height());

    std::vector<std::vector<int>> indices0(imageRef.width()* imageRef.height());
    //std::vector<int> indices1(imageRef.width() * imageRef.height());

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

    int N = 16;
    int perturbIdx = 0;
    int iterations = 0;

    auto init = [&]() {
        for (int i = 0; i < NSplat; i++)
        {
            glm::vec3 r0 = glm::vec3(pcg3d({ i, 0, 0xFFFFFFFF })) / glm::vec3(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

            Splat s;
            s.pos.x = glm::mix(r0.x, (float)imageRef.width() - 1, r0.x);
            s.pos.y = glm::mix(r0.y, (float)imageRef.height() - 1, r0.y);
            s.radius = 8;
            s.color = { 0.5f ,0.5f ,0.5f };
            splats[i] = s;
        }

        beta1t = 1.0f;
        beta2t = 1.0f;
        splatAdams.clear();
        splatAdams.resize(NSplat);

        perturbIdx = 0;
        iterations = 0;
    };

    init();

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
            // std::fill(indices0.begin(), indices0.end(), -1);
            // std::fill(indices1.begin(), indices1.end(), -1);
            for (int i = 0; i < indices0.size(); i++)
                indices0[i].clear();

            drawSplats(&image0, indices0.data(), splats, perturbIdx, scale);
            //if(k + 1 == N)
            //{
            //    tex0->upload(image0);
            //}
            //tex0->upload(image0);

            drawSplats(&image1, nullptr, splats, perturbIdx, -scale);
            //tex1->upload(image1);

            // accumurate derivatives
        
            for (int y = 0; y < image0.height(); y++)
            {
                for (int x = 0; x < image0.width(); x++)
                {
                    glm::vec3 d0 = imageRef(x, y) - image0(x, y);
                    glm::vec3 d1 = imageRef(x, y) - image1(x, y);
                    float fwh0 = lengthSquared(d0);
                    float fwh1 = lengthSquared(d1);
                    float df = fwh0 - fwh1;

                    for (int i : indices0[y * image0.width() + x])
                    {
                        if (i != focus && focus != -1)
                            continue;

                        uint32_t r = splatRng(i, perturbIdx);

                        // based on the paper, no div by 2, no div by eps
                        // only s_i is taken into account.
                        // checking bits is enough to determine the signs.
                        dSplats[i].pos.x += df * signAt(r, SIGNBIT_POS_X);
                        dSplats[i].pos.y += df * signAt(r, SIGNBIT_POS_Y);

                        dSplats[i].radius += df * signAt(r, SIGNBIT_RADIUS);

                        dSplats[i].color.x += df * signAt(r, SIGNBIT_COL_R);
                        dSplats[i].color.y += df * signAt(r, SIGNBIT_COL_G);
                        dSplats[i].color.z += df * signAt(r, SIGNBIT_COL_B);
                    }
                }
            }
            perturbIdx++;
        }

        // gradient decent
        beta1t *= ADAM_BETA1;
        beta2t *= ADAM_BETA2;

        float alpha = 1.0f;
        // based on the paper, no div by N, but use purb amount as learning rate.

        for (int i = 0; i < splats.size(); i++)
        {
            if (i != focus && focus != -1)
                continue;

            splats[i].pos.x = splatAdams[i].pos[0].optimize(splats[i].pos.x, dSplats[i].pos.x, POS_PURB * alpha, beta1t, beta2t);
            splats[i].pos.y = splatAdams[i].pos[1].optimize(splats[i].pos.y, dSplats[i].pos.y, POS_PURB * alpha, beta1t, beta2t);
            
            if ( i == focus)
                printf("%.5f %.5f\n", dSplats[i].color.x / (float)N, dSplats[i].color.y / (float)N);

            splats[i].radius = splatAdams[i].radius.optimize(splats[i].radius, dSplats[i].radius, RADIUS_PURB * alpha, beta1t, beta2t);

            splats[i].color.x = splatAdams[i].color[0].optimize(splats[i].color.x, dSplats[i].color.x, COLOR_PURB * alpha, beta1t, beta2t );
            splats[i].color.y = splatAdams[i].color[1].optimize(splats[i].color.y, dSplats[i].color.y, COLOR_PURB * alpha, beta1t, beta2t );
            splats[i].color.z = splatAdams[i].color[2].optimize(splats[i].color.z, dSplats[i].color.z, COLOR_PURB * alpha, beta1t, beta2t );

            //splats[i].color -= 0.0001f * dSplats[i].color / (float)N;

            // constraints
            splats[i].pos.x = glm::clamp(splats[i].pos.x, 0.0f, (float)imageRef.width() - 1);
            splats[i].pos.y = glm::clamp(splats[i].pos.y, 0.0f, (float)imageRef.height() - 1);
            splats[i].radius = glm::clamp(splats[i].radius, 4.0f, RADIUS_MAX);
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
                mse += lengthSquared(d * 255.0f);
            }
        }
        mse /= (image0.height() * image0.width() * 3 );

        float scaling = 0.01f;
        for (int i = 0; i < splats.size(); i++)
        {
            DrawCircle(glm::vec3(splats[i].pos.x, -splats[i].pos.y, 0.0f)* scaling, {0,0,1}, glm::u8vec3(splats[i].color * 255.0f), splats[i].radius* scaling);
        }

        iterations++;

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
        ImGui::SetNextWindowSize({ 600, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::Text("mse = %.5f", mse);
        ImGui::Text("iterations = %d", iterations);
        ImGui::Text("perturbIdx = %d", perturbIdx);

        if (ImGui::Button("Restart"))
        {
            init();
        }

        ImGui::Image(textureRef, ImVec2(textureRef->width() * 2, textureRef->height() * 2));
        ImGui::Image(tex0, ImVec2(tex0->width() * 2, tex0->height() * 2));
        ImGui::Image(tex1, ImVec2(tex1->width() * 2, tex1->height() * 2));

        ImGui::End();

        //ImGui::SetNextWindowPos({ 800, 20 }, ImGuiCond_Once);
        //ImGui::SetNextWindowSize({ 600, 300 }, ImGuiCond_Once);
        //ImGui::Begin("Params");
        //ImGui::SliderFloat("scale", &scale, 0, 1);
        //ImGui::End();

        EndImGui();
    }

    pr::CleanUp();

    return 0;
}
