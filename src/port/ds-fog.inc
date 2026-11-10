// (11-10) The launcher's fog and embers, on the GPU: one full-screen pixel shader drawn through the SDL renderer's own
// Direct3D 11 device, from an ImGui draw callback between the background image and the menus. Included by ds-menu.cpp
// (Windows only). The fog: three layers of domain-warped gradient-noise fBm (parallax, denser at the ground), lit cold
// by the sky and warm by the torches of the key art. The embers: loose single sparks per noise cell in three depths,
// rising with a slow drift and a wobble, cooling from white-yellow to deep red over their life, flickering, each with
// a motion streak and a soft glow (the near ones larger and softer). Premultiplied alpha, dithered (no banding).
// Nothing is drawn when the renderer is not Direct3D 11 or the shader does not compile (the plain art stays).
#if defined(_WIN32)
#include <d3d11.h>
#include <d3dcompiler.h>

namespace FogGpu {

SDL_Renderer*            g_renderer = nullptr;
ID3D11Device*            g_dev      = nullptr;
ID3D11VertexShader*      g_vs       = nullptr;
ID3D11PixelShader*       g_ps       = nullptr;
ID3D11Buffer*            g_cb       = nullptr;
ID3D11BlendState*        g_blend    = nullptr;
ID3D11RasterizerState*   g_rs       = nullptr;
ID3D11DepthStencilState* g_dss      = nullptr;
bool                     g_failed   = false;
bool                     g_logged   = false;
Uint64                   g_t0       = 0;

struct alignas(16) Params {
	float res[2];   // the render target, pixels
	float time;     // seconds
	float intensity;
	float img[4];   // the key art's rectangle on screen: x, y, w, h (pixels)
};
Params g_params {};

const char* kShader = R"HLSL(
cbuffer P : register(b0) { float2 res; float time; float intensity; float4 img; };

float4 vs_main(uint id : SV_VertexID) : SV_Position {
	float2 p = float2((id << 1) & 2, id & 2);
	return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
}

float hash12(float2 p) { float3 p3 = frac(p.xyx * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return frac((p3.x + p3.y) * p3.z); }
float2 hash22(float2 p) { float3 p3 = frac(p.xyx * float3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return frac((p3.xx + p3.yz) * p3.zy); }
float3 hash33(float3 p) { p = frac(p * float3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return -1 + 2 * frac((p.xxy + p.yxx) * p.zyx); }

float gnoise(float3 p) {
	float3 i = floor(p), f = frac(p);
	float3 u = f * f * f * (f * (f * 6 - 15) + 10);
	float n000 = dot(hash33(i), f);
	float n100 = dot(hash33(i + float3(1, 0, 0)), f - float3(1, 0, 0));
	float n010 = dot(hash33(i + float3(0, 1, 0)), f - float3(0, 1, 0));
	float n110 = dot(hash33(i + float3(1, 1, 0)), f - float3(1, 1, 0));
	float n001 = dot(hash33(i + float3(0, 0, 1)), f - float3(0, 0, 1));
	float n101 = dot(hash33(i + float3(1, 0, 1)), f - float3(1, 0, 1));
	float n011 = dot(hash33(i + float3(0, 1, 1)), f - float3(0, 1, 1));
	float n111 = dot(hash33(i + float3(1, 1, 1)), f - float3(1, 1, 1));
	return lerp(lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y), lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y), u.z);
}

float fbm(float3 p, int oct) {
	float s = 0, a = 0.5;
	[loop] for (int i = 0; i < oct; i++) { s += a * gnoise(p); p = float3(p.xy * 2.03 + float2(17.1, -9.7), p.z * 1.6); a *= 0.5; }
	return s;
}

// One fog layer: domain-warped fBm, drifting with the wind, slowly evolving in its own time axis. 0 .. 1.
float fogLayer(float2 p, float t, float scale, float2 wind, float seed) {
	float3 q = float3(p * scale + wind * t, seed + t * 0.04);
	float2 w = float2(fbm(q, 4), fbm(q + float3(5.2, 1.3, 3.1), 4));
	float n = fbm(q + float3(w * 1.7, 0) + float3(-wind * t * 0.35, t * 0.025), 5);
	return saturate(n * 1.55 + 0.5);
}

// The torches of the key art (image uv) and their flicker.
static const float2 kTorch[4] = { float2(0.009, 0.50), float2(0.075, 0.595), float2(0.437, 0.67), float2(0.963, 0.69) };

float torchLight(float2 iuv, float t, float spread) {
	float aspect = img.z / max(img.w, 1);
	float l = 0;
	[unroll] for (int i = 0; i < 4; i++) {
		float2 d = (iuv - kTorch[i]) * float2(aspect, 1);
		float flick = 0.82 + 0.10 * sin(t * 7.3 + i * 1.7) + 0.08 * sin(t * 13.1 + i * 4.1);
		l += exp(-dot(d, d) * spread) * flick;
	}
	return l;
}

float3 embers(float2 px, float t) {
	float unit = res.y / 1080.0;
	float3 acc = 0;
	[unroll] for (int L = 0; L < 3; L++) {
		float k     = L / 2.0;                       // 0 far .. 1 near
		float cell  = lerp(46.0, 140.0, k) * unit;   // one possible spark per cell
		float rise  = lerp(28.0, 80.0, k) * unit;    // px / s upwards
		float drift = rise * 0.30;                   // px / s to the right (the wind)
		float2 q    = px + float2(-drift, rise) * t; // the field moves with the sparks
		float2 c    = floor(q / cell);
		[unroll] for (int dy = -1; dy <= 1; dy++)
		[unroll] for (int dx = -1; dx <= 1; dx++) {
			float2 id = c + float2(dx, dy) + L * 131.0;
			float2 h  = hash22(id);
			float  h3 = hash12(id + 7.7);
			float  h4 = hash12(id + 3.3);
			float  per  = 5.0 + 7.0 * h3;                     // seconds per life
			float  life = frac(t / per + h.x * 7.0);
			float  wob  = t * (0.6 + 0.9 * h3) + h.y * 6.2831;
			float2 lp   = (c + float2(dx, dy) + 0.15 + 0.7 * h) * cell;
			lp.x += (sin(wob) * 0.22 + sin(wob * 2.7 + 1.3) * 0.07) * cell;
			lp.y += cos(wob * 0.8) * 0.10 * cell;
			float2 sp = lp - float2(-drift, rise) * t;        // back to the screen
			// Where sparks are born: low on the screen and near the torches; loose elsewhere.
			float2 su  = sp / res;
			float2 siu = (sp - img.xy) / max(img.zw, 1);
			float tnear = torchLight(siu, 0, 9.0);
			float prob = (0.05 + 0.40 * smoothstep(0.35, 1.0, su.y) + 0.55 * saturate(tnear)) * lerp(0.75, 0.45, k);
			float alive = step(h4, prob);
			float fade  = smoothstep(0.0, 0.10, life) * (1 - smoothstep(0.50, 1.0, life));
			float flick = 0.65 + 0.35 * sin(t * (9 + 13 * h.x) + h.y * 40) * sin(t * (4.3 + 5 * h.y) + h3 * 9);
			float hot   = 1 - life;
			float3 col  = lerp(float3(0.75, 0.10, 0.02), float3(1.0, 0.50, 0.12), saturate(hot * 1.4));
			col         = lerp(col, float3(1.0, 0.88, 0.62), saturate(hot - 0.6) * 2.5);
			// The streak: along the velocity, tail behind the spark.
			float2 vel  = float2(drift + cos(wob) * 0.22 * cell * (0.6 + 0.9 * h3), -rise - sin(wob * 0.8) * 0.08 * cell);
			float2 dir  = normalize(vel);
			float2 rel  = px - sp;
			float along = dot(rel, dir), across = dot(rel, float2(-dir.y, dir.x));
			float tail  = lerp(2.5, 9.0, k) * unit * (0.6 + 0.8 * h.x);
			float rad   = lerp(0.75, 1.9, k) * unit * (0.8 + 0.4 * h3);
			float back  = along - clamp(along, -tail, 0);
			float d2    = back * back + across * across;
			float taper = 1 - 0.6 * saturate(-along / tail);
			float core  = exp(-d2 / (rad * rad)) * taper;
			float glow  = exp(-d2 / (rad * rad * 16)) * 0.22;
			float b     = alive * fade * flick * (0.55 + 0.9 * h3) * lerp(0.7, 1.0, k) * (1.0 - 0.25 * k);
			acc += col * (core * 1.8 + glow) * b;
		}
	}
	return acc;
}

float4 ps_main(float4 pos : SV_Position) : SV_Target {
	float2 px = pos.xy;
	float2 uv = px / res;
	float2 p  = float2(uv.x * res.x / res.y, uv.y);
	float2 iuv = (px - img.xy) / max(img.zw, 1);
	float t = time;

	// Fog: far (large, slow) .. near (small, faster).
	float g  = smoothstep(0.40, 1.0, uv.y);        // the ground
	float d1 = fogLayer(p, t, 1.0, float2(0.030, 0.003), 1.0);
	float d2 = fogLayer(p, t, 2.1, float2(0.055, -0.004), 7.0);
	float d3 = g > 0.01 ? fogLayer(p, t, 3.8, float2(0.090, 0.0), 13.0) : 0;
	float dens = smoothstep(0.30, 0.85, d1) * 0.50
	           + smoothstep(0.38, 0.90, d2) * 0.38 * (0.30 + 0.70 * g)
	           + smoothstep(0.45, 0.95, d3) * 0.30 * g;
	dens *= 0.30 + 0.95 * g;
	// The logo (top left) stays clean.
	dens *= 1 - 0.75 * smoothstep(0.42, 0.05, uv.y) * smoothstep(0.34, 0.02, uv.x);
	float a = saturate(dens * intensity);

	float tl = torchLight(iuv, t, 26.0);
	float3 cool = float3(0.56, 0.61, 0.69) * (0.80 + 0.30 * d2) * (0.85 + 0.25 * (1 - uv.y));
	float3 warm = float3(1.0, 0.52, 0.22);
	float3 fogc = lerp(cool, warm * 0.9, saturate(tl * 0.9)) + warm * tl * 0.18;

	float3 e = embers(px, t) * saturate(intensity * 1.4);
	float3 rgb = fogc * a + e;
	float dith = (hash12(px + frac(t) * 61.0) + hash12(px * 1.37 + 17.0) - 1.0) / 255.0;
	return float4(rgb + dith, saturate(a + dith));
}
)HLSL";

template <class T>
void Release(T*& p) {
	if (p != nullptr) p->Release();
	p = nullptr;
}

void Shutdown() {
	Release(g_vs);
	Release(g_ps);
	Release(g_cb);
	Release(g_blend);
	Release(g_rs);
	Release(g_dss);
	g_dev    = nullptr; // (SDL's: not ours to release)
	g_failed = false;
	g_logged = false;
}

bool Init() {
	if (g_ps != nullptr) return true;
	if (g_failed || g_renderer == nullptr) return false;
	g_failed = true;
	g_dev    = SDL_RenderGetD3D11Device(g_renderer);
	if (g_dev == nullptr) {
		std::printf("[port] launcher fog: the renderer is not Direct3D 11, no fog\n");
		return false;
	}
	g_dev->Release(); // (SDL_RenderGetD3D11Device adds a reference; SDL keeps the device alive)
	HMODULE dll = LoadLibraryW(L"d3dcompiler_47.dll");
	auto*   compile = dll != nullptr ? reinterpret_cast<pD3DCompile>(GetProcAddress(dll, "D3DCompile")) : nullptr;
	if (compile == nullptr) {
		std::printf("[port] launcher fog: d3dcompiler_47.dll not found, no fog\n");
		return false;
	}
	ID3DBlob* vsb = nullptr;
	ID3DBlob* psb = nullptr;
	ID3DBlob* err = nullptr;
	const size_t len = std::strlen(kShader);
	HRESULT hr = compile(kShader, len, "fog", nullptr, nullptr, "vs_main", "vs_4_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vsb, &err);
	if (SUCCEEDED(hr)) {
		Release(err);
		hr = compile(kShader, len, "fog", nullptr, nullptr, "ps_main", "ps_4_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &psb, &err);
	}
	if (FAILED(hr)) {
		std::printf("[port] launcher fog: shader not compiled (0x%08lx): %s\n", static_cast<unsigned long>(hr),
		            err != nullptr ? static_cast<const char*>(err->GetBufferPointer()) : "");
		Release(err);
		Release(vsb);
		Release(psb);
		return false;
	}
	hr = g_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_vs);
	if (SUCCEEDED(hr)) hr = g_dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_ps);
	Release(vsb);
	Release(psb);
	D3D11_BUFFER_DESC bd {};
	bd.ByteWidth      = sizeof(Params);
	bd.Usage          = D3D11_USAGE_DYNAMIC;
	bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	if (SUCCEEDED(hr)) hr = g_dev->CreateBuffer(&bd, nullptr, &g_cb);
	D3D11_BLEND_DESC bl {};
	bl.RenderTarget[0].BlendEnable           = TRUE;
	bl.RenderTarget[0].SrcBlend              = D3D11_BLEND_ONE; // (premultiplied: the fog's colour x alpha, the embers added)
	bl.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
	bl.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
	bl.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ZERO;
	bl.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ONE;
	bl.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
	bl.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	if (SUCCEEDED(hr)) hr = g_dev->CreateBlendState(&bl, &g_blend);
	D3D11_RASTERIZER_DESC rd {};
	rd.FillMode        = D3D11_FILL_SOLID;
	rd.CullMode        = D3D11_CULL_NONE;
	rd.DepthClipEnable = TRUE;
	if (SUCCEEDED(hr)) hr = g_dev->CreateRasterizerState(&rd, &g_rs);
	D3D11_DEPTH_STENCIL_DESC dd {};
	dd.DepthEnable   = FALSE;
	dd.StencilEnable = FALSE;
	if (SUCCEEDED(hr)) hr = g_dev->CreateDepthStencilState(&dd, &g_dss);
	if (FAILED(hr)) {
		std::printf("[port] launcher fog: Direct3D objects not made (0x%08lx), no fog\n", static_cast<unsigned long>(hr));
		Shutdown();
		g_failed = true;
		return false;
	}
	g_failed = false;
	g_t0     = SDL_GetPerformanceCounter();
	std::printf("[port] launcher fog: GPU shader ready\n");
	return true;
}

// The ImGui draw callback: SDL's queued draws first, then ours over them with every state SDL relies on put back.
void Callback(const ImDrawList* /*list*/, const ImDrawCmd* /*cmd*/) {
	if (!Init()) return;
	SDL_RenderFlush(g_renderer);
	ID3D11DeviceContext* ctx = nullptr;
	g_dev->GetImmediateContext(&ctx);
	if (ctx == nullptr) return;

	ID3D11RenderTargetView* rtv = nullptr;
	ID3D11DepthStencilView* dsv = nullptr;
	ctx->OMGetRenderTargets(1, &rtv, &dsv);
	if (rtv == nullptr) {
		Release(dsv);
		ctx->Release();
		return;
	}
	// The target's size.
	{
		ID3D11Resource* r = nullptr;
		rtv->GetResource(&r);
		ID3D11Texture2D* tex = nullptr;
		if (r != nullptr && SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex)))) {
			D3D11_TEXTURE2D_DESC td {};
			tex->GetDesc(&td);
			g_params.res[0] = static_cast<float>(td.Width);
			g_params.res[1] = static_cast<float>(td.Height);
			tex->Release();
		}
		Release(r);
	}
	if (g_params.res[0] <= 0 || g_params.res[1] <= 0) {
		Release(rtv);
		Release(dsv);
		ctx->Release();
		return;
	}
	g_params.time = static_cast<float>(std::fmod(
	    static_cast<double>(SDL_GetPerformanceCounter() - g_t0) / static_cast<double>(SDL_GetPerformanceFrequency()), 3600.0));

	// Save.
	D3D11_VIEWPORT           vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] {};
	UINT                     nvp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_RECT               sc[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] {};
	UINT                     nsc = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	ID3D11RasterizerState*   old_rs = nullptr;
	ID3D11BlendState*        old_bs = nullptr;
	FLOAT                    old_bf[4] {};
	UINT                     old_mask = 0;
	ID3D11DepthStencilState* old_ds   = nullptr;
	UINT                     old_sref = 0;
	ID3D11VertexShader*      old_vs   = nullptr;
	ID3D11PixelShader*       old_ps   = nullptr;
	ID3D11GeometryShader*    old_gs   = nullptr;
	ID3D11Buffer*            old_vcb  = nullptr;
	ID3D11Buffer*            old_pcb  = nullptr;
	ID3D11InputLayout*       old_il   = nullptr;
	D3D11_PRIMITIVE_TOPOLOGY old_topo {};
	ctx->RSGetViewports(&nvp, vp);
	ctx->RSGetScissorRects(&nsc, sc);
	ctx->RSGetState(&old_rs);
	ctx->OMGetBlendState(&old_bs, old_bf, &old_mask);
	ctx->OMGetDepthStencilState(&old_ds, &old_sref);
	ctx->VSGetShader(&old_vs, nullptr, nullptr);
	ctx->PSGetShader(&old_ps, nullptr, nullptr);
	ctx->GSGetShader(&old_gs, nullptr, nullptr);
	ctx->VSGetConstantBuffers(0, 1, &old_vcb);
	ctx->PSGetConstantBuffers(0, 1, &old_pcb);
	ctx->IAGetInputLayout(&old_il);
	ctx->IAGetPrimitiveTopology(&old_topo);

	// Draw.
	D3D11_MAPPED_SUBRESOURCE m {};
	if (SUCCEEDED(ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		std::memcpy(m.pData, &g_params, sizeof(g_params));
		ctx->Unmap(g_cb, 0);
	}
	D3D11_VIEWPORT full {};
	full.Width    = g_params.res[0];
	full.Height   = g_params.res[1];
	full.MaxDepth = 1.0f;
	ctx->RSSetViewports(1, &full);
	ctx->RSSetState(g_rs);
	const FLOAT zero[4] {};
	ctx->OMSetBlendState(g_blend, zero, 0xffffffffu);
	ctx->OMSetDepthStencilState(g_dss, 0);
	ctx->IASetInputLayout(nullptr);
	ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	ctx->VSSetShader(g_vs, nullptr, 0);
	ctx->PSSetShader(g_ps, nullptr, 0);
	ctx->GSSetShader(nullptr, nullptr, 0);
	ctx->VSSetConstantBuffers(0, 1, &g_cb);
	ctx->PSSetConstantBuffers(0, 1, &g_cb);
	ctx->Draw(3, 0);

	// Restore.
	ctx->RSSetViewports(nvp, vp);
	ctx->RSSetScissorRects(nsc, sc);
	ctx->RSSetState(old_rs);
	ctx->OMSetBlendState(old_bs, old_bf, old_mask);
	ctx->OMSetDepthStencilState(old_ds, old_sref);
	ctx->VSSetShader(old_vs, nullptr, 0);
	ctx->PSSetShader(old_ps, nullptr, 0);
	ctx->GSSetShader(old_gs, nullptr, 0);
	ctx->VSSetConstantBuffers(0, 1, &old_vcb);
	ctx->PSSetConstantBuffers(0, 1, &old_pcb);
	ctx->IASetInputLayout(old_il);
	ctx->IASetPrimitiveTopology(old_topo);
	Release(old_rs);
	Release(old_bs);
	Release(old_ds);
	Release(old_vs);
	Release(old_ps);
	Release(old_gs);
	Release(old_vcb);
	Release(old_pcb);
	Release(old_il);
	Release(rtv);
	Release(dsv);
	ctx->Release();
}

// Queued on the background list after the art; the art's rectangle in ImGui units -> target pixels (the callback scales).
void Draw(ImDrawList* dl, float W, float H, float img_x, float img_y, float img_w, float img_h, float intensity) {
	if (g_failed || g_renderer == nullptr || W <= 0 || H <= 0) return;
	const ImGuiIO& io = ImGui::GetIO();
	const float    sx = io.DisplayFramebufferScale.x > 0 ? io.DisplayFramebufferScale.x : 1.0f;
	const float    sy = io.DisplayFramebufferScale.y > 0 ? io.DisplayFramebufferScale.y : 1.0f;
	g_params.img[0]    = img_x * sx;
	g_params.img[1]    = img_y * sy;
	g_params.img[2]    = img_w * sx;
	g_params.img[3]    = img_h * sy;
	g_params.intensity = intensity;
	dl->AddCallback(Callback, nullptr);
	dl->AddCallback(ImGui::GetPlatformIO().DrawCallback_ResetRenderState, nullptr);
}

} // namespace FogGpu
#endif
