# Sombras nativas (Demon's Souls PS5, KytyPS5) — especificación para el siguiente agente

Estado al 25-11: medidor `KYTY_NATIVO_PASE_SOMBRAS=2` y base B1 (`=3`, copia de la capa) escritos; **=3 completo (dibujo nativo + comparación) y =1 NO escritos**.
Normas del usuario: Ghidra para todo dato del juego; nunca leer memoria del juego con un candado tomado ni desde el hilo de la GPU; cada fallo aborta en silencio y vuelve al emulador; una línea por cascada en el log.

## 1. Ghidra (projelf_par1: direcciones del juego tal cual; eboot.raw en /home/claude/gh está desplazado +0xbdb0)

- `CascadedShadows` 0xa11220: por cascada i=0..3 lanza un trabajo (0xa04ed0 RunFunctionAsRenderJob) → 0xa13650 (viewport desde render+0x758+(i+0x29)*0x50) → **0xa13e60(ctx)**.
  - ctx[0x10] = lista visible de la cascada (render+0xe8b8 − i*0xae0); ctx[0x15] = i; ctx[0xa] = render+0xdf0+i*8 (objeto de vista de la cascada);
  - **ctx+0xa0 (ctx[0x14]) = sesgo de profundidad**: (pendiente 1.1, constante 7.5) pases 0..2, (1.1, 2.0) pase 3 (0xa11220 copia *(param+0x250+i*8); 0xa13e60 lo pasa a 0x902350 = descriptor de estado de rasterizado).
- **0xa13e60** empieza con **0xa00eb0(ctx)** (vista de la cascada) y luego dibuja: `0xab57d0(renderer, lista, ctx, cubo, i)` con cubos 0,1,3; 0xa57af0 (VirtualGeo); 0xa9c9a0; `0xab6080(i)` y `(i+4)`.
- **0xa00eb0** (leído hasta el RET 0xa0484d):
  - `0x9098d0(global+0x650)` (global = *(eboot+0x2f0a5f8)): reserva un bloque nuevo del montón del fotograma (0x7f9cf0) y lo pone como base del CBuffer +0x650; lo devuelve (R15); se rellena desde R13 = ctx[0xa] (misma disposición 0x720 bytes que el búfer de cámara: +0x1c0, +0x2dc..+0x314, +0x6c0..+0x6d0, +0x710…).
  - 0xa047de: `0x9131c0(cmd, dst = base de global+0x678, src = bloque nuevo, tamaño)` → **la GPU copia la vista de la cascada sobre el búfer de cámara fijo** (269c83000 en la prueba) DENTRO de la DCB, antes de los draws.
  - 0xa04801: **`ctx+0x100 = bloque nuevo` (vista de la cascada, legible desde la CPU)**; 0xa0481f: `ctx+0xb8 = base global+0x678` (búfer de cámara fijo = tabla[1] del VS).
  - Por eso todos los volcados de 269c83000 desde la CPU mostraban la cámara principal: en memoria CPU el contenido de la cascada solo llega cuando la GPU ejecuta esa copia.
- **0xab57d0(renderer, lista, ctx, cubo, pase)**: entradas de 0x20 bytes en `lista+0x178+cubo*0x28` (inicio, fin). Devuelve el nº de entradas. Por entrada:
  - `item = *(entrada+8)`; item[0] = material (material+0x20 técnica, 0xa84ee0 la enlaza al cambiar; material+0x3f4 = máscara de recursos);
  - item[2] = instancia → 0xabc4b0 → **0xabc100**: `sub = *(inst+0x130) + *(int*)(entrada+0x14)*0x48`; **CMesh = *(*(sub+0x20)+0x70)**; 0x9f80a0(sub, ctx) elige vértices; 0xaa1e20(CMesh, ctx) = 0x9142c0(cmd, CMesh+8) enlaza el **índice** (V# en CMesh+8: base 48 bits, nº en +0x10 (dword +8), 32 bits si (dword+0xC & 0x3FFF0000)==0x40000 si no 16).
  - **skinning**: 0x9f80a0: si sub+0x38 != 0 y max(*(u32*)(sub38+0x1c), +0x20) != 0 → vértices por fotograma (sub+0x30 + k*0x60); si no, estáticos (CMesh+0x238).
  - lista de USER_DATA: `0x7f9cf0` reserva POPCOUNT(material+0x3f4) punteros; bit b → ctx[0x16+b] (bit0 ctx+0xb0, bit1 ctx+0xb8, …); **0x9819d0(*ctx, lista)** → 0x1d0edf0(+0x38 + etapa*0x70) / 0x1d0cfd0 copia el puntero (2 dwords) al array de USER_DATA de la etapa (el SET_SH_REG lo emite Agc al draw). Un puntero nuevo por objeto.
  - draw: **0xaba900** → 0xac3580(item[4]) → `0x914360(cmd, 4 (lista de triángulos), n = *(u32*)(item[4]+0x38), inicio = *(u32*)(item[4]+0x3c))`.
  - 0x25b3a10 = nombres de cubo: 0 kRS_Depth_Occluders, 1 kRS_Depth_Normal, 2 kRS_Depth_DiscardTest, 3 kRS_Depth_Fadeout, 4 …InvLODFade, 5 kRS_GBuffer, 6 …Motion, 7-11 SkirtDecal*, 12 Standard_Normal, 13 Standard_Sky, 14-18 Alpha_SkyMinus8..4.
- CMesh (0xaa06a0): +0x298 bloque de vértices, +0x2a0 bloque de índices (registrados por MALLA-NATIVA); +8.. 14 CBuffer de 0x28; +0x238.. punteros a los CBuffer; +0x290 = puntero a la tabla de V# (en GPU) que es ctx+0xb0/tabla[0].
- Matriz del objeto en la CPU (instancia+0x38..+0x58 3x3, traslación +0x5c..+0x64, escala +0x68) existe, pero **el VS usa la del búfer de instancias (tabla[2])**.

## 2. Mediciones confirmadas (logs J:\DEMON SOULS PRUEBAS\logs, 24/25-11)
- Lista de 0xab57d0 = draws de la DCB al 100 % (coinciden=llamadas); ~1,3-3,2 M entradas/sesión; nativa_sin_skinning 33-42 %.
- VS de profundidad: etapa **ES** (NGG), gs/hs/ls = 0; mismo VS pases 0-2, otro en pase 3; exporta solo pos0 + prim (sin capa, sin instance_id).
- Estado GPU por cascada: misma imagen **2826f0000, 1024x1024, 4 capas (DB_DEPTH_VIEW slice = pase), D32**, viewport escala (512,-512,1) desplaz (512,512,0) z[0,1], recorte 0,0-1024,1024; sesgo escala -120 (pases 0-2) / -32 (pase 3), desplaz -1.1.
- La imagen la lee otro pase como textura 2D array (tipo 13), capas 0..3 (T# 02826f00 c1600000 00ffc0ff d1800924 00000003 00700000).
- Posición: tabla[0] V#[1] (base = bloque +0x298 + 0xa0 en la muestra), **stride 12, formato 74 (32_32_32_FLOAT)**; V#[0] = índice 16 bits (= CMesh+0x2a0).
- Índice del primer draw marcado coincide con el V# del CMesh de la primera entrada.

## 3. VS de profundidad (ES @26c92d600, log 20261124-211602-941, 121 dwords)

```
GPU codigo es @26c92d600: 121 dwords
GPU codigo es +0000: bfa00003 93ebff03 00080008 8700ff03 000000ff 8f6a8c6b 887c6a00 bf800000
GPU codigo es +0020: bf900009 81ea6bc0 90fe6ac1 f8000941 00000000 81ea00c0 bf8cff0f 90fe6ac1
GPU codigo es +0040: 7e060280 f40c0604 fa000000 bf8cc07f f4080c0c fa000010 f408080d fa000700
GPU codigo es +0060: f40c000d fa0006e0 f4080d0e fa000010 f4001a8d fa0002d8 f408090f fa000014
GPU codigo es +0080: f4040e0d fa0002d0 f4080a0d fa000200 f40c020d fa0001e0 f4080b0d fa000030
GPU codigo es +00a0: f40c040d fa000000 f4040e8d fa0001c0 bf8cc07f e0082000 800c0005 f4280c1a
GPU codigo es +00c0: fa000020 f42c061a fa000000 080872f9 86860624 bf8c3f70 100c0422 100a0406
GPU codigo es +00e0: 100e0402 d5410002 04184301 3e0a0205 3e0e0201 3e040020 d5410001 04160004
GPU codigo es +0100: 3e0e0000 bf8cc07f 100c0432 100a041e 1000041a d5410002 041a0231 3e0a021d
GPU codigo es +0120: 3e000219 3e040e30 3e0a0e1c 3e000e18 d5410002 81aa04ff 466a6000 40020905
GPU codigo es +0140: 466a6000 d5410000 80e200ff 466a6000 d5410002 0408d427 d51f0001 20020826
GPU codigo es +0160: d5410000 04007025 100a042a 1008040e 100c040a d5410002 04160229 d5410007
GPU codigo es +0180: 0412020d 3e0c0209 3e040028 3e0e000c 3e0c0008 d5410005 00be042e d5410004
GPU codigo es +01a0: 004e0412 d5410002 005e0416 d5410001 04160e2d 3e080e11 d5410000 040a0e15
GPU codigo es +01c0: 3e020c2c d5410002 04120c10 3e000c14 3e04023a 3e00023b f80008cf 01030002
GPU codigo es +01e0: bf810000
```

Desensamblado `llvm-mc -triple=amdgcn -mcpu=gfx1013 --disassemble` (gfx1013 sin errores; gfx1100 da 44 inválidas):

```
	.text
	s_inst_prefetch 0x3
	s_bfe_u32 vcc_hi, s3, 0x80008
	s_and_b32 s0, s3, 0xff
	s_lshl_b32 vcc_lo, vcc_hi, 12
	s_or_b32 m0, s0, vcc_lo
	s_nop 0
	s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
	s_sub_i32 vcc_lo, 64, vcc_hi
	s_lshr_b64 exec, -1, vcc_lo
	exp prim v0, off, off, off done
	s_sub_i32 vcc_lo, 64, s0
	s_waitcnt expcnt(0)
	s_lshr_b64 exec, -1, vcc_lo
	v_mov_b32_e32 v3, 0
	s_load_dwordx8 s[24:31], s[8:9], null
	s_waitcnt lgkmcnt(0)
	s_load_dwordx4 s[48:51], s[24:25], 0x10
	s_load_dwordx4 s[32:35], s[26:27], 0x700
	s_load_dwordx8 s[0:7], s[26:27], 0x6e0
	s_load_dwordx4 s[52:55], s[28:29], 0x10
	s_load_dword vcc_lo, s[26:27], 0x2d8
	s_load_dwordx4 s[36:39], s[30:31], 0x14
	s_load_dwordx2 s[56:57], s[26:27], 0x2d0
	s_load_dwordx4 s[40:43], s[26:27], 0x200
	s_load_dwordx8 s[8:15], s[26:27], 0x1e0
	s_load_dwordx4 s[44:47], s[26:27], 0x30
	s_load_dwordx8 s[16:23], s[26:27], null
	s_load_dwordx2 s[58:59], s[26:27], 0x1c0
	s_waitcnt lgkmcnt(0)
	buffer_load_format_xyz v[0:2], v5, s[48:51], 0 idxen
	s_buffer_load_dwordx4 s[48:51], s[52:55], 0x20
	s_buffer_load_dwordx8 s[24:31], s[52:55], null
	v_sub_f32_sdwa v4, s36, s57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v6, s34, v2
	v_mul_f32_e32 v5, s6, v2
	v_mul_f32_e32 v7, s2, v2
	v_mad_f32 v2, v1, s33, v6
	v_mac_f32_e32 v5, s5, v1
	v_mac_f32_e32 v7, s1, v1
	v_mac_f32_e32 v2, s32, v0
	v_mad_f32 v1, s4, v0, v5
	v_mac_f32_e32 v7, s0, v0
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v6, s50, v2
	v_mul_f32_e32 v5, s30, v2
	v_mul_f32_e32 v0, s26, v2
	v_mad_f32 v2, s49, v1, v6
	v_mac_f32_e32 v5, s29, v1
	v_mac_f32_e32 v0, s25, v1
	v_mac_f32_e32 v2, s48, v7
	v_mac_f32_e32 v5, s28, v7
	v_mac_f32_e32 v0, s24, v7
	v_mad_f32 v2, 0x466a6000, v2, -vcc_lo
	v_madmk_f32 v1, v5, 0x466a6000, v4
	v_mad_f32 v0, 0x466a6000, v0, -s56
	v_mad_f32 v2, s39, vcc_lo, v2
	v_mac_f32_e64 v1, -s38, v4
	v_mad_f32 v0, s37, s56, v0
	v_mul_f32_e32 v5, s42, v2
	v_mul_f32_e32 v4, s14, v2
	v_mul_f32_e32 v6, s10, v2
	v_mad_f32 v2, s41, v1, v5
	v_mad_f32 v7, s13, v1, v4
	v_mac_f32_e32 v6, s9, v1
	v_mac_f32_e32 v2, s40, v0
	v_mac_f32_e32 v7, s12, v0
	v_mac_f32_e32 v6, s8, v0
	v_mad_f32 v5, s46, v2, s47
	v_mad_f32 v4, s18, v2, s19
	v_mad_f32 v2, s22, v2, s23
	v_mad_f32 v1, s45, v7, v5
	v_mac_f32_e32 v4, s17, v7
	v_mad_f32 v0, s21, v7, v2
	v_mac_f32_e32 v1, s44, v6
	v_mad_f32 v2, s16, v6, v4
	v_mac_f32_e32 v0, s20, v6
	v_mac_f32_e32 v2, s58, v1
	v_mac_f32_e32 v0, s59, v1
	exp pos0 v2, v0, v3, v1 done
	s_endpgm
```

Cadena (s[8:9] = puntero a la lista de USER_DATA; s[24:31] = lista[0..3]):
- s[24:25]=lista[0] (tabla de V#): `s[48:51] = V# en +0x10` → `buffer_load_format_xyz v[0:2]` = posición (x,y,z) (v5 = id de vértice).
- s[26:27]=lista[1] (búfer de cámara, 0x720 bytes; en la GPU = la vista de la cascada copiada por 0x9131c0): carga +0x700 (s32..35), +0x6e0 (s0..7), +0x2d8 (vcc_lo), +0x2d0 (s56,s57), +0x200 (s40..43), +0x1e0 (s8..15), +0x30 (s44..47), +0x0 (s16..23), +0x1c0 (s58,s59).
- s[28:29]=lista[2]: V# en +0x10 → s_buffer_load x8 @0 (s24..31) y x4 @0x20 (s48..51) = 3x3 del objeto (filas s24-26, s28-30, s48-50).
- s[30:31]=lista[3] +0x14: s36..39.
- Operaciones:
  1. a = (s0,s1,s2)·p ; b = (s4,s5,s6)·p ; c = (s32,s33,s34)·p   (cámara +0x6e0/+0x700)
  2. objeto: u = s24*a + s25*b + s26*c ; v = s28*a + s29*b + s30*c ; w = s48*a + s49*b + s50*c
  3. w' = 15000*w − cam[+0x2d8] ; v' = 15000*v + (s36 − cam[+0x2d4]) ; u' = 15000*u − cam[+0x2d0]
     luego w' += s39*cam[+0x2d8] ; v' += −s38*(s36 − cam[+0x2d4]) ; u' += s37*cam[+0x2d0]
  4. P = (s8,s9,s10)·(u',v',w') ; Q = (s12,s13,s14)·(u',v',w') ; R = (s40,s41,s42)·(u',v',w')   (cam +0x1e0/+0x200)
  5. W = s44*P + s45*Q + s46*R + s47 (cam +0x30) ; x = s16*P + s17*Q + s18*R + s19 + s58*W ; y = s20*P + s21*Q + s22*R + s23 + s59*W   (cam +0x0 y +0x1c0)
     `exp pos0 x, y, 0, W` (z del clip = 0: la profundidad sale de W/… según la proyección; verificar con la vista real)
  (Paso 3 exacto: v4 = s36 − cam[+0x2d4]; v0 = 15000*u − cam[+0x2d0] + s37*cam[+0x2d0]; v1 = 15000*v + v4 − s38*v4; v2 = 15000*w − cam[+0x2d8] + s39*cam[+0x2d8].)

## 4. Emulador (código ya escrito)
- local/ds-nativo-sombras.h: modo (0/2/3), contadores, marca DCB de 6 dwords (MarkHeader/"SOMB"/"MARC"/pase/índice lo/hi), GpuDrawDump (registros), NoteTextureRead, GuestServeRequests (lecturas en hilo del juego).
- loader/demonsSoulsNativoSombras.inc: gancho 0xab57d0 (trampolín 12 bytes, comprobación de 32 bytes, se niega si port-r lo enganchó); recorrido por trozos SIN candado de memoria del juego y candado de MALLA-NATIVA solo para el mapa (arreglo del congelamiento 11-24, pila: HandleFault→SendCommandSync con r.mutex tomado); pide preparación a MALLA-NATIVA; volcados VISTA CASCADA (base de global+0x650), TABLA VS, camara DESPUES.
- graphicsRun.cpp: salto de la marca (antes del bloque SSRClear); en DrawIndex =2 vuelca registros; **=3: la copia se hace en el draw SIGUIENTE al primero de la cascada, reintenta cada fotograma, imprime primer éxito y fallos cada 500**.
- renderCompute.cpp `RenderExecutor::DsShadowCopyLayer(buffer, dirección, capa, &motivo)` (marco de DsNativeClear: Role/Spec/FrameCapture/SkipShader, PopPendingOperations, Envio1 + RenderLockGuard, cache.m_lock; EndRendering; Transit de la capa a TransferSrc, imagen propia VMA (formato/tamaño de la de la caché, 1 capa), vkCmdCopyImage aspecto profundidad, propia a ShaderReadOnly, Transit de vuelta a CurrentState() previo).
- render.h: `m_ds_last_depth_id`, `m_ds_last_depth_addr`; depthRenderTarget.cpp: se asignan al final de ResolveRenderDepthTarget (r.image_id de `cache.FindImage(desc)`, z_write) — así B1 usa la misma imagen que el draw (FindImageFromRange no la encontraba).
- descriptors.cpp (ResolveTextureUncached): NoteTextureRead (=2).
- cmd: J:\DEMON SOULS PRUEBAS\pruebas cmd 2\9-PROBAR-NATIVO-SOMBRAS.cmd (=2) y 9-PROBAR-NATIVO-SOMBRAS-3.cmd (=3); variable ya en las LISTAS.
- B1 aún sin validar con éxito (primer intento: "la cache no tiene la imagen"; corrección escrita, sin probar).

## 5. Pendiente (=3 completo y =1)
1. Captura en el gancho (hilo del juego, sin candados): por cascada y entrada sin skinning: bloque de vértices, V# posición (base/stride), índice (base/formato/n/inicio de item[4]+0x38/+0x3c), 12 float de tabla[2] (lista del objeto: hay que localizarla igual que el medidor TABLA VS, o reconstruir con material+0x3f4 → ctx[0x16+b]), 4 float de tabla[3]+0x14, 0x720 bytes de *(ctx+0x100), sesgo ctx+0xa0. Cola SPSC fija → GPU; llena = descartar el fotograma.
2. Shaders nuevos por el camino de gpu_tiler_shaders (*_spv.h, CMake): VS de profundidad que reproduce §3 y compute de comparación.
3. GPU: copia (B1) + dibujo en imagen propia D32 (pipeline gráfico propio siguiendo uno del emulador; dynamic rendering; viewport y sesgo de §2) + comparación → búfer host-visible, readback con fence al fotograma siguiente; línea cada 10 s por cascada.
4. Vértices/índices: MallaNativa::Lookup en el hilo de render (respeta su r.mutex; si devuelve null → abortar fotograma).
5. =1: solo si =3 ≥99,9 % en esa cascada durante la sesión; saltar en la DCB los draws cubiertos.
