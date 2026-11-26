# Estado del port de Demon's Souls (24/11/2026, 10:40)

Léelo entero antes de tocar nada. Código en D:\KytyPS5-build\src. Compila COMPILAR.cmd (log: D:\KytyPS5-build\build-pruebas.log).
Pruebas en J:\DEMON SOULS PRUEBAS (logs en logs\, .cmd en "pruebas cmd").
Mapa de símbolos del emulador: D:\KytyPS5-build\_Build\windows\kyty_emulator_clang_lld_link.map

## Normas del usuario
- Nada a ciegas: Ghidra (analyzeHeadless de E:\ghidra_12.2_PUBLIC) sobre el eboot ANTES de tocar algo que dependa del juego. Cada informe con apartado GHIDRA (funciones y direcciones).
- Todo a nativo, nunca parches. No descartar nada: seguir hasta que funcione perfecto.
- Una sola cosa encendida por prueba. Formato COMPILA / PRUEBA, respuestas cortas en español.
- Varios agentes en paralelo, uno por tarea, sin pararse.

## Datos clave
- eboot descifrado: E:\demons souls remake\version dem\PPSA01341-app0\eboot.bin (SELF). Código: posición en archivo = VA + 0xff50. rodata VA 0x1e7c000 = archivo 0x1e8bb80; data VA 0x2570000 = archivo 0x257f1e0. Las direcciones "Ghidra" antiguas van +0xbdb0.
- El lanzador SOBRESCRIBE settings.ini al arrancar: los valores por defecto van en el código; las pruebas se activan con los .cmd (PROBAR-x / QUITAR-x) con el lanzador cerrado.
- Gráfica simulada 8 GB: opción simular_vram_8gb en settings.ini / lanzador.
- device_commit_files a veces sube versiones viejas: usar carpetas de salida con nombre único y verificar con stage tras 15 s.

## Lo que YA funciona (encendido por defecto)
- Texturas ~95 % nativas (TEX-DOBLE y TEX-COLA en image/sparseImage.cpp).
- Render targets ~99 %. Búferes VRAM se liberan (BUF-VRAM).
- VRAM dentro de 8 GB (recolector VRAM-DESG con presupuesto real + texturas lejanas KYTY_VRAM_LEJANAS, textureCache.cpp).
- Subida repartida rehecha (SUBIDA-REP3, solo capas cambiadas).
- Memoria nativa (KYTY_NATIVE_MEMORY / RAIZ).
- Rendimiento: ~14 ms por fotograma en Boletaria, CPU ~11-13 ms (cuello: hilo de traducción, ~3200 draws/fotograma a ~3 µs; GPU al 65 %).

## En curso (apagado por defecto, se enciende con su variable)
1. MALLA-NATIVA (KYTY_MALLA_NATIVA=1; local/malla-nativa.h, bufferCache.cpp MallaPrepare, demonsSoulsCopy.cpp ganchos 0xaa06a0/0xaa1d10).
   - Ghidra: vértices en bloque malla+0x298 (de 0xaa1f70), índices en malla+0x2a0 (0xac26c0); 0xaa06a0 toma la propiedad; 0xaa1d10 libera en diferido. La CPU lee contenido por 0x909940/0x909970 (~250/s).
   - Último arreglo: reserva con vmaCreateBuffer como streamBuffer (antes 168 fallos "reserva VMA"). Lectura con ReadProcessMemory (sin cierres). Preparación bajo demanda.
   - En 8 GB no cabe (sin presupuesto). Probar con simular_vram_8gb=0. PENDIENTE DE PROBAR.
2. FRAME-NATIVO2 (KYTY_FRAME_NATIVO2=1 modo COPIA; =2 modo VIVO con C5). local/frame-nativo2.h, bufferCache.cpp, memoryAddressSpace.inc, vulkanWindow.cpp, p7-compat.h.
   - Importa los 4 arenas DefaultTagg (120 MiB, direcciones tomadas del asignador del juego). El driver NVIDIA acepta importar si la memoria está RW (sonda KYTY_UNIFICADA).
   - Modo VIVO: funcionó pero lento (C5, idle 30 ms) y fallos gráficos. Modo COPIA nuevo: copia al traducir a la anilla de subida sin C5. PENDIENTE DE PROBAR.
   - TaggedFrameAllocator: Init 0x8114a0, NextFrame ~0x811860 (idx en 0x811c27), sin espera a GPU.
3. PROT-MENOS4 (KYTY_PROT_MENOS=1; local/prot-menos.h, regionManager.h, memoryTracker.cpp, kernel/memory.cpp NativePeekWritten, videoOut.cpp NoteFlip). La v2 iba lenta (intro 27 ms) porque HotRedirty se llamaba en CADA envío y hacía un GetWriteWatch con reset por tramo, y HotHint repetía siempre. v4: una lectura sin reset por región y envío, reset solo en tramos escritos, sin vaivén de degradación, línea [prot-menos] propia cada 5 s. PENDIENTE DE COMPILAR Y PROBAR. 0xab46f0 escribe +0x174..+0x193 de cada objeto por fotograma.
4. KYTY_RAM_NATIVA2 (=2 solo mide, =1 devuelve): trozos libres del montón CPU del juego (free 0x802f90, árbol heap+0x88, candado [heap+0x80]). Sin probar.
5. Montón del emulador como SegmentHeap: línea comentada en src\port\ds-port.rc (ds-port.manifest). Sin probar.

## Pendientes de rehacer (no se descartan)
- KYTY_FLUSH_ASYNC: el hilo de render espera al lote (ganancia 0). Rehacer sin espera.
- KYTY_ARRANQUE_LIGERO: activo pero la RAM sube igual (no era Tiron6). La RAM del menú: juego 4,75 GB (VirtualMemo 1,95, cpuheap 0,85...), emulador 2,75, Windows 3,3.
- KYTY_TEX_CONFIAR_GPU: DeviceLost (descarga de texturas parciales por streaming, trozos desde +0x20000; SetCurrentStreamingChunk 0x901330).
- Hilo gráfico nativo: el juego reconstruye la DCB cada fotograma (0x921be0/0x921cb0 -> submitGraphics 0x1d19f00); buscar otra vía. KYTY_DS_OPTCULL (r_EnableOptimizedFrustumCulling, +0xce del objeto de ajustes) sin probar.

## Siguiente paso
1. Comprobar consola (device_bash) y lanzar Ghidra headless sobre el eboot.
2. Probar con el usuario, de una en una: MALLA-NATIVA (sin tope 8 GB), FRAME-NATIVO2 (modo COPIA), y lo que entregue el agente de PROT-MENOS2.

## Ghidra sin consola (24/11 10:50) — cómo se hizo
No hay device_bash. Se copian a la nube ghidra_12.2_PUBLIC_20261016.zip.part0/part1 + OpenJDK25U linux (E:\ghidra_12.2_PUBLIC) + eboot.bin,
se monta imagen plana (código VA+0xff50, rodata +0xfb80, data +0xf1e0) y analyzeHeadless con BinaryLoader x86:LE:64 -noanalysis + script Dec.java (descompila direcciones dadas).

## GHIDRA 24/11 (confirmado)
- 0xaa06a0 = creación de malla: busca bloques por FourCC con 0xaa1f50: POS0, COL0-2, TEX0-3, QUT0, BONI, BONW, SAT_. Crea hasta 14 descriptores V# de 0x30 bytes con 0x909460(...,0x800,...). Toma propiedad: bloque vértices -> malla+0x298, índices -> malla+0x2a0. Reserva 0xb0 con 0x80a4e0 (asignador 0x2ea0d18).
- 0xaa1d10 = liberar: 0x903460(asignador global [0x2f65f58]) sobre +0x298, +0x2a0 y +0x220 (si no flag +0x229&4).
- 0xaa1f70 = carga tabla de bloques (n*0x18, puntero relativo->absoluto en +0x10). 0xac26c0 = carga índices (tipo 1 -> 16 bits, 0 -> 32 bits).
- 0x909940 = base de V# (48 bits). 0x909970 = copia CPU desde V# (memcpy 0x1e73d00), tamaño = stride(dword1 bits16-29) * num_records.
- 0x8114a0 = Init TaggedFrameAllocator: 22 listas (cada 0x30 desde +0xc8), +0x88 = asignador padre, +0x94 = nº de fotogramas.
- 0x811860 = NextFrame: vacía las 22 listas, suma uso en +0x98, máximo en +0xa0, idx +0x90 = (idx+1) % [+0x94], llama vtable+0x100 del padre (devuelve memoria). Después candado giratorio [+0x40]+0x7c0 (cede con 0x8416e0 cada 10000 vueltas) y pone a 0 0x968 bytes de estadística. SIN espera a GPU.
- 0xab46f0 = actualización por objeto (escribe +0x174..+0x193).
