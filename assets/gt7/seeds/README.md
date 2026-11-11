# Semillas de shaders de Gran Turismo 7 (PPSA01317), válidas para cualquier GPU

Igual que en Demon's Souls: la semilla guarda lo que **el juego** le dio al emulador (código de shader de PS5 y
estado), no el SPIR-V ya traducido. Por eso un mismo fichero sirve para cualquier gráfica: en el PC de cada
jugador el emulador traduce los shaders con **su** recompilador y opciones y crea los pipelines en **su** driver.

## Qué graba el emulador

Al jugar, `_ShaderCache\PPSA01317\guest.seeds` (acumulativo, sin duplicados) recibe un registro:

- **P (programa)** por cada shader nuevo: etapa, código guest (o el código expandido con las funciones llamadas),
  back code, user data, la descripción de entradas de la etapa (`ShaderVertexInputInfo` / `ShaderPixelInputInfo` /
  `ShaderComputeInputInfo`, sin punteros ni campos del host como el tamaño de subgrupo), la especialización de
  recursos y el cursor de push data. Su identificador es el hash de esas entradas (lo mismo que la caché de disco
  usa como clave, **sin** la parte de la sesión: fabricante/ID de GPU/opciones).
- **G (pipeline)** por cada pipeline gráfico nuevo: los programas (por su identificador), formatos de render,
  estado fijo (blend, raster, muestras...) y el layout de vértices.

Solo se escribe al crear un programa o pipeline nuevo (nunca por draw), en un hilo aparte. `KYTY_PIPELINE_SEEDS=0`
lo desactiva. Si en una partida aparece un shader que no estaba en las semillas, se compila como siempre y se
añade al fichero para la próxima vez.

## Precompilación

`kyty_emulator.exe` con `KYTY_PRECOMPILE=1` (el lanzador lo hace al pulsar JUGAR la primera vez, o tras cambiar
de driver/emulador/semillas/opciones, o con "Precompilar de nuevo"): arranca hasta crear el dispositivo Vulkan,
sin ejecutar el juego, y con todos los hilos:

1. traduce y compila cada programa a la caché de shaders en disco (`_ShaderCache\PPSA01317\*.bin`) y crea los
   pipelines de compute;
2. crea todos los pipelines gráficos en la caché de pipelines del driver;
3. guarda ambas y sale. El lanzador espera con barra de progreso y botón "Omitir" (fichero `.stop`) y solo
   después lanza el juego, así que durante la carrera no se compila nada de lo que había en las semillas.

Lee `guest.seeds` del jugador y el fichero del release (`KYTY_PRECOMPILE_SEEDS`), fusionados sin repetir.

## Generar el fichero de semillas para un release

1. Juega con el build actual (cuanto más variado, mejor: menús, varias pistas, coches, clima, repeticiones,
   Scapes...). Cierra el juego normalmente.
2. El fichero está en `C:\Gran Turismo 7 PC\_ShaderCache\PPSA01317\guest.seeds`. Cópialo con el nombre de la
   versión del juego:

   ```
   copy "C:\Gran Turismo 7 PC\_ShaderCache\PPSA01317\guest.seeds" assets\gt7\seeds\seeds-PPSA01317_01.710.000.seeds
   ```

   Para juntar semillas de varias personas basta concatenarlos (las cabeceras repetidas y los registros
   duplicados se saltan):

   ```
   copy /b a.seeds + b.seeds assets\gt7\seeds\seeds-PPSA01317_01.710.000.seeds
   ```

3. Súbelo al repositorio (Git LFS si pesa mucho) y lanza la build: CMake lo instala junto al exe y el lanzador
   lo encuentra solo (también vale en una carpeta `seeds\` al lado del exe).

## Límite

Los registros copian estructuras del emulador; si cambia su tamaño en un build nuevo (`SeedFormat()` en
`pipelineCache.cpp`), los ficheros antiguos se ignoran (el propio del jugador se aparta como `guest.seeds.old`)
y hay que volver a generar el fichero del release con ese build. Las semillas SPIR-V del build anterior
(`pipelines.seeds`, `seed-spirv\`) ya no se usan y se pueden borrar.
