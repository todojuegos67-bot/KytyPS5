# Semillas de shaders de Gran Turismo 7 (PPSA01317)

Si existe `assets/gt7/seeds/PPSA01317/`, la build de Windows la instala como `seeds\PPSA01317\` dentro del
zip del release. Al pulsar JUGAR, el lanzador copia a `_ShaderCache\PPSA01317\` los ficheros que falten
(shaders traducidos `*.bin`, `seed-spirv\*.spv`) y añade los registros de `pipelines.seeds`; luego
precompila los pipelines en el driver del jugador.

## Qué graba el emulador

En cada partida, cada pipeline nuevo se apunta en `_ShaderCache\PPSA01317\`:

- `pipelines.seeds`: un registro por pipeline (estados, layout, formatos, hash de cada SPIR-V). Acumulativo.
- `seed-spirv\<hash>.spv`: el SPIR-V de cada módulo.
- `*.bin`: la caché de shaders traducidos que ya existía.

`KYTY_PIPELINE_SEEDS=0` desactiva la grabación.

## Generar el paquete

1. Juega con una carpeta limpia (`_ShaderCache` vacía) recorriendo menús, varias pistas, coches, clima,
   repeticiones, Scapes... Cuantas más partidas, más pipelines.
2. Cierra el juego (las semillas se terminan de escribir al salir).
3. Copia la carpeta entera:

   ```
   robocopy "<carpeta del juego>\_ShaderCache\PPSA01317" assets\gt7\seeds\PPSA01317 /E /XF precompile*.* seeds-merged.txt
   ```

4. Súbela al repositorio (con Git LFS si pesa: la caché real ronda 1,2 GB) y lanza la build.

## Límite importante

La traducción de shaders depende de la GPU: la clave de la caché incluye fabricante e ID de dispositivo
de la gráfica y los interruptores del recompilador (`ShaderDiskCache::SessionKey`). Cada semilla guarda
esa sesión y la precompilación salta las de otra sesión. Por eso un paquete hecho con una gráfica
concreta solo ahorra la traducción y la compilación a jugadores con el **mismo modelo de GPU** (y las
mismas opciones, p. ej. bindless). Para cubrir más GPUs, junta en el paquete semillas de varias.
