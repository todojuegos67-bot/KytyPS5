## Novedades de esta versión

**Memoria nativa (por defecto)**
- La memoria del juego ahora es memoria privada normal de Windows. Ya no se usa la vista compartida que provocaba los reinicios del PC.
- El emulador ve lo que escribe el juego mediante fallos de protección de página, y copia los datos de la GPU de forma segura.
- Resultado: el juego es estable y ya no reinicia Windows.

**Sin parpadeos**
- Antes, a veces la CPU leía los datos de visibilidad de los objetos antes de que la GPU terminara de escribirlos. Ahora se espera a la GPU y los datos se devuelven antes de que la CPU los lea.
- Desaparecen los parpadeos de objetos en Boletaria.

**Rendimiento**
- Las ventanas de copia hacia la GPU se agrandan de 16–64 KiB a 1 MiB cuando el juego escribe de forma seguida, así se hacen muchas menos copias.
- En la zona de Boletaria, unos 13,8 ms por fotograma de media en una partida larga.

**Menos RAM**
- Las texturas que solo están en la GPU dejan de ocupar también RAM. Si no se tocan en 60 s, se liberan otra vez.
- Esas texturas se guardan primero en un archivo temporal junto al exe (kyty-ram17-*.tmp, máximo 8 GB, siempre deja 2 GB libres en el disco). Si el juego las vuelve a necesitar, se leen de ahí y no se pierde ninguna.
- Ahorro medido: unos 1,2 GB de RAM.

Todo esto va activado por defecto. Para desactivar una parte, pon en settings.ini, sección [Avanzado], en extra: KYTY_RAIZ=0, KYTY_PARPADEO6=0, KYTY_REND1=0, KYTY_RAM16=0 o KYTY_RAM17=0.

Sin cambios en la calidad ni en los trofeos. No incluye DLSS 5: necesita tu propio nvngx_dlssnr.dll.
