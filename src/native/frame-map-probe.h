#pragma once
// (f5) KYTY_NATIVE_RENDER=map (frame-map-probe.cpp). The hooks are in native-render.h; this is the probe's own API.
//
// Lines in the log (every 5 s, GPU thread, at a flip):
//   [f5] <s> s, <n> frames | hilo GPU: en pasadas / fuera (DCB, ACB) ms/frame | marcas, sin registro, no casan, ajenas,
//        pila llena, fin sin inicio | coste f5 ms/frame
//   [f5] pasada id=<id> "<name>" cola=<DCB|ACB|DCB+ACB> excl=<ms> incl=<ms> ms/frame llamadas= draws= dispatches= esperas=
//   [f5] fuera cola=<q> ...           (time of a queue with no scope open)
//   [f5] orden (1 frame; ...): +id:q -id:q ...   (scope begins/ends of one frame, in translation order)
//   [f5] nombre id=<id> "<name>" eboot+0x<off> estatico|dinamico cat=<c>   (once per id)
//   [f5] RESUMEN ...                  (every 60 s: whole run, ranked by exclusive GPU-thread ms per frame)
// excl: the GPU thread's time while this pass is the innermost open scope of the queue it translates; incl: while it is
// open at any depth. Draws / dispatches / waits: PM4 packets the GPU thread counted (LiveCounters::g_pm4) in that time.
namespace NativeRender::FrameMapProbe {
void PrintSummary();
} // namespace NativeRender::FrameMapProbe
