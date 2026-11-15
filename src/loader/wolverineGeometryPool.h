#ifndef EMULATOR_SRC_LOADER_WOLVERINEGEOMETRYPOOL_H_
#define EMULATOR_SRC_LOADER_WOLVERINEGEOMETRYPOOL_H_

namespace Loader {

struct Program;

// Marvel's Wolverine (PPSA03671) 1.05: the GeometryBufferManager's lazy commit, compiled in and off. On, the game maps its
// 2.7 GB geometry pool's 2 MiB pages as blocks land in them instead of the whole pool at boot, and the pool's sparse buffer
// (KYTY_SPARSE_GEOMETRY, on by default) has memory only there. Nothing on another title or version.
void InstallWolverineGeometryPool(Program* main_program);

} // namespace Loader

#endif /* EMULATOR_SRC_LOADER_WOLVERINEGEOMETRYPOOL_H_ */
