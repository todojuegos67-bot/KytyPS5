#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"

namespace Libs::Graphics {

hw_ctx_indirect_func_t g_hw_ctx_indirect_func[Pm4::CX_NUM] = {};
hw_sh_indirect_func_t  g_hw_sh_indirect_func[Pm4::SH_NUM]  = {};
hw_uc_indirect_func_t  g_hw_uc_indirect_func[Pm4::UC_NUM]  = {};

namespace {

constexpr auto MakeShaderDispatchTable() {
	std::array<hw_sh_parser_func_t, Pm4::SH_NUM> g_hw_sh_func {};

	for (uint32_t slot = 0; slot < 32; slot++) {
		g_hw_sh_func[Pm4::SPI_SHADER_USER_DATA_PS_0 + slot * 1] = HwShSetPsUserSgpr;
		g_hw_sh_func[Pm4::SPI_SHADER_USER_DATA_GS_0 + slot * 1] = HwShSetGsUserSgpr;
		g_hw_sh_func[Pm4::SPI_SHADER_USER_DATA_HS_0 + slot * 1] = HwShSetHsUserSgpr;
	}
	for (uint32_t slot = 0; slot < 16; slot++) {
		g_hw_sh_func[Pm4::COMPUTE_USER_DATA_0 + slot * 1] = HwShSetCsUserSgpr;
	}

	for (uint32_t slot = 0; slot < 4; slot++) {
		g_hw_sh_func[Pm4::SPI_SHADER_USER_ACCUM_PS_0 + slot] = HwShIgnoreUserAccumulator;
	}

	for (uint32_t slot = 0; slot < 4; slot++) {
		g_hw_sh_func[Pm4::SPI_SHADER_USER_ACCUM_ESGS_0 + slot] = HwShIgnoreUserAccumulator;
	}
	for (uint32_t slot = 0; slot < 4; slot++) {
		g_hw_sh_func[Pm4::SPI_SHADER_USER_ACCUM_LSHS_0 + slot] = HwShIgnoreUserAccumulator;
	}

	for (uint32_t slot = 0; slot < 4; slot++) {
		g_hw_sh_func[Pm4::COMPUTE_USER_ACCUM_0 + slot] = HwShIgnoreUserAccumulator;
	}
	g_hw_sh_func[Pm4::COMPUTE_PGM_LO]                  = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_PGM_HI]                  = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_PGM_RSRC1]               = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_PGM_RSRC2]               = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_START_X]                 = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_START_Y]                 = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_START_Z]                 = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_NUM_THREAD_X]            = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_NUM_THREAD_Y]            = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_NUM_THREAD_Z]            = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_RESOURCE_LIMITS]         = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_TMPRING_SIZE]            = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_PGM_RSRC3]               = HwShSetCsRegisters;
	g_hw_sh_func[Pm4::COMPUTE_PACE_ID]                 = HwShSetCsRegisters;

	return g_hw_sh_func;
}

constexpr auto MakeOpcodeDispatchTable() {
	std::array<cp_op_parser_func_t, 256> g_cp_op_func {};

	g_cp_op_func[Pm4::IT_NOP]                       = CpOpNop;
	g_cp_op_func[Pm4::IT_SET_BASE]                  = CpOpSetBase;
	g_cp_op_func[Pm4::IT_CLEAR_STATE]               = CpOpClearState;
	g_cp_op_func[Pm4::IT_INDEX_BASE]                = CpOpIndexBase;
	g_cp_op_func[Pm4::IT_INDEX_BUFFER_SIZE]         = CpOpIndexBufferSize;
	g_cp_op_func[Pm4::IT_DISPATCH_DIRECT]           = CpOpDispatchDirect;
	g_cp_op_func[Pm4::IT_DISPATCH_INDIRECT]         = CpOpDispatchIndirect;
	g_cp_op_func[Pm4::IT_DRAW_INDIRECT]             = CpOpDrawIndirect;
	g_cp_op_func[Pm4::IT_DRAW_INDEX_INDIRECT]       = CpOpDrawIndirect;
	g_cp_op_func[Pm4::IT_DRAW_INDIRECT_MULTI]       = CpOpDrawIndirectMulti;
	g_cp_op_func[Pm4::IT_DRAW_INDEX_INDIRECT_MULTI] = CpOpDrawIndirectMulti;
	g_cp_op_func[Pm4::IT_DRAW_INDEX_2]              = CpOpDrawIndex;
	g_cp_op_func[Pm4::IT_DRAW_INDEX_OFFSET_2]       = CpOpDrawIndexOffset;
	g_cp_op_func[Pm4::IT_DRAW_INDEX_MULTI_INST]     = CpOpDrawIndexMultiInst;
	g_cp_op_func[Pm4::IT_PFP_SYNC_ME]               = CpOpPfpSyncMe;
	g_cp_op_func[Pm4::IT_INDEX_TYPE]                = CpOpIndexType;
	g_cp_op_func[Pm4::IT_NUM_INSTANCES]             = CpOpNumInstances;
	g_cp_op_func[Pm4::IT_DRAW_INDEX_AUTO]           = CpOpDrawIndexAuto;
	g_cp_op_func[Pm4::IT_COND_EXEC]                 = CpOpCondExec;
	g_cp_op_func[Pm4::IT_SET_PREDICATION]           = CpOpSetPredication;
	g_cp_op_func[Pm4::IT_WRITE_DATA]                = CpOpWriteData;
	g_cp_op_func[Pm4::IT_WAIT_REG_MEM]              = CpOpWaitRegMem32;
	g_cp_op_func[Pm4::IT_INDIRECT_BUFFER]           = CpOpIndirectBuffer;
	g_cp_op_func[Pm4::IT_COPY_DATA]                 = CpOpCopyData;
	g_cp_op_func[Pm4::IT_EVENT_WRITE]               = CpOpEventWrite;
	g_cp_op_func[Pm4::IT_DMA_DATA]                  = CpOpDmaData;
	g_cp_op_func[Pm4::IT_ACQUIRE_MEM]               = CpOpAcquireMem;
	g_cp_op_func[Pm4::IT_RELEASE_MEM]               = CpOpReleaseMem;
	g_cp_op_func[Pm4::IT_MEM_SEMAPHORE]             = CpOpMemSemaphore;
	g_cp_op_func[Pm4::IT_REWIND]                    = CpOpRewind;
	g_cp_op_func[Pm4::IT_SET_CONTEXT_REG]           = CpOpSetContextReg;
	g_cp_op_func[Pm4::IT_SET_SH_REG]                = CpOpSetShaderReg;
	g_cp_op_func[Pm4::IT_SET_UCONFIG_REG]           = CpOpSetUconfigReg;
	g_cp_op_func[Pm4::IT_SET_UCONFIG_REG_INDEX]     = CpOpSetUconfigReg;
	g_cp_op_func[Pm4::IT_SET_CONTEXT_REG_INDIRECT]  = CpOpIndirectCxRegs;
	g_cp_op_func[Pm4::IT_SET_SH_REG_INDIRECT]       = CpOpIndirectShRegs;
	g_cp_op_func[Pm4::IT_SET_UCONFIG_REG_INDIRECT]  = CpOpIndirectUcRegs;
	g_cp_op_func[Pm4::IT_GET_LOD_STATS]             = CpOpGetLodStats;
	g_cp_op_func[Pm4::IT_WAIT_REG_MEM_64]           = CpOpWaitRegMem64;

	return g_cp_op_func;
}

constexpr auto MakeCustomOpcodeDispatchTable() {
	std::array<cp_op_parser_func_t, Pm4::R_NUM> g_cp_op_custom_func {};

	g_cp_op_custom_func[Pm4::R_DISPATCH_RESET] = CpOpDispatchReset;
	g_cp_op_custom_func[Pm4::R_WAIT_FLIP_DONE] = CpOpWaitFlipDone;
	g_cp_op_custom_func[Pm4::R_PUSH_MARKER]    = CpOpPushMarker;
	g_cp_op_custom_func[Pm4::R_POP_MARKER]     = CpOpPopMarker;
	g_cp_op_custom_func[Pm4::R_FLIP]           = CpOpFlip;
	g_cp_op_custom_func[Pm4::R_CONTEXT_STATE]  = CpOpContextState;
	g_cp_op_custom_func[Pm4::R_PSSR]           = CpOpPssr;

	return g_cp_op_custom_func;
}

} // namespace

constinit const std::array<hw_sh_parser_func_t, Pm4::SH_NUM> g_hw_sh_func =
    MakeShaderDispatchTable();
constinit const std::array<cp_op_parser_func_t, 256> g_cp_op_func = MakeOpcodeDispatchTable();
constinit const std::array<cp_op_parser_func_t, Pm4::R_NUM> g_cp_op_custom_func =
    MakeCustomOpcodeDispatchTable();

void GraphicsInitJmpTables() {
	GraphicsInitJmpTablesCxIndirect();
	GraphicsInitJmpTablesShIndirect();
	GraphicsInitJmpTablesUcIndirect();
}

} // namespace Libs::Graphics
