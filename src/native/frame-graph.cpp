// (f5) Frame graph: hazards between nodes in the game's order (frame-graph.h). No Vulkan here.
#include "native/frame-graph.h"

#include <unordered_map>

namespace NativeRender {

void FrameGraph::Reset() { m_nodes.clear(); }

uint32_t FrameGraph::Add(const Node& node) {
	m_nodes.push_back(node);
	return static_cast<uint32_t>(m_nodes.size() - 1);
}

namespace {
bool Writes(Access a) {
	return a == Access::Write || a == Access::ReadWrite || a == Access::RenderTarget || a == Access::DepthTarget;
}
bool Reads(Access a) { return a != Access::Write && a != Access::RenderTarget; }
} // namespace

std::vector<FrameGraph::Barrier> FrameGraph::Hazards() const {
	struct Last {
		int32_t writer = -1;
		Access  write_access = Access::Write;
		std::vector<std::pair<uint32_t, Access>> readers; // since the last write
	};
	std::unordered_map<ResourceId, Last> last;
	std::vector<Barrier>                 out;
	for (uint32_t i = 0; i < m_nodes.size(); ++i) {
		const Node& n = m_nodes[i];
		for (const ResourceUse& u: n.uses) {
			Last& l = last[u.resource];
			if (Reads(u.access) && l.writer >= 0) {
				const auto w = static_cast<uint32_t>(l.writer);
				out.push_back({w, i, u.resource, l.write_access, u.access, m_nodes[w].queue != n.queue});
			}
			if (Writes(u.access)) {
				for (const auto& r: l.readers)
					if (r.first != i) out.push_back({r.first, i, u.resource, r.second, u.access, m_nodes[r.first].queue != n.queue});
				if (l.readers.empty() && l.writer >= 0 && static_cast<uint32_t>(l.writer) != i) {
					const auto w = static_cast<uint32_t>(l.writer);
					out.push_back({w, i, u.resource, l.write_access, u.access, m_nodes[w].queue != n.queue});
				}
				l.writer       = static_cast<int32_t>(i);
				l.write_access = u.access;
				l.readers.clear();
			} else {
				l.readers.emplace_back(i, u.access);
			}
		}
	}
	return out;
}

} // namespace NativeRender
