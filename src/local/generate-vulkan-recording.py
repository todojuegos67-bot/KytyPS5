#!/usr/bin/env python3
"""Generate conservative Vulkan thunks from this checkout's registry and dispatcher.

Every populated dispatcher slot is wrapped, including unsupported commands. Only
value arguments, known flat arrays and explicitly deep-copied structures may be
deferred. Everything else drains the producer before calling the original API.
Platform guards are copied from the actual Vulkan-Hpp dispatcher declaration.
"""
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def generate(registry, header):
    root = ET.parse(registry).getroot()
    types = {e.get('name') or e.findtext('name'): e for e in root.findall('./types/type')}
    commands = {e.get('name') or e.findtext('proto/name'): e for e in root.findall('./commands/command')}
    custom = {'VkWriteDescriptorSet', 'VkDependencyInfo', 'VkRenderingInfo'}
    def vulkan(element):
        return 'vulkan' in element.get('api', 'vulkan').split(',')

    def decl(element):
        # Registry comments are prose, not part of a C prototype.
        return ''.join(element.itertext()).split('//')[0].strip()

    def flat(name, seen=()):
        if name in seen: return False
        t = types.get(name)
        if t is None: return name != 'void'
        if t.get('alias'): return flat(t.get('alias'), (*seen, name))
        if t.get('category') not in ('struct', 'union'): return True
        return all('*' not in decl(m) and flat(m.findtext('type'), (*seen, name))
                   for m in t.findall('member') if vulkan(m))

    def copyable(name):
        t = types.get(name)
        if name in custom or flat(name): return True
        if t is None: return False
        if t.get('alias'): return copyable(t.get('alias'))
        members = [m for m in t.findall('member') if vulkan(m)]
        return (t.get('category') == 'struct' and
                any(m.findtext('name') == 'pNext' for m in members) and
                all(m.findtext('name') == 'pNext' or
                    ('*' not in decl(m) and flat(m.findtext('type'))) for m in members))

    # Descriptor writes and view destruction are ordered after the commands recorded so far.
    ORDERED_HOST_CALLS = {'vkUpdateDescriptorSets', 'vkDestroyImageView'}
    # These touch no command pool and do not depend on unreplayed commands.  A timeline
    # wait is safe because every submit is published to the worker at once (a tick
    # still being recorded could never have been waited for in the direct design).
    NO_DRAIN_CALLS = {'vkGetSemaphoreCounterValue', 'vkGetSemaphoreCounterValueKHR',
                      'vkWaitSemaphores', 'vkWaitSemaphoresKHR', 'vkCreateImageView',
                      'vkGetBufferDeviceAddress', 'vkGetBufferDeviceAddressKHR',
                      # Native XPR records allocate while the worker may still write
                      # other sets of the same pool (vkUpdateDescriptorSets needs only
                      # the destination set synchronized).
                      'vkAllocateDescriptorSets',
                      # Likewise a free: a set is freed only once the GPU passed the tick
                      # it retired at, so no unreplayed packet names it (once a frame).
                      'vkFreeDescriptorSets',
                      # New objects no queued packet can reference.
                      'vkCreateDescriptorSetLayout', 'vkCreateSampler'}
    # Live `tracem`: a GPU timestamp after each emulator-side work command (LiveTrace::VulkanMark);
    # the ids are the tags tools/local/live-trace.py marks prints by name.
    MARK_IDS = {'vkCmdDispatch': 1, 'vkCmdDispatchIndirect': 2, 'vkCmdDispatchBase': 3,
                'vkCmdCopyBuffer': 10, 'vkCmdCopyBuffer2': 11, 'vkCmdCopyImage': 12, 'vkCmdCopyImage2': 13,
                'vkCmdCopyBufferToImage': 14, 'vkCmdCopyBufferToImage2': 15,
                'vkCmdCopyImageToBuffer': 16, 'vkCmdCopyImageToBuffer2': 17,
                'vkCmdFillBuffer': 20, 'vkCmdUpdateBuffer': 21,
                'vkCmdClearColorImage': 30, 'vkCmdClearDepthStencilImage': 31, 'vkCmdClearAttachments': 32,
                'vkCmdBlitImage': 40, 'vkCmdBlitImage2': 41, 'vkCmdResolveImage': 42, 'vkCmdResolveImage2': 43,
                'vkCmdPipelineBarrier': 50, 'vkCmdPipelineBarrier2': 51}
    declarations, installs, names, deferred = [], [], [], []
    section = header.read_text().split('class DispatchLoaderDynamic :', 1)[1].split(
        'DispatchLoaderDynamic()', 1)[0]
    for line in section.splitlines():
        if line.lstrip().startswith('#'):
            declarations.append(line.strip())
            installs.append(line.strip())
            continue
        match = re.match(r'\s*PFN_(vk\w+)\s+\1\s*=\s*0;', line)
        if not match: continue
        name = match[1]
        element = commands[name]
        while element.get('alias'): element = commands[element.get('alias')]
        params = [p for p in element.findall('param') if vulkan(p)]
        pnames = [p.findtext('name') for p in params]
        ret = element.findtext('proto/type')
        copies = []
        args = []
        # Besides vkCmd*, calls that must keep their order relative to recorded commands
        # but need no result: queue them instead of draining.
        ok = (name.startswith('vkCmd') or name in ORDERED_HOST_CALLS) and ret == 'void'
        # A host allocator cannot be copied; the renderer never passes one. Such a call is
        # queued only without it (vkDestroyImageView drained the stream five times a frame).
        allocator = False
        for p in params:
            n, t, d = p.findtext('name'), p.findtext('type'), decl(p)
            if n == 'pAllocator' and t == 'VkAllocationCallbacks':
                allocator = True
                args.append('nullptr')
                continue
            if '*' not in d and '[' not in d:
                args.append(n)
                continue
            size = p.get('len')
            array = re.search(r'\[(\d+)\]', d)
            if array: size = array[1]
            # No pointer-to-pointer, output pointer, string or complex lengths.
            if ('**' in d or not d.startswith('const ') or not copyable(t) and t != 'void'):
                ok = False
            if size is None: size = '1'
            if not (size.isdigit() or size in pnames): ok = False
            if t == 'void' and size == '1': ok = False
            copies.append(f'auto copied_{n} = writer.Copy({n}, {size});')
            args.append('copied_' + n)
        names.append(name)
        mark_id = None
        mark = (f'LiveTrace::VulkanMark(commandBuffer, {mark_id}, {"true" if mark_id < 10 else "false"}); '
                if mark_id is not None and ret == 'void' else '')
        declarations.append(f'static VKAPI_ATTR {ret} VKAPI_CALL Wrapped_{name}({", ".join(decl(p) for p in params)}) {{')
        # Commands that do GPU work or synchronize: unchanged between two barriers = none in between.
        if re.match(r'vkCmd(Draw|Dispatch|Copy|Fill|Clear|Blit|Resolve|UpdateBuffer|BeginRendering|PipelineBarrier|WriteTimestamp|ExecuteCommands)', name):
            declarations.append('    ++g_work_calls;')
        # Commands that change memory or images (not barriers, render pass begins, timestamps).
        if re.match(r'vkCmd(Draw|Dispatch|Copy|Fill|Clear|Blit|Resolve|UpdateBuffer|ExecuteCommands)', name):
            declarations.append('    ++g_writing_calls;')
        if ok:
            deferred.append(name)
            declarations.append('    if (auto* stream = pAllocator == nullptr ? RecordingStream() : nullptr) {'
                                if allocator else '    if (auto* stream = RecordingStream()) {')
            declarations.append('        if (stream->Enqueue([&](Writer& writer) {')
            declarations.extend('            ' + c for c in copies)
            mutates_state = (name.startswith('vkCmdSet') or name.startswith('vkCmdBindPipeline') or
                             name.startswith('vkCmdBindShaders') or name.startswith('vkCmdExecute') or
                             name.startswith('vkCmdBeginRenderPass') or name.startswith('vkCmdNextSubpass'))
            invalidate = 'InvalidateRawState(); ' if mutates_state else ''
            declarations.append(f'            writer.Command([=] {{ {invalidate}original.{name}({", ".join(args)}); }});')
            declarations.append(f'        }})) {{ {mark}return; }}')
            declarations.append('    }')
        # A progress query observes previously submitted work and neither
        # accesses a command pool nor executes commands still being recorded.
        # Preserve the real query; do not wait for our CPU recording queue.
        if name not in NO_DRAIN_CALLS:
            declarations.append('    BeforeDirect();')
        if mark:
            declarations.append(f'    original.{name}({", ".join(pnames)});')
            declarations.append(f'    {mark}')
        else:
            declarations.append(f'    return original.{name}({", ".join(pnames)});')
        declarations.append('}')
        installs.append(f'if (original.{name}) dispatcher.{name} = Wrapped_{name};')
    return ('// Generated; edit generate-vulkan-recording.py, not this file.\n' +
            '\n'.join(declarations) + '\nvoid InstallDispatch() {\n' +
            'auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;\noriginal = dispatcher;\n' +
            '\n'.join(installs) + '\n}\n'), names, deferred


if __name__ == '__main__':
    result, names, deferred = generate(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(result)
    print(f'Vulkan recording: {len(deferred)} copied commands, {len(names)} guarded dispatcher entries')
