#!/usr/bin/env python3
"""Low-RAM workaround for building ggml-vulkan's generated shader TUs.

vulkan-shaders-gen emits one .comp.cpp per shader family; mul_mm.comp.cpp is
~79MB of SPIR-V byte-array initializers, which OOM-kills gcc's cc1plus on
hosts with ~2GB RAM at any optimization level.

This script splits each oversized generated TU into chunk .cpp files at
array-definition boundaries (keeping the leading #include so the extern
declarations in ggml-vulkan-shaders.hpp stay in scope), compiles each chunk
with the exact flags from compile_commands.json (forced -O0; these TUs are
only byte arrays), then partial-links the chunks with `ld -r` into the object
file path the cmake target expects. Rerun `cmake --build` afterwards; make
sees the fresh .o and skips the monolithic compile.

Usage: python3 nariq/rdna3/lowmem-build.py [build-dir]
"""
import json, os, re, subprocess, sys

BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'build-vk')
GENDIR = os.path.join(BUILD, 'ggml/src/ggml-vulkan')
OBJDIR = os.path.join(GENDIR, 'CMakeFiles/ggml-vulkan.dir')
SPLIT_IF_LARGER = 12 * 1024 * 1024   # bytes
CHUNK = 6 * 1024 * 1024

cc = json.load(open(os.path.join(BUILD, 'compile_commands.json')))

def cmd_for(fname):
    for e in cc:
        if e['file'].endswith('/' + fname):
            return e['command'].replace(' -O3 ', ' -O0 ').replace(' -DNDEBUG', '').split(' '), e['directory']
    sys.exit(f'no compile_commands entry for {fname}')

def split_build(fname):
    path = os.path.join(GENDIR, fname)
    src = open(path).read()
    m = re.search(r'^const ', src, re.M)
    header, body = src[:m.start()], src[m.start():]
    starts = [mm.start() for mm in re.finditer(r'^const ', body, re.M)]
    defs = [body[starts[i]:starts[i+1] if i+1 < len(starts) else len(body)]
            for i in range(len(starts))]
    chunks, cur, cursz = [], [], 0
    for d in defs:
        if cursz + len(d) > CHUNK and cur:
            chunks.append(cur); cur, cursz = [], 0
        cur.append(d); cursz += len(d)
    if cur:
        chunks.append(cur)
    cmd0, workdir = cmd_for(fname)
    print(f'{fname}: {len(defs)} arrays -> {len(chunks)} chunks')
    objs = []
    for i, ch in enumerate(chunks):
        cf = path.replace('.comp.cpp', f'_part{i:02d}.cpp')
        open(cf, 'w').write(header + ''.join(ch))
        of = cf[:-4] + '.o'
        c, j = [], 0
        while j < len(cmd0):
            if cmd0[j] == '-o':
                c += ['-o', of]; j += 2; continue
            if cmd0[j] == '-c':
                c += ['-c', cf]; j += 2; continue
            c.append(cmd0[j]); j += 1
        r = subprocess.run(c, cwd=workdir, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'compile failed: {of}\n{r.stderr[-2000:]}')
        objs.append(of)
    final = os.path.join(OBJDIR, fname + '.o')
    subprocess.run(['ld', '-r', '-o', final] + objs, check=True)
    os.utime(final, None)
    print(f'  linked {len(objs)} chunks -> {final}')

built = False
for f in sorted(os.listdir(GENDIR)):
    if f.endswith('.comp.cpp') and '_part' not in f:
        if os.path.getsize(os.path.join(GENDIR, f)) > SPLIT_IF_LARGER:
            split_build(f)
            built = True
if not built:
    print('no oversized generated TUs; nothing to do')
