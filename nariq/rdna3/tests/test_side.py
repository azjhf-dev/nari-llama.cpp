"""Independent dense FP64 transform oracle for isolated GLSL side kernels."""
from pathlib import Path
import numpy as np
from test_product import execute


def matrix(length):
    m=next(m for m in (1,12,20,68) if length%m==0 and (length//m)&(length//m-1)==0)
    run=length//m
    h=np.array([[(-1)**((i&j).bit_count()) for j in range(run)] for i in range(run)],dtype=np.float64)
    if m==1: return h/np.sqrt(length)
    residues={i*i%(m-1) for i in range(1,m-1)}
    paley=np.array([[1 if r==c or r==0 else -1 if c==0 else 1 if (c-r)%(m-1) in residues else -1 for c in range(m)] for r in range(m)])
    return np.kron(paley,h)/np.sqrt(length)


def run_sides(shader_dir):
    shader_dir=Path(shader_dir)
    results=[]
    rng=np.random.default_rng(7411)
    # Identity, block Sylvester and every Paley factor. Fused parts and shards.
    cases=[(64,0,0,1,[]),(128,1,16,1,[]),(128,1,64,1,[]),
           (64,2,0,1,[]),(192,2,0,1,[]),(320,2,0,1,[]),
           (1088,2,0,1,[]),(384,2,0,2,[128,256]),(640,2,0,2,[256,384])]
    for n,kind,block,shards,parts in cases:
        for inverse in (0,1):
            x=rng.uniform(-1,1,(3,n)).astype(np.float32)
            config=np.array([kind,block,shards,inverse,len(parts)]+parts,dtype=np.uint32)
            actual,env=execute((shader_dir/'side.spv').read_bytes(),[config.tobytes(),bytes(4),x.tobytes()],x.nbytes,[n,n,0,3])
            expected=x.astype(np.float64).copy()
            if kind==1:
                mat=matrix(block)
                expected=(x.reshape(-1,block).astype(np.float64)@mat.T).reshape(x.shape)
            elif kind==2:
                mat=matrix(n//shards)
                if inverse: mat=mat.T
                sizes=parts or [n]
                offsets=np.cumsum([0]+sizes)
                for shard in range(shards):
                    indices=np.concatenate([np.arange(offsets[p]+shard*(size//shards),offsets[p]+(shard+1)*(size//shards)) for p,size in enumerate(sizes)])
                    expected[:,indices]=x[:,indices].astype(np.float64)@mat.T
            error=float(np.max(np.abs(actual.reshape(3,n)-expected)))
            np.testing.assert_allclose(actual.reshape(3,n),expected,rtol=5e-5,atol=5e-6)
            results.append({'n':n,'kind':kind,'block':block,'shards':shards,'parts':parts,'inverse':inverse,'max_abs_error':error})
    for mode in (0,1):
        n=67
        perm=rng.permutation(n).astype(np.uint32)
        scale=rng.uniform(-2,2,n).astype(np.float32)
        x=rng.uniform(-1,1,(3,n)).astype(np.float32)
        expected=x[:,perm]*(scale if mode==0 else scale[perm])
        actual,env=execute((shader_dir/'gather.spv').read_bytes(),[perm.tobytes(),scale.tobytes(),x.tobytes()],x.nbytes,[n,n,mode,3])
        np.testing.assert_array_equal(actual.reshape(3,n),expected)
        results.append({'gather_mode':mode,'n':n,'exact':True})
    return results


def run_composed(shader_dir):
    """Compose all shader stages and compare to native CPU dense dequantizer."""
    import ctypes
    import os
    library=os.environ.get('NARIQ_TEST_LIBRARY')
    if not library:
        return {'status':'skipped','reason':'Set NARIQ_TEST_LIBRARY to built libggml-base for native composed check'}
    lib=ctypes.CDLL(str(Path(library).resolve()))
    ptr=ctypes.c_void_p
    lib.ggml_trellis_dequantize.argtypes=[ctypes.c_int,ptr,ctypes.c_int64,ctypes.c_int64,ptr,ptr,ptr,ptr,ptr,ptr,ctypes.c_int,ptr]
    lib.ggml_trellis_dequantize.restype=ctypes.c_bool
    shader_dir=Path(shader_dir)
    reports=[]
    for bits in range(1,7):
        K,N,batch=192,320,3
        rng=np.random.default_rng(9350+bits)
        codes=rng.integers(0,2**32,N*K*bits//32,dtype=np.uint32)
        scales=rng.uniform(.01,.2,K//16).astype(np.float16)
        pre=rng.uniform(-2,2,K).astype(np.float32)
        post=rng.uniform(-2,2,N).astype(np.float32)
        pin=rng.permutation(K).astype(np.int32)
        pout=rng.permutation(N).astype(np.int32)
        desc=np.array([2,0,1,2,0,1,2,128,192],dtype=np.int32)
        dense=np.empty((N,K),dtype=np.float32)
        assert lib.ggml_trellis_dequantize(43+bits,codes.ctypes.data,K,N,scales.ctypes.data,pre.ctypes.data,post.ctypes.data,pin.ctypes.data,pout.ctypes.data,desc.ctypes.data,len(desc),dense.ctypes.data)
        x=rng.uniform(-1,1,(batch,K)).astype(np.float32)
        prepared,_=execute((shader_dir/'gather.spv').read_bytes(),[pin.tobytes(),pre.tobytes(),x.tobytes()],x.nbytes,[K,K,0,batch])
        cfg=np.array([2,0,1,0,0],dtype=np.uint32)
        rotated,_=execute((shader_dir/'side.spv').read_bytes(),[cfg.tobytes(),bytes(4),prepared.tobytes()],x.nbytes,[K,K,0,batch])
        product,_=execute((shader_dir/'product.spv').read_bytes(),[codes.tobytes(),scales.tobytes(),rotated.tobytes()],N*batch*4,[K,N,bits,batch])
        cfg=np.array([2,0,1,1,2,128,192],dtype=np.uint32)
        unrotated,_=execute((shader_dir/'side.spv').read_bytes(),[cfg.tobytes(),bytes(4),product.tobytes()],N*batch*4,[N,N,0,batch])
        out,_=execute((shader_dir/'gather.spv').read_bytes(),[pout.tobytes(),post.tobytes(),unrotated.tobytes()],N*batch*4,[N,N,1,batch])
        expected=x.astype(np.float64)@dense.astype(np.float64).T
        error=float(np.max(np.abs(out.reshape(batch,N)-expected)))
        np.testing.assert_allclose(out.reshape(batch,N),expected,rtol=2e-4,atol=2e-4)
        reports.append({'bits':bits,'K':K,'N':N,'batch':batch,'max_abs_error':error})
    return {'status':'passed','cases':reports,'oracle':'native CPU ggml_trellis_dequantize plus FP64 product'}
