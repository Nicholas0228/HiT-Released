#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cuda_runtime.h>
#include <math_constants.h>

// Stable height ordering and the original union-find/first-leaf/DFS conventions.
// Integer metadata and already-computed FP32 or FP64 linkage heights.
__device__ int root(int* parent, int x) {
    int r = x;
    while (parent[r] != r) r = parent[r];
    while (parent[x] != x) {
        int next = parent[x];
        parent[x] = r;
        x = next;
    }
    return r;
}

template <typename scalar_t>
__global__ void metadata_kernel(const scalar_t* raw, const int64_t* lengths,
                                const scalar_t* threshold, int64_t* groups,
                                int64_t* counts, int64_t* order,
                                int64_t* intervals, scalar_t* scores, int N) {
    const int b = blockIdx.x, tid = threadIdx.x, n = lengths[b];
    int capacity=1;
    while (capacity < N) capacity *= 2;
    extern __shared__ double storage[];
    scalar_t* distance = reinterpret_cast<scalar_t*>(storage);
    int* index = reinterpret_cast<int*>(distance + capacity);
    int* parent = index + capacity;
    int* accepted = parent + 2*capacity;
    int* left = accepted + 2*capacity;
    int* right = left + capacity;
    int* low = right + capacity;
    int* high = low + 2*capacity;
    if (n < 1 || n > N) {
        if (tid == 0) counts[b] = -1;
        return;
    }
    for (int i = tid; i < N; i += blockDim.x) {
        groups[b*N+i] = 0;
        order[b*N+i] = 0;
        if (i < N-1) {
            scores[b*(N-1)+i] = 0.;
            int64_t* iv = intervals + (b*(N-1)+i)*4;
            iv[0]=0; iv[1]=1; iv[2]=0; iv[3]=1;
        }
    }
    for (int i = tid; i < 2*N; i += blockDim.x) {
        parent[i]=i; accepted[i]=i; low[i]=0; high[i]=0;
    }
    int sort_size = 1;
    while (sort_size < n-1) sort_size *= 2;
    for (int i = tid; i < sort_size; i += blockDim.x) {
        distance[i] = i < n-1 ? raw[(b*(N-1)+i)*4+2] : CUDART_INF;
        index[i] = i < n-1 ? i : INT_MAX;
    }
    __syncthreads();
    // The original raw row index breaks equal-height ties, matching mergesort.
    for (int width=2; width <= sort_size; width *= 2) {
        for (int stride=width/2; stride; stride/=2) {
            for (int i=tid; i<sort_size; i+=blockDim.x) {
                int j=i^stride;
                if (j > i) {
                    scalar_t a=distance[i], d=distance[j];
                    int ai=index[i], di=index[j];
                    bool greater=a>d || (a==d && ai>di);
                    bool less=a<d || (a==d && ai<di);
                    bool ascending=(i & width)==0;
                    if ((ascending && greater) || (!ascending && less)) {
                        distance[i]=d; distance[j]=a; index[i]=di; index[j]=ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        counts[b]=0;
        scalar_t tau=*threshold;
        for (int m=0; m<n-1; ++m) {
            int raw_index=index[m];
            if (raw_index<0 || raw_index>=n-1) { counts[b]=-2; break; }
            int x=static_cast<int>(raw[(b*(N-1)+raw_index)*4]);
            int y=static_cast<int>(raw[(b*(N-1)+raw_index)*4+1]);
            if (x<0 || x>=n || y<0 || y>=n || !isfinite(distance[m])) {
                counts[b]=-2; break;
            }
            x=root(parent,x); y=root(parent,y);
            if (x==y) { counts[b]=-3; break; }
            left[m]=min(x,y); right[m]=max(x,y);
            parent[x]=n+m; parent[y]=n+m;
            scalar_t score=scalar_t(1)-distance[m];
            scores[b*(N-1)+m]=score;
            if (score >= tau) { accepted[x]=n+m; accepted[y]=n+m; }
        }
    }
    __syncthreads();
    if (counts[b] < 0) return;
    // Parallel read-only root walks avoid quadratic serial work on a long chain.
    for (int i=tid; i<n; i+=blockDim.x) {
        int r=i;
        while (accepted[r] != r) r=accepted[r];
        low[i]=r;
    }
    for (int i=tid; i<2*n-1; i+=blockDim.x) parent[i]=-1;
    __syncthreads();
    if (tid == 0) {
        int number=0;
        for (int i=0; i<n; ++i) {
            int r=low[i];
            if (parent[r] < 0) parent[r]=number++;
            groups[b*N+i]=parent[r];
        }
        counts[b]=number;
        // Reuse accepted[] as the DFS stack after all root walks have finished.
        int top=1, cursor=0;
        accepted[0]=2*n-2;
        while (top) {
            int node=accepted[--top];
            if (node < n) {
                low[node]=cursor; high[node]=cursor+1;
                order[b*N+cursor++]=node;
            } else {
                int m=node-n;
                accepted[top++]=right[m]; accepted[top++]=left[m];
            }
        }
        if (cursor != n) { counts[b]=-4; return; }
        for (int m=0; m<n-1; ++m) {
            int x=left[m], y=right[m];
            low[n+m]=min(low[x],low[y]); high[n+m]=max(high[x],high[y]);
            int64_t* iv=intervals+(b*(N-1)+m)*4;
            iv[0]=low[x]; iv[1]=high[x]; iv[2]=low[y]; iv[3]=high[y];
        }
    }
}

std::vector<torch::Tensor> metadata(torch::Tensor raw, torch::Tensor lengths,
                                  torch::Tensor threshold) {
    TORCH_CHECK(raw.is_cuda() && (raw.scalar_type()==torch::kFloat64 || raw.scalar_type()==torch::kFloat32)
                && raw.is_contiguous());
    TORCH_CHECK(raw.dim()==3 && raw.size(2)==4);
    TORCH_CHECK(lengths.is_cuda() && lengths.scalar_type()==torch::kInt64 && lengths.is_contiguous());
    TORCH_CHECK(threshold.is_cuda() && threshold.scalar_type()==raw.scalar_type() && threshold.numel()==1);
    TORCH_CHECK(raw.device()==lengths.device() && raw.device()==threshold.device());
    int B=raw.size(0), N=raw.size(1)+1;
    TORCH_CHECK(N>=1 && N<=1280 && lengths.numel()==B);
    c10::cuda::CUDAGuard guard(raw.device());
    auto integer=raw.options().dtype(torch::kInt64);
    auto groups=torch::empty({B,N},integer), counts=torch::empty({B},integer);
    auto order=torch::empty({B,N},integer), intervals=torch::empty({B,N-1,4},integer);
    auto scores=torch::empty({B,N-1},raw.options());
    // NP=2048 at N=1280: FP32 needs 96 KiB; FP64 needs 104 KiB.
    int capacity=1;
    while (capacity < N) capacity *= 2;
    AT_DISPATCH_FLOATING_TYPES(raw.scalar_type(), "hnet_metadata", [&] {
        int shared_bytes=(sizeof(scalar_t)+11*sizeof(int))*capacity;
        cudaDeviceProp properties;
        C10_CUDA_CHECK(cudaGetDeviceProperties(&properties,raw.get_device()));
        TORCH_CHECK(shared_bytes<=properties.sharedMemPerBlockOptin,
                    "HNet metadata requires ",shared_bytes," shared-memory bytes per block");
        C10_CUDA_CHECK(cudaFuncSetAttribute(metadata_kernel<scalar_t>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
        metadata_kernel<scalar_t><<<B,128,shared_bytes,at::cuda::getCurrentCUDAStream()>>>(
            raw.data_ptr<scalar_t>(),lengths.data_ptr<int64_t>(),threshold.data_ptr<scalar_t>(),
            groups.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),order.data_ptr<int64_t>(),
            intervals.data_ptr<int64_t>(),scores.data_ptr<scalar_t>(),N);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {groups,counts,order,intervals,scores};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("metadata", &metadata); }
