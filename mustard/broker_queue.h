
#include <cuda_runtime.h>

#pragma once

// requires NVSHMEM atomics because updates can be remote
class BrokerWorkDistributor
{
public:
	typedef unsigned int Ticket;
	// typedef unsigned int HT_t;
	// typedef unsigned long long int HT;

	volatile Ticket *tickets;
	unsigned int *ring_buffer;
    int N;

	// HT *head_tail;
	unsigned int *head;
	unsigned int *tail;
	int *count;

    BrokerWorkDistributor(int size)
    {
        N = size;
        tickets = (volatile Ticket*) nvshmem_malloc(sizeof(Ticket) * N);
        ring_buffer = (unsigned int*) nvshmem_malloc(sizeof(unsigned int) * N);
        count = (int *) nvshmem_malloc(sizeof(int));
        head = (unsigned int*) nvshmem_malloc(sizeof(unsigned int));
        tail = (unsigned int*) nvshmem_malloc(sizeof(unsigned int));
    }

    void free_mem()
    {
        nvshmem_free((void *)tickets);
        nvshmem_free(ring_buffer);
        nvshmem_free(count);
        nvshmem_free(head);
        nvshmem_free(tail);
    }

	__device__ static __forceinline__ void backoff()
	{
		//__threadfence();
        // nvshmem_quiet();
        nvshmem_fence();
	}

	template <typename L>
	__device__ static __forceinline__ L uncachedLoad(const L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ static __forceinline__ L atomicLoad(const L* l)
	{
		return *l;
	}

	// __device__ HT_t* head(HT* head_tail, int pe)
	// {
	// 	return reinterpret_cast<HT_t*>(nvshmem_ulonglong_g(head_tail, pe)) + 1;
	// }

	// __device__ HT_t* tail(HT* head_tail, int pe)
	// {
    //     printf("In TAIL\n");
    //     HT_t* res = reinterpret_cast<HT_t*>(nvshmem_ulonglong_g(head_tail, pe));
    //     printf("TAIL done\n");
	// 	return res;
	// }

	__forceinline__ __device__ void waitForTicket(const unsigned int P, const Ticket number, int pe)
	{
		while ((nvshmem_uint_g((unsigned int *)&tickets[P], pe)) != number)
		{
			backoff(); // back off
		}
	}

	__forceinline__ __device__ bool ensureDequeue(int pe)
	{
		// int Num = atomicLoad(count);
		int Num = nvshmem_int_atomic_fetch(count, pe);
		bool ensurance = false;

		while (!ensurance && Num > 0)
		{
			if (nvshmem_int_atomic_fetch_add(count, -1, pe) > 0)
			{
				ensurance = true;
			}
			else
			{
				Num = nvshmem_int_atomic_fetch_add(count, 1, pe) + 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ bool ensureEnqueue(int pe)
	{
		// int Num = atomicLoad(count);
		int Num = nvshmem_int_atomic_fetch(count, pe);
		bool ensurance = false;

		while (!ensurance && Num < N)
		{
			if (nvshmem_int_atomic_fetch_add(count, 1, pe) < N)
			{
				ensurance = true;
			}
			else
			{
				Num = nvshmem_int_atomic_fetch_add(count, -1, pe) - 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ void readData(unsigned int& val, int pe)
	{
		const unsigned int Pos = nvshmem_uint_atomic_fetch_inc(head, pe);
		const unsigned int P = Pos % N;

		waitForTicket(P, 2 * (Pos / N) + 1, pe);
		val = nvshmem_uint_g(&ring_buffer[P], pe);
		//__threadfence();
        nvshmem_fence();
        //nvshmem_quiet();
        nvshmem_uint_p((unsigned int *)&tickets[P], 2 * ((Pos + N) / N), pe);
		//tickets[P] = 2 * ((Pos + N) / N);
	}

	__forceinline__ __device__ void putData(const unsigned int data, int pe)
	{
        // printf("In PUT for %d\n", data);
		const unsigned int Pos = nvshmem_uint_atomic_fetch_inc(tail, pe);
		const unsigned int P = Pos % N;
		const unsigned int B = 2 * (Pos / N);
        // printf("Consts assigned\n");

		waitForTicket(P, B, pe);
        // printf("Wait for ticket done\n");
        nvshmem_uint_p(&ring_buffer[P], data, pe);
        // printf("Data put\n");
		//__threadfence();
        nvshmem_fence();
        //nvshmem_quiet();
        nvshmem_uint_p((unsigned int *)&tickets[P], B + 1, pe);
		//tickets[P] = B + 1;
	}

	/* __device__ void init(int my_pe)
	{
		const int lid = threadIdx.x + blockIdx.x * blockDim.x;

		if (lid == 0)
		{
			// *count = 0;
			// *head_tail = 0x0ULL;
            nvshmem_int_p(count, 0, my_pe)
            nvshmem_longlong_p(head_tail, 0x0ULL, my_pe)
		}

		for (int v = lid; v < N; v += blockDim.x * gridDim.x)
		{
			// ring_buffer[v] = T(0x0);
			tickets[v] = 0x0;
		}
	} */

	__device__ inline bool enqueue(const unsigned int & data, int pe)
	{
        // printf("In ENQUEUE: %d\n", data);
		bool writeData = ensureEnqueue(pe);
        // printf("Ensured for %d\n", data);
		if (writeData)
		{
			putData(data, pe);
		} else {
        }
		return false;
	}

	__device__ inline void dequeue(bool& hasData, unsigned int & data, int pe)
	{
		hasData = ensureDequeue(pe);
		if (hasData)
		{
			readData(data, pe);
		}
	}

	__device__ int size(int pe) const
	{
		return nvshmem_int_atomic_fetch(count, pe);
	}
};