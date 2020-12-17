#include<stdio.h>

__global__ void farthestPointSampling(int batchSize, int nPoints, int outNPoints, 
    const float * dataset, float * temp, int * indexs, float * outPoints){
        if(outNPoints <= 0)
            return;
        
        int tid = threadIdx.x;
        __shared__ float outcomeDist[1024];
        __shared__ int outcomeIndex[1024];

        for(int i = blockIdx.x; i < batchSize; i += gridDim.x){
            int addPointIndex = 0;
            
            if(tid == 0){
                indexs[i * outNPoints] = addPointIndex;
                outPoints[(i * outNPoints) * 3] = dataset[(i * nPoints + addPointIndex) * 3];
                outPoints[(i * outNPoints) * 3 + 1] = dataset[(i * nPoints + addPointIndex) * 3 + 1];
                outPoints[(i * outNPoints) * 3 + 2] = dataset[(i * nPoints + addPointIndex) * 3 + 2]; 
            }

            //Initialize a temporary collection used to store the intermediate 
            //value of the distance between points
            for(int k = tid; k < nPoints; k += blockDim.x){
                temp[i * nPoints + k] = 1e38;
            }
                
            
            __syncthreads();
            //compute the longest distance for each turn
            for(int s = 1; s < outNPoints; s++){
                float bestDist = -1;
                int bestIndex = 0;
                float x, y, z, x_, y_, z_, dist;
                x = dataset[i * nPoints * 3 + addPointIndex * 3];
                y = dataset[i * nPoints * 3 + addPointIndex * 3 + 1];
                z = dataset[i * nPoints * 3 + addPointIndex * 3 + 2];
                for (int d = tid; d < nPoints; d += blockDim.x){
                    float priorDist = temp[i * nPoints + d];
                    x_ = dataset[i * nPoints * 3 + d * 3];
                    y_ = dataset[i * nPoints * 3 + d * 3 + 1];
                    z_ = dataset[i * nPoints * 3 + d * 3 + 2];
                    dist = (x_ - x) * (x_ - x) + (y_ - y) * (y_ - y) + (z_ - z) * (z_ - z);
                    dist = min(dist, priorDist);
                    //update the temporary collection
                    if(dist != priorDist) 
                        temp[i * nPoints + d] = dist;
                    //set the longest distance value
                    if(dist > bestDist){
                        bestDist = dist;
                        bestIndex = d;
                    }
                }
                outcomeDist[tid] = bestDist;
                outcomeIndex[tid] = bestIndex;
                // printf("finish one turn %d, bestDist: %f, bestIndex: %d\n", threadIdx.x, bestDist, bestIndex);
                __syncthreads();

                //merge outputs from all thread 
                for(int stride = blockDim.x / 2; stride > 0; stride /=2 ){
                    if(tid < stride){
                        if(outcomeDist[tid] < outcomeDist[tid + stride]){
                            outcomeDist[tid] = outcomeDist[tid + stride];
                            outcomeIndex[tid] = outcomeIndex[tid + stride];
                        }
                    }
                    __syncthreads();
                }

                addPointIndex = outcomeIndex[0];
                                
                if(tid == 0){
                    indexs[i * outNPoints + s] = addPointIndex;
                    outPoints[(i * outNPoints + s) * 3] = dataset[i * nPoints * 3 + addPointIndex * 3];
                    outPoints[(i * outNPoints + s) * 3 + 1] = dataset[i * nPoints * 3 + addPointIndex * 3 + 1];
                    outPoints[(i * outNPoints + s) * 3 + 2] = dataset[i * nPoints * 3 + addPointIndex * 3 + 2]; 

                }
            }
        }
    }

void farthestPointSamplingLauncher(int batchSize, int nPoints, int outNPoints, 
    const float * dataset, float * temp, int * indexs, float * outPoints){

        int inputSize = sizeof(float) * batchSize * nPoints * 3;
        int outputSize = sizeof(float) * outNPoints * batchSize;
        float * datasetCuda;
        float * tempCuda;
        int * indexsCuda;
        // float * outPointsCuda;

        cudaMalloc((void**)&datasetCuda, inputSize);
        cudaMalloc((void**)&tempCuda, sizeof(float) * batchSize * nPoints);
        cudaMalloc((void**)&indexsCuda, outputSize);
        // cudaMalloc((void**)&outPointsCuda, outputSize * 3);

        cudaMemcpy(datasetCuda, dataset, inputSize, cudaMemcpyHostToDevice);
        cudaMemcpy(tempCuda, temp, sizeof(float) * batchSize * nPoints, cudaMemcpyHostToDevice);

        dim3 grid(min(batchSize, 32));
        dim3 block(min(nPoints, 1024));

        farthestPointSampling<<<grid, block>>>(batchSize, nPoints, outNPoints, datasetCuda, tempCuda, indexsCuda, outPoints);
        //farthestPointSampling<<<grid, block>>>(batchSize, nPoints, outNPoints, dataset, temp, indexs);
        // cudaMemcpy(outPoints, outPointsCuda, outputSize * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(indexs, indexsCuda, outputSize, cudaMemcpyDeviceToHost);
        cudaFree(datasetCuda);
        cudaFree(tempCuda);
        cudaFree(indexsCuda);
    }


__global__ void farthestGrad(int batchSize, int nPoints, int outNPoints, const int * pointsIndexs, const float * pointsGrad, float * backGrad){
    for(int i = blockIdx.x; i < batchSize; i += gridDim.x)
        for(int j = threadIdx.x; j < nPoints; j += blockDim.x)
            backGrad[(i * nPoints + j) * 3] = 0;

    
    for(int i = blockIdx.x; i < batchSize; i += gridDim.x){
        for(int j = threadIdx.x; j < outNPoints; j += blockDim.x){
            int index = pointsIndexs[i * outNPoints + j];
            atomicAdd(&backGrad[(i * nPoints + index) * 3], pointsGrad[(i * outNPoints + j) * 3]);
            atomicAdd(&backGrad[(i * nPoints + index) * 3 + 1], pointsGrad[(i * outNPoints + j) * 3 + 1]);
            atomicAdd(&backGrad[(i * nPoints + index) * 3 + 2], pointsGrad[(i * outNPoints + j) * 3 + 2]);
        }
    }
}

void farthestPointSampleGrad(int batchSize, int nPoints, int outNPoints, const int * pointsIndexs, const float * pointsGrad, float * backGrad){
    dim3 grid(min(batchSize, 32));
    dim3 block(min(outNPoints, 1024));
    farthestGrad<<<grid, block>>>(batchSize, nPoints, outNPoints, pointsIndexs, pointsGrad, backGrad);
}
